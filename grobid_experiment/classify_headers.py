#!/usr/bin/env python3

"""
Header classification experiment using pre-extracted headers.
Loads headers from individual PDF JSON files and tries different classification methods.
"""

import argparse
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz

from transformers import pipeline as hf_pipeline
from datasets import Dataset

# Combined lexicon for all target chapter types
TARGET_LEX = {
    # CONCLUSION
    "conclusion", "conclusions", "concluding remarks", "final remarks", "general conclusions",
    "general conclusion", "discussion and conclusion", "discussion and conclusions",
    "summary and conclusion", "summary and conclusions", "concluding thoughts", 
    "final thoughts", "closing remarks",
    
    # FUTURE_WORK  
    "future work", "future direction", "future directions", "future research",
    "prospects", "limitations and future work", "conclusions and future work",
    "discussion and future work", "future studies", "further research", 
    "next steps", "outlook", "directions for future research",
    
    # SUMMARY
    "summary", "summary of findings", "executive summary", "abstract of findings",
    
    # DISCUSSION
    "discussion", "general discussion", "discussion and interpretation",
    
    # RECOMMENDATIONS
    "recommendations", "recommendations for practice", "recommendations for policy",
    "practical implications",
    
    # LIMITATIONS
    "limitations", "limitations and future work",
    
    # IMPLICATIONS
    "implications", "theoretical implications", "practical implications"
}

def normalize_header_text(s: str) -> str:
    """Normalize header text for comparison"""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\-&]", " ", s.lower())).strip()

def generate_output_filename(method: str, nli_model: Optional[str] = None, 
                           fuzzy_thresh: float = 70.0, use_partial_matching: bool = False,
                           batch_size: int = 16, base_name: str = "classified_results") -> str:
    """Generate descriptive filename based on classification parameters"""
    parts = [base_name, method]
    
    if method == "lexical":
        # Add fuzzy threshold
        parts.append(f"fuzzy{int(fuzzy_thresh)}")
        
        # Add partial matching if enabled
        if use_partial_matching:
            parts.append("partial")
    
    elif method == "nli":
        # Add model name (simplified)
        if nli_model:
            model_short = nli_model.split('/')[-1].replace('-', '').replace('_', '')
            parts.append(model_short)
        
        # Add batch size if not default
        if batch_size != 16:
            parts.append(f"bs{batch_size}")
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parts.append(timestamp)
    
    return "_".join(parts) + ".json"

def save_metrics_to_csv(metrics: Dict[str, Any], method: str, parameters: Dict[str, Any], 
                       csv_file: Path = Path("classification_metrics.csv")):
    """Save metrics to CSV file for comparison"""
    
    # Prepare row data
    row_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "total_predictions": metrics["total_predictions"],
        "total_gold_pages": metrics["total_gold_pages"]
    }
    
    # Add method-specific parameters
    if method == "lexical":
        row_data.update({
            "fuzzy_thresh": parameters.get("fuzzy_thresh", 70.0),
            "use_partial_matching": parameters.get("use_partial_matching", False),
            "nli_model": "",
            "batch_size": ""
        })
    elif method == "nli":
        row_data.update({
            "fuzzy_thresh": "",
            "use_partial_matching": "",
            "nli_model": parameters.get("nli_model", ""),
            "batch_size": parameters.get("batch_size", 16)
        })
    
    # Convert to DataFrame
    df_new = pd.DataFrame([row_data])
    
    # Append to existing CSV or create new one
    if csv_file.exists():
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Save to CSV
    df_combined.to_csv(csv_file, index=False)
    print(f"Metrics saved to {csv_file}")
    
    return csv_file

def lexical_classify(text: str, fuzzy_thresh: float = 70.0, use_partial_matching: bool = False) -> tuple:
    """Binary classification using fuzzy string matching against combined lexicon
    Returns: (label, confidence)
    """
    norm_text = normalize_header_text(text)
    
    # Option 3: Partial matching (if enabled)
    if use_partial_matching:
        # Check if any lexicon term is a substring of the header
        for lexicon_term in TARGET_LEX:
            if lexicon_term in norm_text or norm_text in lexicon_term:
                # Use fuzzy score for confidence, but ensure it's at least above threshold
                best_score = max([fuzz.ratio(norm_text, lx) for lx in TARGET_LEX])
                return "Relevant", max(best_score / 100.0, (fuzzy_thresh + 1) / 100.0)
    
    # Standard fuzzy matching
    target_scores = [fuzz.ratio(norm_text, lx) for lx in TARGET_LEX]
    best_score = max(target_scores) if target_scores else 0
    
    # Binary classification: Relevant or Not Relevant
    if best_score >= fuzzy_thresh:
        return "Relevant", best_score / 100.0
    else:
        return "Not Relevant", best_score / 100.0


def batch_nli_classify_single_gpu(zs, header_texts: List[str], batch_size=16):
    """Binary NLI classification using datasets on a single GPU"""
    if not header_texts:
        return []
    
    # Create dataset
    dataset = Dataset.from_dict({
        "text": header_texts,
        "candidate_labels": [["a conclusion, summary, discussion, future work, recommendations, limitations, or implications section", "an introduction, methodology, or random text"]] * len(header_texts)
    })
    
    def process_batch(batch):
        results = zs(batch["text"], 
                    candidate_labels=batch["candidate_labels"][0],
                    hypothesis_template="This header is a chapter title about {}.")
        
        if isinstance(results, dict):
            results = [results]
        
        # Map back to simple labels with threshold
        mapped_labels = []
        for r in results:
            if (r["labels"][0] == "a conclusion, summary, discussion, future work, recommendations, limitations, or implications section" 
                and r["scores"][0] >= 0.8):
                mapped_labels.append("Relevant")
            else:
                mapped_labels.append("Not Relevant")
        
        return {
            "predicted_label": mapped_labels,
            "confidence": [float(r["scores"][0]) for r in results]
        }
    
    classified = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text", "candidate_labels"]
    )
    
    return list(zip(classified["predicted_label"], classified["confidence"]))

def process_nli_chunk(chunk_data):
    """Process a chunk of headers on a specific GPU"""
    texts, gpu_id, model_name, batch_size = chunk_data
    
    import torch
    from transformers import pipeline as hf_pipeline
    
    # Set GPU for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    
    # Initialize model for this GPU
    zs = hf_pipeline("zero-shot-classification", model=model_name, device=device)
    
    # Process this chunk
    results = batch_nli_classify_single_gpu(zs, texts, batch_size)
    
    return results, len(texts), gpu_id

def batch_nli_classify_multi_gpu(header_texts: List[str], nli_model: str, num_gpus=6, batch_size=16):
    """Distribute NLI classification across multiple GPUs"""
    if not header_texts:
        return []
    
    import torch
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Check available GPUs
    available_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    print(f"Using {available_gpus} GPUs for NLI classification")
    
    # Split texts across GPUs
    chunks = []
    chunk_size = len(header_texts) // available_gpus
    for i in range(available_gpus):
        start_idx = i * chunk_size
        if i == available_gpus - 1:
            # Last chunk gets remaining items
            end_idx = len(header_texts)
        else:
            end_idx = start_idx + chunk_size
        
        if start_idx < len(header_texts):
            chunks.append((header_texts[start_idx:end_idx], i, nli_model, batch_size))
    
    # Process chunks in parallel
    all_results = []
    chunk_results = [None] * len(chunks)
    
    with ProcessPoolExecutor(max_workers=available_gpus) as executor:
        future_to_chunk = {executor.submit(process_nli_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                results, count, gpu_id = future.result()
                chunk_results[chunk_idx] = results
                print(f"GPU {gpu_id} processed {count} headers")
            except Exception as exc:
                print(f"Chunk {chunk_idx} generated an exception: {exc}")
                chunk_results[chunk_idx] = []
    
    # Combine results in original order
    for chunk_result in chunk_results:
        if chunk_result:
            all_results.extend(chunk_result)
    
    return all_results


def load_pdf_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all PDF results from JSON files"""
    pdf_results = []
    json_files = list(results_dir.glob("*.json"))
    
    print(f"Loading {len(json_files)} PDF result files...")
    
    for json_file in tqdm(json_files, desc="Loading PDFs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert new format to expected format
            all_predictions = []
            for page_data in data.get('pages', []):
                page_num = page_data['page_number']
                headers = page_data.get('detected_headers', [])
                
                for header in headers:
                    prediction = {
                        'text': header.get('text', ''),
                        'page': page_num,
                        'confidence': header.get('confidence', 0.0),
                        'bbox': header.get('bbox', []),
                        'word_count': header.get('word_count', 1)
                    }
                    all_predictions.append(prediction)
            
            # Create standardized format
            pdf_result = {
                'doc_id': data.get('doc_id', json_file.stem + '.pdf'),
                'full_path': data.get('full_path', ''),
                'total_pages': data.get('total_pages', 0),
                'all_predictions': all_predictions,
                'processing_info': data.get('processing_info', {})
            }
            
            pdf_results.append(pdf_result)
                
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    return pdf_results

def reclassify_headers(pdf_results: List[Dict[str, Any]], 
                      method: str = "lexical",
                      nli_model: Optional[str] = None,
                      batch_size: int = 16,
                      verbose: bool = False,
                      num_gpus: int = 6,
                      use_partial_matching: bool = False,
                      fuzzy_thresh: float = 70.0) -> List[Dict[str, Any]]:
    """Reclassify headers using different methods"""
    
    # Collect all headers for batch processing if using NLI
    all_headers = []
    header_to_pdf = {}
    nli_results = []
    
    if method == "nli":
        if not nli_model:
            raise ValueError("NLI model required for NLI method")
        
        for pdf_idx, pdf_data in enumerate(pdf_results):
            for header_idx, header in enumerate(pdf_data["all_predictions"]):
                all_headers.append(header["text"])
                header_to_pdf[len(all_headers) - 1] = (pdf_idx, header_idx)
        
        print(f"Batch processing {len(all_headers)} headers with NLI using {num_gpus} GPUs...")
        nli_results = batch_nli_classify_multi_gpu(all_headers, nli_model, num_gpus=num_gpus, batch_size=batch_size)
    
    # Process each PDF
    updated_results = []
    nli_idx = 0
    
    for pdf_data in tqdm(pdf_results, desc=f"Classifying with {method}"):
        updated_pdf = pdf_data.copy()
        updated_headers = []
        
        if verbose:
            print(f"\nProcessing {pdf_data['doc_id']}...")
        
        for header in pdf_data["all_predictions"]:
            text = header["text"]
            
            if method == "lexical":
                label, score = lexical_classify(text, fuzzy_thresh=fuzzy_thresh, use_partial_matching=use_partial_matching)
                updated_header = header.copy()
                updated_header.update({
                    "label": label,
                    "label_score": score,
                    "classification_method": "lexical"
                })
                updated_headers.append(updated_header)
                
            elif method == "nli":
                label, score = nli_results[nli_idx]
                nli_idx += 1
                updated_header = header.copy()
                updated_header.update({
                    "label": label,
                    "label_score": score,
                    "classification_method": "nli"
                })
                updated_headers.append(updated_header)
            
            if verbose:
                print(f"  Page {header['page']}: '{text[:50]}...' -> {updated_headers[-1]['label']} ({updated_headers[-1]['label_score']:.3f})")
        
        # Update PDF data
        updated_pdf["all_predictions"] = updated_headers
        
        # Count relevant predictions
        relevant_candidates = [h for h in updated_headers if h["label"] == "Relevant"]
        not_relevant_candidates = [h for h in updated_headers if h["label"] == "Not Relevant"]
        
        updated_pdf["relevant_candidates"] = len(relevant_candidates)
        updated_pdf["not_relevant_candidates"] = len(not_relevant_candidates)
        
        updated_results.append(updated_pdf)
    
    return updated_results

def evaluate_predictions(results: List[Dict[str, Any]], gold_csv: Path) -> Dict[str, Any]:
    """Evaluate predictions at PAGE level - each detected page is a prediction"""
    # Load ground truth data from filtered CSV
    import pandas as pd
    
    df = pd.read_csv(gold_csv)
    
    # Create mapping from full_path to ground truth entries
    gold_by_path = {}
    gold_by_doc_id = {}
    
    for _, row in df.iterrows():
        full_path = row['full_path']
        page_num = row['page_number']
        label = row['label']
        
        # Map by full path
        if full_path not in gold_by_path:
            gold_by_path[full_path] = []
        gold_by_path[full_path].append({
            'page_number': page_num,
            'label': label,
            'chapter_title': row.get('chapter_title', '')
        })
        
        # Also map by doc_id for backwards compatibility
        doc_id = Path(full_path).name
        if doc_id not in gold_by_doc_id:
            gold_by_doc_id[doc_id] = []
        gold_by_doc_id[doc_id].append({
            'page_number': page_num,
            'label': label,
            'chapter_title': row.get('chapter_title', '')
        })
    
    # Binary evaluation: all target labels vs predictions
    true_positives = 0   
    false_positives = 0  
    false_negatives = 0  
    
    # Track all predictions and ground truth
    all_predictions = set()  # (doc_id, page_num) tuples of predicted relevant pages
    all_gold_pages = set()   # (doc_id, page_num) tuples of actual target pages
    
    # All target labels from ground truth
    target_labels = ["CONCLUSION", "FUTURE_WORK", "SUMMARY", "DISCUSSION", "RECOMMENDATIONS", "LIMITATIONS", "IMPLICATIONS"]
    
    # Collect all "Relevant" predictions
    for result in results:
        doc_id = result["doc_id"]
        
        # Get all headers classified as "Relevant"
        relevant_headers = [h for h in result.get("all_predictions", []) 
                          if h.get("label", "") == "Relevant"]
        
        # Each detected header represents a page-level prediction
        for header in relevant_headers:
            page_num = header["page"]
            all_predictions.add((doc_id, page_num))
    
    # Collect ground truth pages (any page with target labels)
    for result in results:
        doc_id = result["doc_id"]
        full_path = result.get("full_path", "")
        
        # Try to find ground truth by full_path first, then by doc_id
        gold_entries = gold_by_path.get(full_path, gold_by_doc_id.get(doc_id, []))
        
        for entry in gold_entries:
            # Check if this entry has any of our target labels
            label = entry['label']
            # Handle cases where label might be NaN or not a string
            if isinstance(label, str) and any(target_label in label for target_label in target_labels):
                page_num = entry['page_number']
                all_gold_pages.add((doc_id, page_num))
    
    # Calculate TP, FP, FN
    true_positives = len(all_predictions.intersection(all_gold_pages))
    false_positives = len(all_predictions - all_gold_pages)  
    false_negatives = len(all_gold_pages - all_predictions)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives, 
        "false_negatives": false_negatives,
        "total_predictions": len(all_predictions),
        "total_gold_pages": len(all_gold_pages)
    }

def main():
    parser = argparse.ArgumentParser(description="Reclassify pre-extracted headers")
    parser.add_argument("--results_dir", default="header_results", help="Directory with PDF JSON files")
    parser.add_argument("--gold", default="../final_groundtruth_filtered.csv", help="CSV file with ground truth")
    parser.add_argument("--method", choices=["lexical", "nli"], 
                        default="lexical", help="Classification method")
    parser.add_argument("--nli_model", default="microsoft/deberta-large-mnli",
                        help="NLI model (for nli method)")
    parser.add_argument("--output", help="Output JSON file with reclassified results (auto-generated if not specified)")
    parser.add_argument("--csv_metrics", default="classification_metrics.csv", help="CSV file to save metrics for comparison")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for NLI")
    parser.add_argument("--num_gpus", type=int, default=6, help="Number of GPUs to use for NLI")
    parser.add_argument("--use_partial_matching", action="store_true", help="Enable partial matching for lexical classification")
    parser.add_argument("--fuzzy_thresh", type=float, default=70.0, help="Fuzzy matching threshold for lexical classification (default: 70.0)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load existing PDF results
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        sys.exit(1)
    
    pdf_results = load_pdf_results(results_dir)
    if not pdf_results:
        print("No PDF results found!")
        sys.exit(1)
    
    print(f"Loaded {len(pdf_results)} PDF results")
    
    # Reclassify headers
    print(f"Reclassifying headers using method: {args.method}")
    updated_results = reclassify_headers(
        pdf_results, 
        method=args.method,
        nli_model=args.nli_model if args.method == "nli" else None,
        batch_size=args.batch_size,
        verbose=args.verbose,
        num_gpus=args.num_gpus,
        use_partial_matching=args.use_partial_matching,
        fuzzy_thresh=args.fuzzy_thresh
    )
    
    # Generate output filename if not specified
    if not args.output:
        args.output = generate_output_filename(
            method=args.method,
            nli_model=args.nli_model if args.method == "nli" else None,
            fuzzy_thresh=args.fuzzy_thresh,
            use_partial_matching=args.use_partial_matching,
            batch_size=args.batch_size
        )
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(updated_results, f, indent=2, ensure_ascii=False)
    print(f"Saved reclassified results to {args.output}")
    
    # Evaluate and save metrics
    if args.gold:
        print("\nEvaluating predictions...")
        metrics = evaluate_predictions(updated_results, Path(args.gold))
        
        # Prepare parameters for CSV export
        parameters = {
            "fuzzy_thresh": args.fuzzy_thresh,
            "use_partial_matching": args.use_partial_matching,
            "nli_model": args.nli_model,
            "batch_size": args.batch_size,
            "output_file": args.output
        }
        
        # Save metrics to CSV
        csv_file = save_metrics_to_csv(
            metrics=metrics,
            method=args.method,
            parameters=parameters,
            csv_file=Path(args.csv_metrics)
        )
        
        print(f"\n=== BINARY EVALUATION RESULTS ({args.method.upper()}) ===")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Total Ground Truth Pages: {metrics['total_gold_pages']}")
        print(f"\nMetrics comparison available in: {csv_file}")

if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method may already be set
        pass
    
    main()