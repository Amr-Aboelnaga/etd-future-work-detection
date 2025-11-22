#!/usr/bin/env python3

"""
LLM-based header classification experiment.
Prompts an LLM to classify detected headers as Conclusion, Future Work, or Other.
"""

import argparse
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from tqdm import tqdm
import time

# For Ollama
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

def create_classification_prompt(header_text: str, doc_context: str = "") -> str:
    """Create a prompt for LLM binary header classification"""
    prompt = f"""You are analyzing academic document headers to classify their section types.

Header text: "{header_text}"

Please classify this header into ONE of these categories:
1. Relevant - Headers for conclusions, future work, summaries, discussions, recommendations, limitations, or implications sections
2. Not Relevant - Any other type of header (introduction, methodology, results, literature review, etc.)

TARGET SECTION TYPES (classify as "Relevant"):
- Conclusion sections, final remarks, closing thoughts, concluding remarks
- Future work, future directions, future research, next steps, prospects
- Summary sections, executive summaries, abstracts of findings
- Discussion sections, general discussion, interpretation of results
- Recommendations, practical implications, policy recommendations
- Limitations, study limitations, constraints
- Implications, theoretical implications, practical implications

Consider typical academic document structure and common header patterns.

Respond with ONLY the category name (Relevant or Not Relevant) followed by a confidence score from 0.0 to 1.0.

NOTE: WE ARE ONLY INTERESTED IN CHAPTER HEADERS OR FULL SECTION HEADERS not SUBSECTION HEADERS
Format: [Category] [Confidence]
Example: Relevant 0.85

Response:"""

    return prompt

def parse_llm_response(response: str) -> Tuple[str, float]:
    """Parse LLM response to extract category and confidence for binary classification"""
    response = response.strip()
    
    # Try to extract category and confidence
    # Look for patterns like "Relevant 0.85" or "Not Relevant 0.9"
    patterns = [
        r'^(Relevant|Not Relevant)\s+(0?\.\d+|1\.0|0|1)$',
        r'^(Relevant|Not Relevant)\s*:\s*(0?\.\d+|1\.0|0|1)$',
        r'^(Relevant|Not Relevant)\s*-\s*(0?\.\d+|1\.0|0|1)$'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, response, re.IGNORECASE)
        if match:
            category = match.group(1).title()
            if category == "Not Relevant":
                category = "Not Relevant"  # Preserve exact casing
            confidence = float(match.group(2))
            return category, confidence
    
    # Fallback: look for relevant indicators
    response_lower = response.lower()
    relevant_indicators = [
        "relevant", "conclusion", "future work", "summary", "discussion",
        "recommendation", "limitation", "implication"
    ]
    
    if any(indicator in response_lower for indicator in relevant_indicators):
        if "not relevant" in response_lower:
            return "Not Relevant", 0.5
        else:
            return "Relevant", 0.5
    else:
        return "Not Relevant", 0.5

class OllamaClassifier:
    """Ollama-based classifier"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://127.0.0.1:11434"):
        if not HAS_REQUESTS:
            raise ImportError("requests package not installed. Run: pip install requests")
        
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {base_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}: {e}")
    
    def classify_single(self, header_text: str) -> Tuple[str, float]:
        """Classify a single header using Ollama chat API"""
        system_message = """You are analyzing academic document headers to classify their section types.

Classify the header into ONE of these categories:
1. Relevant - Headers for conclusions, future work, summaries, discussions, recommendations, limitations, or implications sections
2. Not Relevant - Any other type of header (introduction, methodology, results, literature review, etc.)

TARGET SECTION TYPES (classify as "Relevant"):
- Conclusion sections, final remarks, closing thoughts, concluding remarks
- Future work, future directions, future research, next steps, prospects
- Summary sections, executive summaries, abstracts of findings
- Discussion sections, general discussion, interpretation of results
- Recommendations, practical implications, policy recommendations
- Limitations, study limitations, constraints
- Implications, theoretical implications, practical implications

Consider typical academic document structure and common header patterns.

NOTE: WE ARE ONLY INTERESTED IN CHAPTER HEADERS OR FULL SECTION HEADERS not SUBSECTION HEADERS

Respond with ONLY the category name (Relevant or Not Relevant) followed by a confidence score from 0.0 to 1.0.

Format: [Category] [Confidence]
Example: Relevant 0.85"""

        user_message = f'Analyze this header text and classify it: "{header_text}"'
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 50
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("message", {}).get("content", "").strip()
            
            category, confidence = parse_llm_response(response_text)
            return category, confidence
            
        except requests.exceptions.Timeout:
            print(f"Timeout processing header '{header_text[:30]}...', retrying...")
            try:
                response = requests.post(self.api_url, json=payload, timeout=90)
                response.raise_for_status()
                result = response.json()
                response_text = result.get("message", {}).get("content", "").strip()
                category, confidence = parse_llm_response(response_text)
                return category, confidence
            except Exception as e2:
                print(f"Retry failed for header '{header_text[:30]}...': {e2}")
                return "Other", 0.1
        except Exception as e:
            print(f"Error processing header '{header_text[:30]}...': {e}")
            return "Other", 0.1
    
    def classify_batch(self, headers: List[str], max_workers: int = 2) -> List[Tuple[str, float]]:
        """Classify headers with parallel workers"""
        import concurrent.futures
        
        results = [None] * len(headers)
        
        def classify_with_index(args):
            index, header_text = args
            # Small delay to avoid overwhelming Ollama
            time.sleep(0.2)
            category, confidence = self.classify_single(header_text)
            return index, category, confidence
        
        # Create list of (index, header) tuples
        indexed_headers = list(enumerate(headers))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use map for better handling
            future_to_index = {}
            
            # Submit all tasks
            for index, header in indexed_headers:
                future = executor.submit(classify_with_index, (index, header))
                future_to_index[future] = index
            
            # Process completed futures with progress bar
            completed_count = 0
            with tqdm(total=len(headers), desc=f"Ollama Classification ({self.model})") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        index, category, confidence = future.result()
                        results[index] = (category, confidence)
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"\nError processing header {index}: {e}")
                        results[index] = ("Other", 0.1)
                    
                    completed_count += 1
                    pbar.update(1)
        
        return results

def classify_single_gpu(headers_chunk, gpu_id, model, base_url_template):
    """Classify a chunk of headers on a specific GPU"""
    try:
        # Calculate the port for this GPU
        base_port = 11434
        port = base_port + gpu_id
        gpu_base_url = base_url_template.replace("11434", str(port))
        
        # Initialize classifier for this GPU
        classifier = OllamaClassifier(model=model, base_url=gpu_base_url)
        
        # Process this chunk
        results = []
        for header_text in tqdm(headers_chunk, desc=f"GPU {gpu_id} (Port {port})", leave=False):
            try:
                # Small delay to avoid overwhelming the server
                time.sleep(0.1)
                category, confidence = classifier.classify_single(header_text)
                results.append((category, confidence))
            except Exception as e:
                print(f"GPU {gpu_id} error processing header '{header_text[:30]}...': {e}")
                results.append(("Not Relevant", 0.1))
        
        return results, len(headers_chunk), gpu_id
    except Exception as e:
        print(f"GPU {gpu_id} initialization failed: {e}")
        # Return error results for all headers in this chunk
        return [("Not Relevant", 0.1)] * len(headers_chunk), len(headers_chunk), gpu_id

def classify_multi_gpu(headers: List[str], model: str, base_url: str = "http://127.0.0.1:11434", num_gpus: int = 6) -> List[Tuple[str, float]]:
    """Distribute header classification across multiple GPU-based Ollama servers"""
    if not headers:
        return []
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    print(f"Distributing {len(headers)} headers across {num_gpus} Ollama servers")
    
    # Split headers across GPUs
    chunks = []
    chunk_size = len(headers) // num_gpus
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            # Last chunk gets remaining headers
            end_idx = len(headers)
        else:
            end_idx = start_idx + chunk_size
        
        if start_idx < len(headers):
            chunks.append((headers[start_idx:end_idx], i, model, base_url))
    
    # Process chunks in parallel across GPUs
    all_results = []
    chunk_results = [None] * len(chunks)
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_chunk = {executor.submit(classify_single_gpu, chunk_data[0], chunk_data[1], chunk_data[2], chunk_data[3]): i 
                          for i, chunk_data in enumerate(chunks)}
        
        with tqdm(total=len(headers), desc="Multi-GPU Classification") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    results, count, gpu_id = future.result()
                    chunk_results[chunk_idx] = results
                    print(f"GPU {gpu_id} processed {count} headers")
                    pbar.update(count)
                except Exception as exc:
                    print(f"Chunk {chunk_idx} generated an exception: {exc}")
                    # Create error results for this chunk
                    chunk_size = len(chunks[chunk_idx][0])
                    chunk_results[chunk_idx] = [("Not Relevant", 0.1)] * chunk_size
                    pbar.update(chunk_size)
    
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

def classify_with_llm(pdf_results: List[Dict[str, Any]], 
                     model: str = "llama2",
                     base_url: str = "http://127.0.0.1:11434",
                     max_workers: int = 8,
                     num_gpus: int = 6,
                     verbose: bool = False) -> List[Dict[str, Any]]:
    """Classify headers using multi-GPU Ollama LLM"""
    
    # Collect all headers for batch processing
    all_headers = []
    header_to_pdf = {}
    
    for pdf_idx, pdf_data in enumerate(pdf_results):
        for header_idx, header in enumerate(pdf_data["all_predictions"]):
            all_headers.append(header["text"])
            header_to_pdf[len(all_headers) - 1] = (pdf_idx, header_idx)
    
    print(f"Classifying {len(all_headers)} headers with {num_gpus} GPU Ollama servers ({model})...")
    
    # Multi-GPU classify
    classifications = classify_multi_gpu(all_headers, model, base_url, num_gpus)
    
    # Apply results back to PDF data
    updated_results = []
    
    for pdf_idx, pdf_data in enumerate(pdf_results):
        updated_pdf = pdf_data.copy()
        updated_headers = []
        
        if verbose:
            print(f"\nProcessing {pdf_data['doc_id']}...")
        
        for header_idx, header in enumerate(pdf_data["all_predictions"]):
            # Find this header's classification result
            global_header_idx = None
            for idx, (pidx, hidx) in header_to_pdf.items():
                if pidx == pdf_idx and hidx == header_idx:
                    global_header_idx = idx
                    break
            
            if global_header_idx is not None:
                label, confidence = classifications[global_header_idx]
                
                updated_header = header.copy()
                updated_header.update({
                    "label": label,
                    "label_score": confidence,
                    "classification_method": f"llm_ollama"
                })
                
                if verbose:
                    print(f"  Page {header['page']}: '{header['text'][:50]}...' -> {label} ({confidence:.3f})")
                
                updated_headers.append(updated_header)
            else:
                # Fallback if something went wrong
                updated_header = header.copy()
                updated_header.update({
                    "label": "Not Relevant",
                    "label_score": 0.1,
                    "classification_method": f"llm_ollama_error"
                })
                updated_headers.append(updated_header)
        
        # Update PDF data
        updated_pdf["all_predictions"] = updated_headers
        
        # Recalculate binary statistics
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
    parser = argparse.ArgumentParser(description="LLM-based header classification")
    parser.add_argument("--results_dir", default="header_results", help="Directory with PDF JSON files")
    parser.add_argument("--gold", default="../final_groundtruth_filtered.csv", help="CSV file with ground truth")
    parser.add_argument("--model", default="llama4:scout", help="Ollama model name (e.g., llama2, mistral, codellama)")
    parser.add_argument("--base_url", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of parallel workers (reduce if getting 500 errors)")
    parser.add_argument("--num_gpus", type=int, default=6, help="Number of GPU Ollama servers (ports 11434, 11435, ..., 11439)")
    parser.add_argument("--output", help="Output JSON file with classified results")
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
    
    # Classify headers with Ollama LLM
    print(f"Classifying headers using Ollama with model: {args.model}...")
    updated_results = classify_with_llm(
        pdf_results,
        model=args.model,
        base_url=args.base_url,
        max_workers=args.max_workers,
        num_gpus=args.num_gpus,
        verbose=args.verbose
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(updated_results, f, indent=2, ensure_ascii=False)
        print(f"Saved classified results to {args.output}")
    
    # Evaluate
    if args.gold:
        print("\nEvaluating predictions...")
        metrics = evaluate_predictions(updated_results, Path(args.gold))
        
        print(f"\n=== BINARY EVALUATION RESULTS (OLLAMA-{args.model.upper()}) ===")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Total Ground Truth Pages: {metrics['total_gold_pages']}")

if __name__ == "__main__":
    main()