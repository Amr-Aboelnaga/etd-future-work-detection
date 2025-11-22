#!/usr/bin/env python3

"""
Improved header classification with semantic similarity and simplified metrics.
"""
print("before importing")
import argparse
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from datetime import datetime
print("mid importing")

import numpy as np
import pandas as pd
from tqdm import tqdm

# For semantic similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
print("after importing")

def generate_semantic_filename(model: str, threshold: float, base_name: str = "semantic_results") -> str:
    """Generate descriptive filename for semantic classification"""
    # Simplify model name
    model_short = model.replace('all-', '').replace('-', '').lower()
    
    # Format threshold as integer if it's a round number, otherwise keep decimal
    if threshold == int(threshold):
        thresh_str = str(int(threshold))
    else:
        thresh_str = f"{threshold:.2f}".replace('.', '_')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    parts = [base_name, "semantic", model_short, f"thresh{thresh_str}", timestamp]
    return "_".join(parts) + ".json"

def save_semantic_metrics_to_csv(metrics: Dict[str, Any], model: str, threshold: float, 
                                parameters: Dict[str, Any], 
                                csv_file: Path = Path("semantic_metrics.csv")):
    """Save semantic classification metrics to CSV file"""
    
    # Prepare row data
    row_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "semantic",
        "model": model,
        "threshold": threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "total_predictions": metrics["total_predictions"],
        "total_gold_pages": metrics["total_gold_pages"],
        "output_file": parameters.get("output_file", "")
    }
    
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
    print(f"Semantic metrics saved to {csv_file}")
    
    return csv_file

# Prototype sections for semantic similarity - all target chapter types
RELEVANT_PROTOTYPES = [
    # CONCLUSION prototypes
    "In conclusion, this study demonstrates",
    "Our findings suggest that",
    "To summarize, the main contributions are",
    "This paper has presented a comprehensive analysis",
    "The results of our experiments show",
    "We have shown that our approach",
    "In summary, we propose a novel method",
    "Concluding remarks and final thoughts",
    "Overall, this work makes several contributions",
    "Our research findings indicate",
    "General conclusions from this work",
    "Final remarks on our approach",
    
    # FUTURE_WORK prototypes
    "Future research directions include",
    "There are several promising avenues for future work",
    "We plan to extend this work in several ways",
    "Future studies should investigate",
    "Several limitations suggest opportunities for improvement",
    "Next steps in this research include",
    "We intend to explore these ideas further",
    "Promising directions for future research",
    "Areas for future investigation include",
    "Our future work will focus on",
    "Directions for future research",
    "Further research is needed",
    
    # SUMMARY prototypes
    "Summary of findings presented here",
    "This executive summary outlines",
    "Abstract of findings from our study",
    "Key findings summarized below",
    
    # DISCUSSION prototypes
    "General discussion of results",
    "Discussion and interpretation of findings",
    "We discuss the implications of these results",
    "This discussion focuses on",
    
    # RECOMMENDATIONS prototypes
    "Our recommendations for practice include",
    "Policy recommendations based on findings",
    "Practical implications of this research",
    "We recommend the following actions",
    
    # LIMITATIONS prototypes
    "Limitations of this study include",
    "Several limitations must be acknowledged",
    "Study limitations and constraints",
    
    # IMPLICATIONS prototypes
    "Theoretical implications of these findings",
    "Practical implications for the field",
    "The implications of our work suggest",
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
]

class SemanticHeaderClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model"""
        print(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Pre-compute prototype embeddings
        print("Computing prototype embeddings...")
        self.relevant_embeddings = self.model.encode(RELEVANT_PROTOTYPES)
        
    def classify_header(self, header_text: str, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Binary classification of a single header using semantic similarity.
        
        Args:
            header_text: The header text to classify
            threshold: Minimum similarity threshold for classification
            
        Returns:
            (label, confidence_score) - "Relevant" or "Not Relevant"
        """
        # Get embedding for the header
        header_embedding = self.model.encode([header_text])
        
        # Compute similarities to relevant prototypes
        relevant_sims = cosine_similarity(header_embedding, self.relevant_embeddings)[0]
        
        # Get best similarity score
        best_similarity = np.max(relevant_sims)
        
        # Binary classification: Relevant or Not Relevant
        if best_similarity >= threshold:
            return "Relevant", best_similarity
        else:
            return "Not Relevant", best_similarity
    
    def batch_classify(self, header_texts: List[str], threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Efficiently classify multiple headers at once"""
        if not header_texts:
            return []
        
        # Batch encode all headers
        header_embeddings = self.model.encode(header_texts)
        
        results = []
        for i, header_embedding in enumerate(header_embeddings):
            header_embedding = header_embedding.reshape(1, -1)
            
            # Compute similarities to relevant prototypes
            relevant_sims = cosine_similarity(header_embedding, self.relevant_embeddings)[0]
            best_similarity = np.max(relevant_sims)
            
            # Binary classification
            if best_similarity >= threshold:
                results.append(("Relevant", best_similarity))
            else:
                results.append(("Not Relevant", best_similarity))
        
        return results

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

def reclassify_with_semantic(pdf_results: List[Dict[str, Any]], 
                           classifier: SemanticHeaderClassifier,
                           threshold: float = 0.3,
                           verbose: bool = False) -> List[Dict[str, Any]]:
    """Reclassify headers using semantic similarity"""
    
    updated_results = []
    
    for pdf_data in tqdm(pdf_results, desc="Semantic classification"):
        updated_pdf = pdf_data.copy()
        headers = pdf_data["all_predictions"]
        
        if verbose:
            print(f"\nProcessing {pdf_data['doc_id']}...")
        
        # Extract header texts for batch processing
        header_texts = [h["text"] for h in headers]
        
        # Batch classify
        classifications = classifier.batch_classify(header_texts, threshold=threshold)
        
        # Update headers with classifications
        updated_headers = []
        for header, (label, score) in zip(headers, classifications):
            updated_header = header.copy()
            updated_header.update({
                "label": label,
                "label_score": float(score),
                "classification_method": "semantic_similarity"
            })
            
            if verbose:
                print(f"  Page {header['page']}: '{header['text'][:50]}...' -> {label} ({score:.3f})")
            
            updated_headers.append(updated_header)
        
        # Update PDF data
        updated_pdf["all_predictions"] = updated_headers
        
        # Recalculate summary statistics for binary classification
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

def print_binary_results(metrics: Dict[str, Any], method_name: str = "Semantic"):
    """Print binary evaluation results matching classify_headers.py format"""
    print(f"\n=== BINARY EVALUATION RESULTS ({method_name.upper()}) ===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Total Ground Truth Pages: {metrics['total_gold_pages']}")

def main():
    parser = argparse.ArgumentParser(description="Semantic header classification")
    parser.add_argument("--results_dir", default="header_results", help="Directory with PDF JSON files")
    parser.add_argument("--gold", default="../final_groundtruth_filtered.csv", help="CSV file with ground truth")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                        help="Sentence transformer model")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Similarity threshold for classification")
    parser.add_argument("--output", help="Output JSON file with results (auto-generated if not specified)")
    parser.add_argument("--csv_metrics", default="semantic_metrics.csv", help="CSV file to save metrics for comparison")
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
    
    # Initialize semantic classifier
    classifier = SemanticHeaderClassifier(model_name=args.model)
    
    # Reclassify headers
    print(f"Classifying headers with semantic similarity (threshold={args.threshold})")
    updated_results = reclassify_with_semantic(
        pdf_results, 
        classifier,
        threshold=args.threshold,
        verbose=args.verbose
    )
    
    # Generate output filename if not specified
    if not args.output:
        args.output = generate_semantic_filename(
            model=args.model,
            threshold=args.threshold
        )
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(updated_results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {args.output}")
    
    # Evaluate and save metrics
    if args.gold:
        print("\nEvaluating predictions...")
        metrics = evaluate_predictions(updated_results, Path(args.gold))
        
        # Prepare parameters for CSV export
        parameters = {
            "output_file": args.output
        }
        
        # Save metrics to CSV
        csv_file = save_semantic_metrics_to_csv(
            metrics=metrics,
            model=args.model,
            threshold=args.threshold,
            parameters=parameters,
            csv_file=Path(args.csv_metrics)
        )
        
        print_binary_results(metrics, "Semantic")
        print(f"\nMetrics comparison available in: {csv_file}")

if __name__ == "__main__":
    main()