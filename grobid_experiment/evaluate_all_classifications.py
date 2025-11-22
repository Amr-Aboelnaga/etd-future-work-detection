#!/usr/bin/env python3

"""
Evaluate all classification results from JSON files in the classification_results directory.
This script finds all .json result files and evaluates them against ground truth using 
the final_groundtruth_filtered.csv file, following the same methodology as the individual 
classification scripts.
"""

import argparse
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import pandas as pd

def load_classification_results(json_file: Path) -> List[Dict[str, Any]]:
    """Load classification results from a JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not isinstance(results, list):
            print(f"Warning: {json_file.name} does not contain a list of results")
            return []
            
        if not results:
            print(f"Warning: {json_file.name} contains empty results")
            return []
            
        return results
    except Exception as e:
        print(f"Error loading {json_file.name}: {e}")
        return []

def evaluate_predictions(results: List[Dict[str, Any]], gold_csv: Path) -> Dict[str, Any]:
    """Evaluate predictions at PAGE level - each detected page is a prediction"""
    # Load ground truth data from filtered CSV
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

def infer_method_from_filename(filename: str) -> str:
    """Infer classification method from filename"""
    filename_lower = filename.lower()
    
    if "semantic" in filename_lower:
        # Extract threshold if present
        if "_t0." in filename_lower or "_t1." in filename_lower:
            # Find threshold pattern
            import re
            match = re.search(r'_t([\d.]+)', filename_lower)
            if match:
                threshold = match.group(1)
                return f"semantic_t{threshold}"
        return "semantic"
    elif "ollama" in filename_lower or "llm" in filename_lower:
        # Extract model name if present
        if "llama" in filename_lower:
            return "ollama_llm"
        return "llm"
    elif "lexical" in filename_lower:
        return "lexical"
    elif "keyword" in filename_lower:
        return "keyword"  
    elif "nli" in filename_lower:
        return "nli"
    elif "ensemble" in filename_lower:
        return "ensemble"
    elif "nil" in filename_lower:
        return "baseline"
    elif any(threshold in filename_lower for threshold in ["50", "60", "70", "80"]):
        # Confidence threshold-based classification
        for threshold in ["50", "60", "70", "80"]:
            if threshold in filename_lower:
                return f"confidence_t{threshold}"
        return "confidence_threshold"
    else:
        return "unknown"

def infer_method_from_data(results: List[Dict[str, Any]]) -> str:
    """Infer method from the classification_method field in the data"""
    for result in results:
        for header in result.get("all_predictions", []):
            method = header.get("classification_method", "")
            if method:
                if "semantic" in method:
                    return "semantic"
                elif "llm" in method or "ollama" in method:
                    return "llm"
                elif "lexical" in method:
                    return "lexical"
                elif "nli" in method:
                    return "nli"
                else:
                    return method
    return "unknown"

def print_evaluation_summary(method: str, metrics: Dict[str, Any], filename: str = "") -> None:
    """Print evaluation summary for a method"""
    file_info = f" ({filename})" if filename else ""
    print(f"\n=== {method.upper()}{file_info} EVALUATION RESULTS ===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Total Ground Truth Pages: {metrics['total_gold_pages']}")

def find_classification_result_files(results_dir: Path) -> List[Path]:
    """Find all JSON classification result files in the results directory"""
    if not results_dir.exists():
        return []
    
    json_files = list(results_dir.glob("*.json"))
    
    # Filter to files that look like classification result files
    valid_files = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if it's a list of results with all_predictions
            if isinstance(data, list) and len(data) > 0:
                if "all_predictions" in data[0] and "doc_id" in data[0]:
                    valid_files.append(json_file)
                    
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    
    return sorted(valid_files)

def main():
    parser = argparse.ArgumentParser(description="Evaluate all classification results from JSON files")
    parser.add_argument("--results_dir", default="classification_results", 
                        help="Directory containing classification result JSON files")
    parser.add_argument("--gold", default="../final_groundtruth_filtered.csv", 
                        help="CSV file with ground truth")
    parser.add_argument("--output", help="Output CSV file with all metrics")
    parser.add_argument("--detailed", action="store_true", help="Show detailed evaluation for each file")
    parser.add_argument("--summary", action="store_true", default=True, 
                        help="Show summary comparison table (default: True)")
    parser.add_argument("--quiet", action="store_true", help="Only show final summary table")
    
    args = parser.parse_args()
    
    # Find all result files
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        sys.exit(1)
    
    result_files = find_classification_result_files(results_dir)
    if not result_files:
        print(f"No valid classification result files found in {results_dir}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(result_files)} classification result files:")
        for f in result_files:
            print(f"  {f.name}")
    
    # Load ground truth
    gold_file = Path(args.gold)
    if not gold_file.exists():
        print(f"Ground truth file {gold_file} does not exist!")
        sys.exit(1)
    
    # Evaluate each file
    all_results = {}
    
    for result_file in result_files:
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"Evaluating: {result_file.name}")
            print(f"{'='*60}")
        
        # Load classification results
        results = load_classification_results(result_file)
        if not results:
            continue
        
        # Infer method
        method = infer_method_from_filename(result_file.name)
        if method == "unknown":
            method = infer_method_from_data(results)
        
        if not args.quiet:
            print(f"Detected method: {method}")
            print(f"Results contain {len(results)} documents")
        
        # Evaluate against ground truth
        metrics = evaluate_predictions(results, gold_file)
        all_results[result_file.name] = {
            "method": method,
            "metrics": metrics,
            "file_path": str(result_file)
        }
        
        if args.detailed:
            print_evaluation_summary(method, metrics, result_file.name)
    
    # Summary comparison
    if args.summary and all_results:
        print(f"\n{'='*100}")
        print("CLASSIFICATION RESULTS SUMMARY")
        print(f"{'='*100}")
        
        # Create comparison table
        print(f"{'Method':<20} {'File':<35} {'Prec':<6} {'Rec':<6} {'F1':<6} {'TP':<4} {'FP':<4} {'FN':<4} {'Pred':<5} {'Gold':<5}")
        print("-" * 100)
        
        # Sort by F1 score (descending)
        sorted_results = sorted(all_results.items(), 
                               key=lambda x: x[1]["metrics"]["f1_score"], 
                               reverse=True)
        
        for filename, data in sorted_results:
            method = data["method"]
            metrics = data["metrics"]
            
            # Truncate filename for display
            display_filename = filename[:34] + "..." if len(filename) > 34 else filename
            
            print(f"{method:<20} {display_filename:<35} "
                  f"{metrics['precision']:<6.3f} {metrics['recall']:<6.3f} {metrics['f1_score']:<6.3f} "
                  f"{metrics['true_positives']:<4} {metrics['false_positives']:<4} {metrics['false_negatives']:<4} "
                  f"{metrics['total_predictions']:<5} {metrics['total_gold_pages']:<5}")
    
    # Save to CSV if requested
    if args.output and all_results:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "filename", "method", "precision", "recall", "f1_score", 
                "true_positives", "false_positives", "false_negatives",
                "total_predictions", "total_gold_pages"
            ])
            
            # Data rows - sorted by F1 score
            sorted_results = sorted(all_results.items(), 
                                   key=lambda x: x[1]["metrics"]["f1_score"], 
                                   reverse=True)
            
            for filename, data in sorted_results:
                method = data["method"]
                metrics = data["metrics"]
                
                writer.writerow([
                    filename, method, 
                    metrics['precision'], metrics['recall'], metrics['f1_score'],
                    metrics['true_positives'], metrics['false_positives'], metrics['false_negatives'],
                    metrics['total_predictions'], metrics['total_gold_pages']
                ])
        
        print(f"\nDetailed metrics saved to: {args.output}")
    
    print(f"\nEvaluation complete! Processed {len(all_results)} result files.")
    if all_results:
        best_result = max(all_results.items(), key=lambda x: x[1]["metrics"]["f1_score"])
        best_filename, best_data = best_result
        print(f"Best performing method: {best_data['method']} ({best_filename}) with F1-Score: {best_data['metrics']['f1_score']:.3f}")

if __name__ == "__main__":
    main()