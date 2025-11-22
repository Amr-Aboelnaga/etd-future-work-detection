#!/usr/bin/env python3

"""
Evaluate all classification results from JSON files in the current directory.
This script finds all .json result files and evaluates them against ground truth
using the same methodology as the individual classification scripts.
"""

import argparse
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

def load_ground_truth(gold_csv: Path) -> Dict[str, Any]:
    """Load gold data from CSV"""
    gold = {}
    with open(gold_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["filename"]:
                gold[row["filename"]] = row
    return gold

def evaluate_predictions(results: List[Dict[str, Any]], gold_csv: Path) -> Dict[str, Any]:
    """Evaluate predictions at PAGE level - each detected page is a prediction"""
    # Load gold data
    gold = load_ground_truth(gold_csv)

    def evaluate_section(section_type: str, gold_key: str):
        """Evaluate section at page level"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        all_predictions = set()
        all_gold_pages = set()
        
        # Normalize section type for comparison (handle both "future_work" and "Future Work")
        normalized_section = section_type.lower().replace("_", " ")
        
        # Collect all predictions for this section type
        for result in results:
            doc_id = result["doc_id"]
            
            # Get all headers classified as this section type
            section_headers = [h for h in result.get("all_predictions", []) 
                             if h.get("label", "").lower() == normalized_section]
            
            # Each detected header represents a page-level prediction
            for header in section_headers:
                page_num = header["page"]
                all_predictions.add((doc_id, page_num))
        
        # Collect ground truth pages
        for result in results:
            doc_id = result["doc_id"]
            if doc_id not in gold:
                continue
                
            gold_row = gold[doc_id]
            gold_page = gold_row.get(gold_key, "")
            
            if gold_page and gold_page.strip() != "":
                gold_page_int = int(gold_page)
                all_gold_pages.add((doc_id, gold_page_int))
        
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
    
    return {
        "conclusion": evaluate_section("conclusion", "conclusion_page"),
        "future_work": evaluate_section("future_work", "future_work_page")
    }

def infer_method_from_filename(filename: str) -> str:
    """Infer classification method from filename"""
    filename_lower = filename.lower()
    
    if "lexical" in filename_lower:
        return "lexical"
    elif "keyword" in filename_lower:
        return "keyword"
    elif "nli" in filename_lower:
        return "nli"
    elif "ensemble" in filename_lower:
        return "ensemble"
    elif "ollama" in filename_lower or "llm" in filename_lower:
        return "llm_ollama"
    else:
        # Try to infer from classification_method field in the data
        return "unknown"

def infer_method_from_data(results: List[Dict[str, Any]]) -> str:
    """Infer method from the classification_method field in the data"""
    for result in results:
        for header in result.get("all_predictions", []):
            method = header.get("classification_method", "")
            if method:
                return method
    return "unknown"

def print_evaluation_summary(method: str, metrics: Dict[str, Any]) -> None:
    """Print evaluation summary for a method"""
    conc_metrics = metrics["conclusion"]
    fut_metrics = metrics["future_work"]
    
    # Combined metrics (macro-average)
    combined_precision = (conc_metrics['precision'] + fut_metrics['precision']) / 2
    combined_recall = (conc_metrics['recall'] + fut_metrics['recall']) / 2  
    combined_f1 = (conc_metrics['f1_score'] + fut_metrics['f1_score']) / 2
    
    print(f"\n=== {method.upper()} EVALUATION RESULTS ===")
    
    print(f"\nCOMBINED PAGE-LEVEL METRICS:")
    print(f"  Precision: {combined_precision:.3f}")
    print(f"  Recall: {combined_recall:.3f}")
    print(f"  F1-Score: {combined_f1:.3f}")
    
    print(f"\nCONCLUSION:")
    print(f"  Precision: {conc_metrics['precision']:.3f}")
    print(f"  Recall: {conc_metrics['recall']:.3f}")
    print(f"  F1-Score: {conc_metrics['f1_score']:.3f}")
    print(f"  TP: {conc_metrics['true_positives']}, FP: {conc_metrics['false_positives']}, FN: {conc_metrics['false_negatives']}")
    print(f"  Predicted {conc_metrics['total_predictions']} pages, Gold: {conc_metrics['total_gold_pages']} pages")
    
    print(f"\nFUTURE WORK:")
    print(f"  Precision: {fut_metrics['precision']:.3f}")
    print(f"  Recall: {fut_metrics['recall']:.3f}")
    print(f"  F1-Score: {fut_metrics['f1_score']:.3f}")
    print(f"  TP: {fut_metrics['true_positives']}, FP: {fut_metrics['false_positives']}, FN: {fut_metrics['false_negatives']}")
    print(f"  Predicted {fut_metrics['total_predictions']} pages, Gold: {fut_metrics['total_gold_pages']} pages")

def find_result_files(directory: Path) -> List[Path]:
    """Find all JSON result files in the directory"""
    json_files = []
    
    # Look for common result file patterns
    patterns = [
        "*results.json",
        "*_results.json", 
        "lexical*.json",
        "keyword*.json",
        "nli*.json",
        "ensemble*.json",
        "ollama*.json",
        "llm*.json"
    ]
    
    for pattern in patterns:
        json_files.extend(directory.glob(pattern))
    
    # Also look for any JSON file that might contain results
    all_json_files = list(directory.glob("*.json"))
    
    # Filter to files that look like result files
    for json_file in all_json_files:
        if json_file not in json_files:
            # Check if it contains classification results
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Check if it's a list of results with all_predictions
                if isinstance(data, list) and len(data) > 0:
                    if "all_predictions" in data[0] and "doc_id" in data[0]:
                        json_files.append(json_file)
                        
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    
    # Remove duplicates and sort
    json_files = sorted(set(json_files))
    return json_files

def main():
    parser = argparse.ArgumentParser(description="Evaluate all classification results from JSON files")
    parser.add_argument("--directory", default=".", help="Directory containing JSON result files")
    parser.add_argument("--gold", required=True, help="CSV file with ground truth")
    parser.add_argument("--output", help="Output CSV file with all metrics")
    parser.add_argument("--detailed", action="store_true", help="Show detailed evaluation for each file")
    parser.add_argument("--summary", action="store_true", help="Show summary comparison table")
    
    args = parser.parse_args()
    
    # Find all result files
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist!")
        sys.exit(1)
    
    result_files = find_result_files(directory)
    if not result_files:
        print(f"No JSON result files found in {directory}")
        sys.exit(1)
    
    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  {f.name}")
    
    # Evaluate each file
    all_results = {}
    
    for result_file in result_files:
        print(f"\n{'='*60}")
        print(f"Evaluating: {result_file.name}")
        print(f"{'='*60}")
        
        try:
            # Load results
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if not isinstance(results, list):
                print(f"Skipping {result_file.name}: Not a list of results")
                continue
            
            if not results:
                print(f"Skipping {result_file.name}: Empty results")
                continue
            
            # Infer method
            method = infer_method_from_filename(result_file.name)
            if method == "unknown":
                method = infer_method_from_data(results)
            
            print(f"Detected method: {method}")
            print(f"Results contain {len(results)} documents")
            
            # Evaluate
            metrics = evaluate_predictions(results, Path(args.gold))
            all_results[result_file.name] = {
                "method": method,
                "metrics": metrics,
                "file_path": str(result_file)
            }
            
            if args.detailed:
                print_evaluation_summary(method, metrics)
                
        except Exception as e:
            print(f"Error processing {result_file.name}: {e}")
    
    # Summary comparison
    if args.summary and all_results:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison table
        print(f"{'Method':<15} {'File':<25} {'Prec':<6} {'Rec':<6} {'F1':<6} {'C-TP':<5} {'C-FP':<5} {'C-FN':<5} {'F-TP':<5} {'F-FP':<5} {'F-FN':<5}")
        print("-" * 90)
        
        for filename, data in all_results.items():
            method = data["method"]
            metrics = data["metrics"]
            
            conc = metrics["conclusion"]
            fut = metrics["future_work"]
            
            # Combined metrics
            combined_precision = (conc['precision'] + fut['precision']) / 2
            combined_recall = (conc['recall'] + fut['recall']) / 2  
            combined_f1 = (conc['f1_score'] + fut['f1_score']) / 2
            
            # Truncate filename for display
            display_filename = filename[:24] + "..." if len(filename) > 24 else filename
            
            print(f"{method:<15} {display_filename:<25} {combined_precision:<6.3f} {combined_recall:<6.3f} {combined_f1:<6.3f} "
                  f"{conc['true_positives']:<5} {conc['false_positives']:<5} {conc['false_negatives']:<5} "
                  f"{fut['true_positives']:<5} {fut['false_positives']:<5} {fut['false_negatives']:<5}")
    
    # Save to CSV if requested
    if args.output and all_results:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "filename", "method", "combined_precision", "combined_recall", "combined_f1",
                "conclusion_precision", "conclusion_recall", "conclusion_f1", 
                "conclusion_tp", "conclusion_fp", "conclusion_fn",
                "future_work_precision", "future_work_recall", "future_work_f1",
                "future_work_tp", "future_work_fp", "future_work_fn"
            ])
            
            # Data rows
            for filename, data in all_results.items():
                method = data["method"]
                metrics = data["metrics"]
                
                conc = metrics["conclusion"]
                fut = metrics["future_work"]
                
                # Combined metrics
                combined_precision = (conc['precision'] + fut['precision']) / 2
                combined_recall = (conc['recall'] + fut['recall']) / 2  
                combined_f1 = (conc['f1_score'] + fut['f1_score']) / 2
                
                writer.writerow([
                    filename, method, combined_precision, combined_recall, combined_f1,
                    conc['precision'], conc['recall'], conc['f1_score'],
                    conc['true_positives'], conc['false_positives'], conc['false_negatives'],
                    fut['precision'], fut['recall'], fut['f1_score'],
                    fut['true_positives'], fut['false_positives'], fut['false_negatives']
                ])
        
        print(f"\nDetailed metrics saved to: {args.output}")
    
    print(f"\nEvaluation complete! Processed {len(all_results)} result files.")

if __name__ == "__main__":
    main()