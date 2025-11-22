#!/usr/bin/env python3
"""
Calculate metrics for PDF section detection by comparing model predictions 
against ground truth labels.
"""

import csv
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse


def parse_labels_csv(csv_path, ignore_similar_chapters=False):
    """Parse the labels CSV file and return ground truth mappings."""
    ground_truth = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            
            # Extract all non-empty page numbers for this file
            pages = set()
            if row['conclusion_page'].strip():
                pages.add(int(row['conclusion_page']))
            if row['future_work_page'].strip():
                pages.add(int(row['future_work_page']))
            if not ignore_similar_chapters and row['similar_chapter_page'].strip():
                pages.add(int(row['similar_chapter_page']))
            
            # Always add the file, even if no labels (empty set means no ground truth pages)
            ground_truth[filename] = pages
    
    return ground_truth


def extract_predictions(results_file):
    """Extract predicted page numbers from conclusion_beginnings.txt file."""
    predictions = {}
    
    if not os.path.exists(results_file):
        return predictions
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for lines like: PDF: ../path/filename.pdf - Page: 123
            match = re.match(r'PDF: .*?([^/]+\.pdf) - Page: (\d+)', line)
            if match:
                filename = match.group(1)
                page_num = int(match.group(2))
                
                if filename not in predictions:
                    predictions[filename] = set()
                predictions[filename].add(page_num)
    
    return predictions


def calculate_metrics(ground_truth, predictions):
    """Calculate precision, recall, and F1 metrics."""
    
    # Only evaluate on files that are in the ground truth (labeled CSV)
    all_files = set(ground_truth.keys())
    
    total_tp = 0  # True positives
    total_fp = 0  # False positives  
    total_fn = 0  # False negatives
    
    detailed_results = []
    
    for filename in all_files:
        gt_pages = ground_truth.get(filename, set())
        pred_pages = predictions.get(filename, set())
        
        # Calculate per-file metrics
        tp = len(gt_pages & pred_pages)  # Intersection
        fp = len(pred_pages - gt_pages)  # Predicted but not in ground truth
        fn = len(gt_pages - pred_pages)  # In ground truth but not predicted
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Store detailed results for analysis
        detailed_results.append({
            'filename': filename,
            'ground_truth': gt_pages,
            'predictions': pred_pages,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt_files': len([f for f in all_files if f in ground_truth]),
        'total_pred_files': len([f for f in all_files if f in predictions]),
        'total_files': len(all_files)
    }
    
    return metrics, detailed_results


def process_single_results_dir(results_dir, labels_path, ignore_similar_chapters, detailed=False):
    """Process a single results directory and return metrics."""
    
    # Parse ground truth labels
    ground_truth = parse_labels_csv(labels_path, ignore_similar_chapters)
    
    # Extract predictions from results directory
    results_file = os.path.join(results_dir, 'conclusion_beginnings.txt')
    if not os.path.exists(results_file):
        return None, f"Error: Results file not found: {results_file}"
    
    predictions = extract_predictions(results_file)
    
    # Calculate metrics
    metrics, detailed_results = calculate_metrics(ground_truth, predictions)
    
    # Create output text
    model_name = os.path.basename(results_dir)
    output_lines = []
    output_lines.append(f"=== Metrics for {model_name} ===")
    output_lines.append(f"Precision: {metrics['precision']:.4f}")
    output_lines.append(f"Recall: {metrics['recall']:.4f}")
    output_lines.append(f"F1 Score: {metrics['f1']:.4f}")
    output_lines.append(f"True Positives: {metrics['total_tp']}")
    output_lines.append(f"False Positives: {metrics['total_fp']}")
    output_lines.append(f"False Negatives: {metrics['total_fn']}")
    output_lines.append(f"Files with ground truth: {metrics['total_gt_files']}")
    output_lines.append(f"Files with predictions: {metrics['total_pred_files']}")
    output_lines.append(f"Total files: {metrics['total_files']}")
    
    if detailed:
        output_lines.append(f"\n=== Detailed Results ===")
        for result in detailed_results:
            if result['tp'] > 0 or result['fp'] > 0 or result['fn'] > 0:
                output_lines.append(f"\nFile: {result['filename']}")
                output_lines.append(f"  Ground Truth: {sorted(result['ground_truth'])}")
                output_lines.append(f"  Predictions: {sorted(result['predictions'])}")
                output_lines.append(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'detailed_results': detailed_results,
        'output_text': '\n'.join(output_lines)
    }, None


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for PDF section detection')
    parser.add_argument('results_dir', nargs='?', help='Path to results directory containing conclusion_beginnings.txt (if not provided, processes all results_* folders)')
    parser.add_argument('--labels', default='etd_labels.csv', help='Path to ground truth labels CSV file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed per-file results')
    parser.add_argument('--ignore-similar-chapters', action='store_true', help='Ignore similar_chapter_page column from labels')
    parser.add_argument('--output-dir', default='metrics', help='Directory to save metric results')
    
    args = parser.parse_args()
    
    # Parse ground truth labels path
    labels_path = args.labels
    if not os.path.isabs(labels_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        labels_path = os.path.join(script_dir, labels_path)
    
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found: {labels_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which results directories to process
    if args.results_dir:
        # Single directory mode
        results_dirs = [args.results_dir]
    else:
        # Auto-discover all results_* directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dirs = []
        for item in os.listdir(script_dir):
            item_path = os.path.join(script_dir, item)
            if os.path.isdir(item_path) and item.startswith('results_'):
                results_dirs.append(item_path)
        
        if not results_dirs:
            print("No results_* directories found in current directory")
            sys.exit(1)
        
        results_dirs.sort()
    
    print(f"Processing {len(results_dirs)} results directories...")
    print(f"Loading ground truth from: {labels_path}")
    print(f"Ignore similar chapters: {args.ignore_similar_chapters}")
    print(f"Output directory: {output_dir}")
    
    # Process each results directory
    all_results = []
    summary_lines = ["=== Summary of All Models ===\n"]
    
    for results_dir in results_dirs:
        print(f"\nProcessing: {os.path.basename(results_dir)}")
        
        result, error = process_single_results_dir(
            results_dir, 
            labels_path, 
            args.ignore_similar_chapters,
            args.detailed
        )
        
        if error:
            print(f"  {error}")
            continue
        
        all_results.append(result)
        
        # Print to console
        print(result['output_text'])
        
        # Save individual result file
        model_name = result['model_name']
        output_file = os.path.join(output_dir, f"{model_name}_metrics.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['output_text'])
        print(f"  Saved: {output_file}")
        
        # Add to summary
        metrics = result['metrics']
        summary_lines.append(f"{model_name}:")
        summary_lines.append(f"  Precision: {metrics['precision']:.4f}")
        summary_lines.append(f"  Recall: {metrics['recall']:.4f}")
        summary_lines.append(f"  F1 Score: {metrics['f1']:.4f}")
        summary_lines.append("")
    
    # Save summary file
    if all_results:
        summary_file = os.path.join(output_dir, "summary_metrics.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\n=== Processing Complete ===")
        print(f"Processed {len(all_results)} models successfully")
        print(f"Individual metrics saved to: {output_dir}/")
        print(f"Summary saved to: {summary_file}")
        
        # Print summary to console
        print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()