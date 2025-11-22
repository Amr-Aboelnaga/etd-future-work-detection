#!/usr/bin/env python3

"""
Filter ground truth CSV to only include PDFs that have corresponding JSON files.
This removes any entries where the PDF doesn't have detection results.
"""
import json
import pandas as pd
import argparse
from pathlib import Path

def filter_groundtruth(csv_path, results_dir, output_path):
    """Filter ground truth to only include PDFs with JSON results"""
    
    # Load the original ground truth CSV
    df = pd.read_csv(csv_path)
    print(f"Original ground truth entries: {len(df)}")
    print(f"Original unique PDFs: {df['full_path'].nunique()}")
    
    # Get all full paths from JSON files in results directory
    results_dir = Path(results_dir)
    json_full_paths = set()
    
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    full_path = data.get('full_path')
                    if full_path:
                        json_full_paths.add(full_path)
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                # Skip invalid JSON files
                continue
        
        print(f"Found {len(json_full_paths)} JSON files with full_path in results directory")
    else:
        print(f"Results directory not found: {results_dir}")
        return
    
    # Filter the dataframe
    original_count = len(df)
    original_pdfs = df['full_path'].nunique()
    
    # Create mask - keep only rows where PDF full_path matches JSON full_path
    mask = df['full_path'].isin(json_full_paths)
    print(f"Rows with matching JSON files: {mask.sum()}")
    print(f"Rows without matching JSON files: {(~mask).sum()}")
    
    # Show some examples of PDFs without JSON files
    missing_pdfs = df[~mask]['full_path'].unique()
    if len(missing_pdfs) > 0:
        print(f"\nPDFs without JSON results (first 10):")
        for pdf_path in missing_pdfs[:10]:
            pdf_name = Path(pdf_path).stem
            print(f"  {pdf_name} (no JSON with matching full_path)")
    
    # Apply the filter
    filtered_df = df[mask].copy()
    filtered_count = len(filtered_df)
    filtered_pdfs = filtered_df['full_path'].nunique()
    
    print(f"\nFiltering results:")
    print(f"  Entries: {original_count} -> {filtered_count} (removed {original_count - filtered_count})")
    print(f"  Unique PDFs: {original_pdfs} -> {filtered_pdfs} (removed {original_pdfs - filtered_pdfs})")
    print(f"  Retention rate: {filtered_count/original_count:.1%}")
    
    # Show some examples of removed PDFs
    removed_mask = ~mask
    removed_pdfs = df[removed_mask]['full_path'].unique()
    
    if len(removed_pdfs) > 0:
        print(f"\nExample PDFs removed (first 10):")
        for i, pdf_path in enumerate(removed_pdfs[:10]):
            pdf_name = Path(pdf_path).stem
            print(f"  {pdf_name} (no {pdf_name}.json)")
        if len(removed_pdfs) > 10:
            print(f"  ... and {len(removed_pdfs) - 10} more")
    
    # Save filtered CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"\nFiltered ground truth saved to: {output_path}")
    
    # Show distribution by label
    if 'label' in filtered_df.columns:
        print(f"\nLabel distribution in filtered data:")
        label_counts = filtered_df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} entries")
    
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Filter ground truth CSV by available JSON results")
    parser.add_argument("--csv_path", default="../final_groundtruth.csv",
                       help="Path to original ground truth CSV file")
    parser.add_argument("--results_dir", default="header_results",
                       help="Directory containing JSON result files")
    parser.add_argument("--output_path", default="../final_groundtruth_filtered.csv",
                       help="Path for filtered ground truth CSV")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be filtered without saving")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output_path)
    
    if not csv_path.exists():
        print(f"Ground truth CSV not found: {csv_path}")
        return
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be saved")
        output_path = None
    
    # Perform filtering
    filtered_df = filter_groundtruth(csv_path, results_dir, output_path if not args.dry_run else None)
    
    if args.dry_run:
        print(f"\nTo actually create the filtered file, run:")
        print(f"python {__file__} --csv_path {csv_path} --results_dir {results_dir} --output_path {output_path}")

if __name__ == "__main__":
    main()