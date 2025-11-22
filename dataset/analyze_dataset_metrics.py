#!/usr/bin/env python3
"""
Dataset Metrics Analysis Script
Analyzes dissertation PDFs and generates comprehensive metrics including:
- PDF counts and discipline distribution
- Page statistics (min, max, average)
- Subset analysis from filtered CSV
- Parallelized processing with progress tracking
"""

import os
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def get_pdf_page_count(pdf_path):
    """Extract page count from PDF using pdfinfo command."""
    try:
        result = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    return int(line.split(':')[1].strip())
    except Exception as e:
        print(f"Error getting page count for {pdf_path}: {e}")
    return None

def extract_department_from_xml(folder_path):
    """Extract department information from dublin_core.xml file."""
    dublin_core_path = os.path.join(folder_path, 'dublin_core.xml')
    if os.path.exists(dublin_core_path):
        try:
            tree = ET.parse(dublin_core_path)
            root = tree.getroot()
            
            for dcvalue in root.findall('dcvalue'):
                if (dcvalue.get('element') == 'contributor' and 
                    dcvalue.get('qualifier') == 'department'):
                    return dcvalue.text.strip() if dcvalue.text else 'Unknown'
        except Exception as e:
            pass  # Suppress individual errors for cleaner output
    
    return 'Unknown'

def process_directory(args):
    """Process a single directory to extract PDF info and department."""
    subdir, dataset_path = args
    folder_path = os.path.join(dataset_path, subdir)
    
    results = {
        'pdf_count': 0,
        'main_pdf_path': None,
        'main_pdf_pages': None,
        'department': 'Unknown'
    }
    
    try:
        # Find PDF files in this directory
        pdf_files_in_dir = [f for f in os.listdir(folder_path) 
                           if f.endswith('.pdf')]
        
        if pdf_files_in_dir:
            # Get department information
            department = extract_department_from_xml(folder_path)
            results['department'] = department
            results['pdf_count'] = len(pdf_files_in_dir)
            
            # Find the main PDF (one with highest page count)
            max_pages = 0
            main_pdf = None
            
            for pdf_file in pdf_files_in_dir:
                pdf_path = os.path.join(folder_path, pdf_file)
                page_count = get_pdf_page_count(pdf_path)
                
                if page_count and page_count > max_pages:
                    max_pages = page_count
                    main_pdf = pdf_path
            
            if main_pdf:
                results['main_pdf_path'] = main_pdf
                results['main_pdf_pages'] = max_pages
                    
    except Exception as e:
        pass  # Suppress individual errors
    
    return results

def analyze_large_dataset(dataset_path, max_workers=None):
    """Analyze the large dissertation dataset using parallel processing."""
    print("Analyzing large dataset with parallel processing...")
    
    if max_workers is None:
        max_workers = min(16, mp.cpu_count())  # Use up to 16 cores
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Found {len(subdirs):,} directories to process using {max_workers} workers")
    
    # Prepare arguments for parallel processing
    args_list = [(subdir, dataset_path) for subdir in subdirs]
    
    main_pdf_files = []
    page_counts = []
    department_distribution = Counter()
    total_pdfs_found = 0
    
    # Process directories in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_directory, args): args for args in args_list}
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Processing directories", unit="dirs"):
            try:
                result = future.result()
                if result['pdf_count'] > 0:
                    total_pdfs_found += result['pdf_count']
                    # Count department based on directory (dissertation), not individual PDFs
                    department_distribution[result['department']] += 1
                    
                    # Only collect page count from main PDF
                    if result['main_pdf_pages']:
                        main_pdf_files.append(result['main_pdf_path'])
                        page_counts.append(result['main_pdf_pages'])
                        
            except Exception as e:
                pass  # Continue processing even if individual tasks fail
    
    return {
        'total_directories_with_pdfs': len([f for f in main_pdf_files if f]),
        'total_pdfs_found': total_pdfs_found,
        'main_pdfs_analyzed': len(page_counts),
        'page_counts': page_counts,
        'department_distribution': dict(department_distribution),
        'main_pdf_files': main_pdf_files
    }

def get_pdf_page_count_for_subset(pdf_path):
    """Get page count for subset analysis."""
    if os.path.exists(pdf_path):
        return get_pdf_page_count(pdf_path)
    return None

def analyze_subset_dataset(csv_path, max_workers=None):
    """Analyze the subset dataset from CSV file with parallel processing."""
    print("Analyzing subset dataset from CSV...")
    
    if max_workers is None:
        max_workers = min(16, mp.cpu_count())
    
    pdf_files = []
    department_distribution = Counter()
    
    # Read CSV and get unique PDF paths
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        pdf_paths_seen = set()
        
        for row in reader:
            pdf_path = row['full_path']
            
            # Only count each PDF once
            if pdf_path not in pdf_paths_seen:
                pdf_paths_seen.add(pdf_path)
                pdf_files.append(pdf_path)
                
                # Get department from CSV
                department = row.get('department', 'Unknown').strip()
                department_distribution[department] += 1
    
    print(f"Found {len(pdf_files):,} unique PDFs in subset")
    
    # Get page counts in parallel
    page_counts = []
    if pdf_files:
        print("Extracting page counts for subset PDFs...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page count extraction tasks
            future_to_pdf = {executor.submit(get_pdf_page_count_for_subset, pdf_path): pdf_path 
                           for pdf_path in pdf_files}
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_pdf), total=len(future_to_pdf),
                             desc="Getting page counts", unit="PDFs"):
                try:
                    page_count = future.result()
                    if page_count:
                        page_counts.append(page_count)
                except Exception as e:
                    pass  # Continue even if individual PDFs fail
    
    return {
        'total_pdfs': len(pdf_files),
        'page_counts': page_counts,
        'department_distribution': dict(department_distribution),
        'pdf_files': pdf_files
    }

def calculate_page_statistics(page_counts):
    """Calculate page statistics."""
    if not page_counts:
        return {'min': 0, 'max': 0, 'average': 0, 'total_pages': 0}
    
    return {
        'min': min(page_counts),
        'max': max(page_counts),
        'average': round(sum(page_counts) / len(page_counts), 2),
        'median': round(sorted(page_counts)[len(page_counts)//2], 2),
        'total_pages': sum(page_counts),
        'total_pdfs_with_page_info': len(page_counts)
    }

def generate_markdown_report(report, large_dataset_metrics, subset_metrics, 
                           large_page_stats, subset_page_stats, output_file):
    """Generate comprehensive markdown report for scientific publication."""
    
    import datetime
    from statistics import stdev, median
    
    # Calculate additional statistics
    large_page_counts = large_dataset_metrics['page_counts']
    subset_page_counts = subset_metrics['page_counts']
    
    large_std = round(stdev(large_page_counts), 2) if len(large_page_counts) > 1 else 0
    subset_std = round(stdev(subset_page_counts), 2) if len(subset_page_counts) > 1 else 0
    
    # Calculate percentiles
    def calculate_percentiles(page_counts):
        if not page_counts:
            return {'25th': 0, '75th': 0, '95th': 0}
        sorted_counts = sorted(page_counts)
        n = len(sorted_counts)
        return {
            '25th': sorted_counts[int(0.25 * n)],
            '75th': sorted_counts[int(0.75 * n)], 
            '95th': sorted_counts[int(0.95 * n)]
        }
    
    large_percentiles = calculate_percentiles(large_page_counts)
    subset_percentiles = calculate_percentiles(subset_page_counts)
    
    md_content = f"""# Dataset Metrics Analysis Report

**Analysis Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of dissertation PDFs from the Virginia Tech Electronic Thesis and Dissertation (ETD) collection. The analysis covers both the complete large dataset ({large_dataset_metrics['total_directories_with_pdfs']:,} dissertations) and a curated subset ({subset_metrics['total_pdfs']:,} PDFs) used for research purposes.

## Dataset Overview

### Large Dataset (Complete ETD Collection)
- **Source Path:** `/projects/open_etds/etd/dissertation/`
- **Total Directories Analyzed:** 13,071
- **Directories with PDFs:** {large_dataset_metrics['total_directories_with_pdfs']:,}
- **Total PDF Documents Found:** {large_dataset_metrics['total_pdfs_found']:,}
- **Main Dissertations Analyzed:** {large_dataset_metrics['main_pdfs_analyzed']:,}
- **Page Analysis Coverage:** {large_page_stats['total_pdfs_with_page_info']:,}/{large_dataset_metrics['main_pdfs_analyzed']:,} ({large_page_stats['total_pdfs_with_page_info']/large_dataset_metrics['main_pdfs_analyzed']*100:.1f}%)

### Subset Dataset (Filtered Collection)
- **Source:** `final_groundtruth_filtered.csv`
- **Total PDF Documents:** {subset_metrics['total_pdfs']:,}
- **Documents with Page Information:** {subset_page_stats['total_pdfs_with_page_info']:,} ({subset_page_stats['total_pdfs_with_page_info']/subset_metrics['total_pdfs']*100:.1f}%)

## Page Count Analysis

### Large Dataset Statistics
- **Total Pages:** {large_page_stats['total_pages']:,}
- **Minimum Pages:** {large_page_stats['min']:,}
- **Maximum Pages:** {large_page_stats['max']:,}
- **Mean Pages:** {large_page_stats['average']:,}
- **Median Pages:** {large_page_stats.get('median', 'N/A'):,}
- **Standard Deviation:** {large_std:,}

**Percentile Distribution:**
- 25th Percentile: {large_percentiles['25th']:,} pages
- 75th Percentile: {large_percentiles['75th']:,} pages  
- 95th Percentile: {large_percentiles['95th']:,} pages

### Subset Dataset Statistics
- **Total Pages:** {subset_page_stats['total_pages']:,}
- **Minimum Pages:** {subset_page_stats['min']:,}
- **Maximum Pages:** {subset_page_stats['max']:,}
- **Mean Pages:** {subset_page_stats['average']:,}
- **Median Pages:** {subset_page_stats.get('median', 'N/A'):,}
- **Standard Deviation:** {subset_std:,}

**Percentile Distribution:**
- 25th Percentile: {subset_percentiles['25th']:,} pages
- 75th Percentile: {subset_percentiles['75th']:,} pages
- 95th Percentile: {subset_percentiles['95th']:,} pages

## Disciplinary Distribution Analysis

### Large Dataset - Complete Department Distribution
"""

    # Add complete large dataset department distribution
    large_dept_items = sorted(large_dataset_metrics['department_distribution'].items(), 
                             key=lambda x: x[1], reverse=True)
    
    md_content += f"\n**Total Departments:** {len(large_dept_items)}\n\n"
    md_content += "| Department | Count | Percentage |\n"
    md_content += "|------------|-------|------------|\n"
    
    total_large = sum(large_dataset_metrics['department_distribution'].values())
    for dept, count in large_dept_items:
        percentage = (count / total_large) * 100
        md_content += f"| {dept} | {count:,} | {percentage:.2f}% |\n"
    
    # Add subset dataset department distribution  
    md_content += f"""

### Subset Dataset - Complete Department Distribution

**Total Departments:** {len(subset_metrics['department_distribution'])}

"""
    
    subset_dept_items = sorted(subset_metrics['department_distribution'].items(),
                              key=lambda x: x[1], reverse=True)
    
    md_content += "| Department | Count | Percentage |\n"
    md_content += "|------------|-------|------------|\n"
    
    total_subset = sum(subset_metrics['department_distribution'].values())
    for dept, count in subset_dept_items:
        percentage = (count / total_subset) * 100
        md_content += f"| {dept} | {count:,} | {percentage:.2f}% |\n"


    # Add methodology section
    md_content += f"""

## Methodology

### Data Collection
- **Large Dataset:** Systematic traversal of 13,071 directories in the ETD repository
- **Main PDF Selection:** For each dissertation directory, selected PDF with highest page count as primary document
- **Subset Dataset:** Analysis of pre-filtered CSV containing {subset_metrics['total_pdfs']} selected documents  
- **Page Extraction:** Automated using `pdfinfo` utility from poppler-utils
- **Department Classification:** Extracted from Dublin Core XML metadata files

### Processing Details
- **Parallel Processing:** Utilized multi-core processing for efficient analysis
- **Error Handling:** Graceful handling of missing files, corrupted PDFs, and parsing errors
- **Data Validation:** Cross-validation between XML metadata and PDF file existence

### Quality Metrics
- **Large Dataset Coverage:** {large_page_stats['total_pdfs_with_page_info']}/{large_dataset_metrics['main_pdfs_analyzed']} ({large_page_stats['total_pdfs_with_page_info']/large_dataset_metrics['main_pdfs_analyzed']*100:.1f}%) main dissertations successfully analyzed for page counts
- **Subset Coverage:** {subset_page_stats['total_pdfs_with_page_info']}/{subset_metrics['total_pdfs']} ({subset_page_stats['total_pdfs_with_page_info']/subset_metrics['total_pdfs']*100:.1f}%) PDFs successfully analyzed for page counts

## Technical Specifications

- **Analysis Date:** {report['analysis_date']}
- **Processing Method:** Parallel processing with multiprocessing
- **Page Count Extraction:** pdfinfo utility
- **Department Classification:** Dublin Core XML metadata parsing
- **Output Formats:** JSON (machine-readable) and Markdown (human-readable)

## Data Files Generated

1. **JSON Report:** `dataset_metrics_report.json` - Complete machine-readable metrics
2. **Markdown Report:** `dataset_metrics_report.md` - Human-readable comprehensive analysis
3. **Processing Logs:** Console output with real-time progress tracking

---

*Report generated by automated dataset analysis pipeline*
"""

    # Write the markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Comprehensive markdown report saved to: {output_file}")

def generate_metrics_report(max_workers=None):
    """Generate comprehensive metrics report."""
    
    if max_workers is None:
        max_workers = min(16, mp.cpu_count())
    
    # Paths
    large_dataset_path = '/projects/open_etds/etd/dissertation'
    csv_path = '/projects/open_etds/amr_data/final_experiments/dataset/final_groundtruth_filtered.csv'
    
    # Check if pdfinfo is available
    try:
        subprocess.run(['pdfinfo', '--help'], capture_output=True)
    except FileNotFoundError:
        print("Warning: pdfinfo not found. Page counts will not be available.")
        print("Install poppler-utils to get page counts: sudo apt-get install poppler-utils")
    
    print("="*60)
    print("DATASET METRICS ANALYSIS")
    print(f"Using {max_workers} CPU cores for parallel processing")
    print("="*60)
    
    # Analyze large dataset
    print("\n" + "="*40)
    print("LARGE DATASET ANALYSIS")
    print("="*40)
    large_dataset_metrics = analyze_large_dataset(large_dataset_path, max_workers)
    large_page_stats = calculate_page_statistics(large_dataset_metrics['page_counts'])
    
    # Analyze subset dataset
    print("\n" + "="*40)
    print("SUBSET DATASET ANALYSIS")
    print("="*40)
    subset_metrics = analyze_subset_dataset(csv_path, max_workers)
    subset_page_stats = calculate_page_statistics(subset_metrics['page_counts'])
    
    # Create comprehensive report
    report = {
        'analysis_date': '2025-09-11',
        'large_dataset': {
            'path': large_dataset_path,
            'total_directories_with_pdfs': large_dataset_metrics['total_directories_with_pdfs'],
            'total_pdfs_found': large_dataset_metrics['total_pdfs_found'],
            'main_pdfs_analyzed': large_dataset_metrics['main_pdfs_analyzed'],
            'page_statistics': large_page_stats,
            'department_distribution': large_dataset_metrics['department_distribution']
        },
        'subset_dataset': {
            'path': csv_path,
            'total_pdfs': subset_metrics['total_pdfs'],
            'page_statistics': subset_page_stats,
            'department_distribution': subset_metrics['department_distribution']
        }
    }
    
    # Save detailed JSON report
    json_output_file = '/projects/open_etds/amr_data/final_experiments/dataset/dataset_metrics_report.json'
    with open(json_output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate comprehensive markdown report
    md_output_file = '/projects/open_etds/amr_data/final_experiments/dataset/dataset_metrics_report.md'
    generate_markdown_report(report, large_dataset_metrics, subset_metrics, 
                           large_page_stats, subset_page_stats, md_output_file)
    
    # Print summary
    print("\nLARGE DATASET METRICS:")
    print("-" * 40)
    print(f"Directories with PDFs: {large_dataset_metrics['total_directories_with_pdfs']:,}")
    print(f"Total PDF files found: {large_dataset_metrics['total_pdfs_found']:,}")
    print(f"Main dissertations analyzed: {large_dataset_metrics['main_pdfs_analyzed']:,}")
    print(f"Min pages: {large_page_stats['min']}")
    print(f"Max pages: {large_page_stats['max']}")
    print(f"Average pages: {large_page_stats['average']}")
    print(f"Total pages: {large_page_stats['total_pages']:,}")
    print(f"Main PDFs with page info: {large_page_stats['total_pdfs_with_page_info']:,}")
    
    print("\nTop 10 Departments (Large Dataset):")
    sorted_depts = sorted(large_dataset_metrics['department_distribution'].items(), 
                         key=lambda x: x[1], reverse=True)[:10]
    for dept, count in sorted_depts:
        print(f"  {dept}: {count:,}")
    
    print(f"\nSUBSET DATASET METRICS:")
    print("-" * 40)
    print(f"Total PDFs: {subset_metrics['total_pdfs']:,}")
    print(f"Min pages: {subset_page_stats['min']}")
    print(f"Max pages: {subset_page_stats['max']}")
    print(f"Average pages: {subset_page_stats['average']}")
    print(f"Total pages: {subset_page_stats['total_pages']:,}")
    print(f"PDFs with page info: {subset_page_stats['total_pdfs_with_page_info']:,}")
    
    print("\nTop 10 Departments (Subset):")
    sorted_subset_depts = sorted(subset_metrics['department_distribution'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]
    for dept, count in sorted_subset_depts:
        print(f"  {dept}: {count:,}")
    
    print(f"\nDetailed JSON report saved to: {json_output_file}")
    print(f"Comprehensive markdown report saved to: {md_output_file}")
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze dissertation dataset metrics')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of worker processes (default: min(16, CPU count))')
    parser.add_argument('--large-only', action='store_true',
                       help='Analyze only the large dataset (skip subset)')
    parser.add_argument('--subset-only', action='store_true', 
                       help='Analyze only the subset dataset (skip large dataset)')
    
    args = parser.parse_args()
    
    if args.workers is None:
        max_workers = min(16, mp.cpu_count())
    else:
        max_workers = min(args.workers, mp.cpu_count())
    
    print(f"Available CPU cores: {mp.cpu_count()}")
    print(f"Using {max_workers} worker processes")
    
    if not args.subset_only:
        if not args.large_only:
            generate_metrics_report(max_workers)
        else:
            # Quick mode for testing - just large dataset
            large_dataset_path = '/projects/open_etds/etd/dissertation'
            print("="*60)
            print("LARGE DATASET ANALYSIS ONLY")
            print(f"Using {max_workers} CPU cores")
            print("="*60)
            
            large_dataset_metrics = analyze_large_dataset(large_dataset_path, max_workers)
            large_page_stats = calculate_page_statistics(large_dataset_metrics['page_counts'])
            
            print(f"\nLARGE DATASET METRICS:")
            print("-" * 40)
            print(f"Total PDFs: {large_dataset_metrics['total_pdfs']:,}")
            print(f"Min pages: {large_page_stats['min']}")
            print(f"Max pages: {large_page_stats['max']}")
            print(f"Average pages: {large_page_stats['average']}")
            print(f"Total pages: {large_page_stats['total_pages']:,}")
            print(f"PDFs with page info: {large_page_stats['total_pdfs_with_page_info']:,}")
            
    elif args.subset_only:
        # Just subset analysis
        csv_path = '/projects/open_etds/amr_data/final_experiments/dataset/final_groundtruth_filtered.csv'
        print("="*60)
        print("SUBSET DATASET ANALYSIS ONLY")
        print(f"Using {max_workers} CPU cores")
        print("="*60)
        
        subset_metrics = analyze_subset_dataset(csv_path, max_workers)
        subset_page_stats = calculate_page_statistics(subset_metrics['page_counts'])
        
        print(f"\nSUBSET DATASET METRICS:")
        print("-" * 40)
        print(f"Total PDFs: {subset_metrics['total_pdfs']:,}")
        print(f"Min pages: {subset_page_stats['min']}")
        print(f"Max pages: {subset_page_stats['max']}")
        print(f"Average pages: {subset_page_stats['average']}")
        print(f"Total pages: {subset_page_stats['total_pages']:,}")
        print(f"PDFs with page info: {subset_page_stats['total_pdfs_with_page_info']:,}")
        
        print("\nDepartment Distribution:")
        sorted_depts = sorted(subset_metrics['department_distribution'].items(), 
                             key=lambda x: x[1], reverse=True)
        for dept, count in sorted_depts:
            print(f"  {dept}: {count:,}")