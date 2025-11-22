#!/usr/bin/env python3
"""
Quick test script to generate markdown report using subset data only
"""

from analyze_dataset_metrics import analyze_subset_dataset, calculate_page_statistics, generate_markdown_report
import json
import datetime

def test_markdown_generation():
    """Test markdown report generation with subset data."""
    
    print("Testing markdown report generation...")
    
    # Analyze subset dataset
    csv_path = '/projects/open_etds/amr_data/final_experiments/dataset/final_groundtruth_filtered.csv'
    subset_metrics = analyze_subset_dataset(csv_path, max_workers=4)
    subset_page_stats = calculate_page_statistics(subset_metrics['page_counts'])
    
    # Create dummy large dataset metrics for testing
    large_dataset_metrics = {
        'total_pdfs': 10000,  # Placeholder
        'page_counts': [100, 200, 300, 400, 500] * 100,  # Dummy data
        'department_distribution': {'Engineering': 2000, 'Sciences': 1500, 'Liberal Arts': 1000}
    }
    large_page_stats = calculate_page_statistics(large_dataset_metrics['page_counts'])
    
    # Create report structure
    report = {
        'analysis_date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'large_dataset': {
            'path': '/projects/open_etds/etd/dissertation',
            'total_pdfs': large_dataset_metrics['total_pdfs'],
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
    
    # Generate markdown report
    md_output_file = '/projects/open_etds/amr_data/final_experiments/dataset/test_subset_report.md'
    generate_markdown_report(report, large_dataset_metrics, subset_metrics, 
                           large_page_stats, subset_page_stats, md_output_file)
    
    print(f"Test markdown report generated: {md_output_file}")
    
    # Also save JSON
    json_output_file = '/projects/open_etds/amr_data/final_experiments/dataset/test_subset_report.json'
    with open(json_output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Test JSON report generated: {json_output_file}")

if __name__ == "__main__":
    test_markdown_generation()