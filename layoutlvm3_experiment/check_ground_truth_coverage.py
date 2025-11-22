#!/usr/bin/env python3

"""
Check coverage of header detection results against ground truth CSV.
For each PDF in final_groundtruth.csv, check if the corresponding JSON file 
has header detections on the specific page numbers listed in the CSV.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_groundtruth(csv_path):
    """Load the ground truth CSV and organize by PDF file"""
    df = pd.read_csv(csv_path)
    
    # Group by PDF path and collect page numbers
    pdf_pages = defaultdict(set)
    pdf_info = {}
    
    for _, row in df.iterrows():
        pdf_path = row['full_path']
        page_num = row['page_number']
        chapter_title = row['chapter_title']
        
        pdf_pages[pdf_path].add(page_num)
        
        # Store additional info for the first occurrence of each PDF
        if pdf_path not in pdf_info:
            pdf_info[pdf_path] = {
                'department': row.get('department', ''),
                'college': row.get('college', ''),
                'entries': []
            }
        
        pdf_info[pdf_path]['entries'].append({
            'page_number': page_num,
            'chapter_title': chapter_title,
            'label': row.get('label', '')
        })
    
    return pdf_pages, pdf_info

def check_json_coverage(json_path, target_pages):
    """Check if JSON file has header detections on target pages"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get pages with header detections
        detected_pages = set()
        pages_with_headers = {}
        
        for page_data in data.get('pages', []):
            page_num = page_data['page_number']
            header_count = page_data.get('header_count', 0)
            
            if header_count > 0:
                detected_pages.add(page_num)
                pages_with_headers[page_num] = {
                    'header_count': header_count,
                    'headers': page_data.get('detected_headers', [])
                }
        
        # Check coverage
        covered_pages = target_pages.intersection(detected_pages)
        missed_pages = target_pages - detected_pages
        
        return {
            'total_pages': data.get('total_pages', 0),
            'pages_with_detections': len(detected_pages),
            'target_pages': target_pages,
            'covered_pages': covered_pages,
            'missed_pages': missed_pages,
            'coverage_ratio': len(covered_pages) / len(target_pages) if target_pages else 0,
            'pages_with_headers': pages_with_headers
        }
        
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {'error': 'Invalid JSON file'}
    except Exception as e:
        return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Check header detection coverage against ground truth")
    parser.add_argument("--csv_path", default="../final_groundtruth_filtered.csv", 
                       help="Path to ground truth CSV file")
    parser.add_argument("--results_dir", default="header_results", 
                       help="Directory containing header detection JSON files")
    parser.add_argument("--output_file", help="Save detailed results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed results for each PDF")
    parser.add_argument("--show_headers", action="store_true",
                       help="Show detected header text for covered pages")
    parser.add_argument("--min_coverage", type=float, default=0.0,
                       help="Only show PDFs with coverage >= this threshold")
    
    args = parser.parse_args()
    
    # Load ground truth
    print(f"Loading ground truth from: {args.csv_path}")
    pdf_pages, pdf_info = load_groundtruth(args.csv_path)
    
    print(f"Found {len(pdf_pages)} unique PDFs in ground truth")
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Check coverage for each PDF
    coverage_results = {}
    total_pdfs = 0
    pdfs_with_results = 0
    pdfs_with_coverage = 0
    total_target_pages = 0
    total_covered_pages = 0
    
    for pdf_path, target_pages in pdf_pages.items():
        total_pdfs += 1
        total_target_pages += len(target_pages)
        
        # Get PDF filename for JSON lookup
        pdf_name = Path(pdf_path).stem
        json_path = results_dir / f"{pdf_name}.json"
        
        coverage = check_json_coverage(json_path, target_pages)
        
        if coverage is None:
            coverage = {'error': 'JSON file not found', 'coverage_ratio': 0}
        elif 'error' not in coverage:
            pdfs_with_results += 1
            if coverage['coverage_ratio'] > 0:
                pdfs_with_coverage += 1
                total_covered_pages += len(coverage['covered_pages'])
        
        coverage_results[pdf_path] = coverage
        
        # Show detailed results if verbose and meets coverage threshold
        if args.verbose and coverage.get('coverage_ratio', 0) >= args.min_coverage:
            print(f"\n{pdf_path}")
            print(f"  PDF name: {pdf_name}")
            
            if 'error' in coverage:
                print(f"  Error: {coverage['error']}")
            else:
                print(f"  Total pages in PDF: {coverage['total_pages']}")
                print(f"  Pages with detections: {coverage['pages_with_detections']}")
                print(f"  Target pages: {sorted(target_pages)}")
                print(f"  Covered pages: {sorted(coverage['covered_pages'])}")
                print(f"  Missed pages: {sorted(coverage['missed_pages'])}")
                print(f"  Coverage: {coverage['coverage_ratio']:.2%}")
                
                # Show ground truth entries
                print(f"  Ground truth entries:")
                for entry in pdf_info[pdf_path]['entries']:
                    status = "✓" if entry['page_number'] in coverage['covered_pages'] else "✗"
                    print(f"    {status} Page {entry['page_number']}: {entry['chapter_title']} ({entry['label']})")
                
                # Show detected headers if requested
                if args.show_headers and coverage['covered_pages']:
                    print(f"  Detected headers on covered pages:")
                    for page_num in sorted(coverage['covered_pages']):
                        headers_info = coverage['pages_with_headers'].get(page_num, {})
                        header_count = headers_info.get('header_count', 0)
                        headers = headers_info.get('headers', [])
                        
                        print(f"    Page {page_num} ({header_count} headers):")
                        for header in headers[:3]:  # Show first 3 headers
                            text = header.get('text', '')
                            confidence = header.get('confidence', 0)
                            print(f"      - '{text}' (conf: {confidence:.3f})")
                        if len(headers) > 3:
                            print(f"      ... and {len(headers) - 3} more")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs in ground truth: {total_pdfs}")
    print(f"PDFs with JSON results: {pdfs_with_results} ({pdfs_with_results/total_pdfs:.1%})")
    print(f"PDFs with some coverage: {pdfs_with_coverage} ({pdfs_with_coverage/total_pdfs:.1%})")
    print(f"Total target pages: {total_target_pages}")
    print(f"Total covered pages: {total_covered_pages} ({total_covered_pages/total_target_pages:.1%})")
    
    # Coverage distribution
    coverage_dist = defaultdict(int)
    for pdf_path, result in coverage_results.items():
        coverage_ratio = result.get('coverage_ratio', 0)
        if 'error' not in result:
            if coverage_ratio == 0:
                coverage_dist['0%'] += 1
            elif coverage_ratio < 0.5:
                coverage_dist['1-49%'] += 1
            elif coverage_ratio < 1.0:
                coverage_dist['50-99%'] += 1
            else:
                coverage_dist['100%'] += 1
        else:
            coverage_dist['Error/Missing'] += 1
    
    print(f"\nCoverage Distribution:")
    for category, count in coverage_dist.items():
        percentage = count / total_pdfs * 100
        print(f"  {category}: {count} PDFs ({percentage:.1f}%)")
    
    # Show PDFs with no results
    no_results = []
    for pdf_path, result in coverage_results.items():
        if result.get('error') == 'JSON file not found':
            no_results.append(Path(pdf_path).stem)
    
    if no_results:
        print(f"\nPDFs missing JSON results ({len(no_results)}):")
        for i, pdf_name in enumerate(sorted(no_results)[:10]):
            print(f"  {pdf_name}.json")
        if len(no_results) > 10:
            print(f"  ... and {len(no_results) - 10} more")
    
    # Save detailed results if requested
    if args.output_file:
        # Prepare serializable results
        serializable_results = {}
        for pdf_path, result in coverage_results.items():
            # Convert sets to lists for JSON serialization
            if 'target_pages' in result:
                result['target_pages'] = list(result['target_pages'])
            if 'covered_pages' in result:
                result['covered_pages'] = list(result['covered_pages'])
            if 'missed_pages' in result:
                result['missed_pages'] = list(result['missed_pages'])
            
            serializable_results[pdf_path] = result
        
        output_data = {
            'summary': {
                'total_pdfs': total_pdfs,
                'pdfs_with_results': pdfs_with_results,
                'pdfs_with_coverage': pdfs_with_coverage,
                'total_target_pages': total_target_pages,
                'total_covered_pages': total_covered_pages,
                'overall_coverage': total_covered_pages / total_target_pages if total_target_pages > 0 else 0,
                'coverage_distribution': dict(coverage_dist)
            },
            'pdf_results': serializable_results
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main()