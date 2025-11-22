#!/bin/bash

# Error Analysis Pipeline Runner
# Runs all Python analysis scripts in the correct order with error handling
# Author: Generated with Claude Code

set -e  # Exit on any error

# Colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}${BOLD}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}${BOLD}✓${NC} $1"
}

print_error() {
    echo -e "${RED}${BOLD}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}${BOLD}⚠${NC} $1"
}

# Function to run a Python script with error handling
run_script() {
    local script_name="$1"
    local description="$2"
    
    print_status "Running $description..."
    echo -e "${BLUE}Command:${NC} python3 $script_name"
    echo
    
    if timeout 600 python3 "$script_name"; then
        print_success "$description completed successfully"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_error "$description timed out after 10 minutes"
        else
            print_error "$description failed with exit code $exit_code"
        fi
        return $exit_code
    fi
    echo
}

# Main execution
main() {
    print_status "Starting Error Analysis Pipeline"
    echo "This script will run all analysis scripts in the correct order:"
    echo "1. Ground Truth Coverage Analysis"
    echo "2. Commonly Missed Pages Analysis" 
    echo "3. Conclusion Metrics on Correct Chapters"
    echo "4. Commonly Missed Conclusion Pages Analysis"
    echo "5. Extract Missed Pages Content"
    echo "6. Create Conclusion Content Files"
    echo
    
    # Check if we're in the right directory
    if [ ! -f "calculate_ground_truth_coverage.py" ]; then
        print_error "Python scripts not found in current directory!"
        print_error "Please run this script from the error_analysis directory"
        exit 1
    fi
    
    # Create timestamp for this run
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    print_status "Analysis run timestamp: $timestamp"
    echo
    
    # Script 1: Ground truth coverage analysis
    run_script "calculate_ground_truth_coverage.py" "Ground Truth Coverage Analysis"
    
    # Script 2: Commonly missed pages analysis
    run_script "analyze_commonly_missed_pages.py" "Commonly Missed Pages Analysis"
    
    # Script 3: Conclusion metrics on correct chapters
    run_script "calculate_conclusion_metrics_on_correct_chapters.py" "Conclusion Metrics on Correct Chapters"
    
    # Script 4: Commonly missed conclusion pages analysis
    run_script "analyze_commonly_missed_conclusion_pages.py" "Commonly Missed Conclusion Pages Analysis"
    
    # Script 5: Extract missed pages content
    run_script "extract_missed_pages_content.py" "Extract Missed Pages Content"
    
    # Script 6: Create conclusion content files
    run_script "create_conclusion_content_files.py" "Create Conclusion Content Files"
    
    
    # Summary
    print_status "Pipeline completed successfully!"
    echo
    print_success "Generated files summary:"
    echo "📊 Coverage Analysis:"
    echo "  • 1_chapter_beginning_detection_coverage_detailed.json"
    echo "  • 1_chapter_beginning_detection_coverage_by_model.csv"
    echo
    echo "📉 Missed Pages Analysis:"
    echo "  • 2_chapter_pages_missed_detailed_analysis.json"
    echo "  • 2_chapter_pages_missed_by_most_models.csv"
    echo "  • 3_chapter_pages_missed_by_all_models.csv"
    echo "  • 4_chapter_pages_miss_rates_by_label_type.csv"
    echo
    echo "📋 Conclusion Detection Analysis:"
    echo "  • 5_conclusion_detection_metrics_detailed.json"
    echo "  • 5_conclusion_detection_metrics_on_detected_chapters.csv"
    echo
    echo "📊 Conclusion Missed Pages:"
    echo "  • 6_conclusion_pages_missed_detailed_analysis.json"
    echo "  • 6_conclusion_pages_missed_by_most_models.csv"
    echo "  • 7_conclusion_pages_missed_by_all_models.csv"
    echo "  • 8_conclusion_pages_miss_rates_by_label_type.csv"
    echo
    echo "📄 Content Analysis:"
    echo "  • 12_missed_chapter_pages_content_analysis.txt"
    echo "  • 13_chapter_pages_missed_by_all_content.txt"
    echo "  • 14_missed_conclusion_pages_content_analysis.txt"
    echo "  • 15_conclusion_pages_missed_by_all_content.txt"
    echo
    print_success "All analysis completed at $(date '+%Y-%m-%d %H:%M:%S')"
}

# Handle script interruption
trap 'print_error "Script interrupted by user"; exit 130' INT TERM

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Run main function
main "$@"