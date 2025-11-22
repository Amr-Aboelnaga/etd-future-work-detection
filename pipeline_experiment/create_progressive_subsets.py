#!/usr/bin/env python3
"""
Create progressive random subsets from final_groundtruth_filtered.csv
- 25%: Random 25% of records
- 50%: Previous 25% + random 25% from remaining
- 75%: Previous 50% + random 25% from remaining  
- 100%: All records
"""

import pandas as pd
import numpy as np

def main():
    # Load the final groundtruth dataset
    final_path = "/projects/open_etds/amr_data/final_experiments/final_groundtruth_filtered.csv"
    output_dir = "/projects/open_etds/amr_data/final_experiments"
    
    print("Loading final_groundtruth_filtered.csv...")
    df = pd.read_csv(final_path)
    total_records = len(df)
    print(f"Total records: {total_records}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Shuffle the dataframe randomly
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate subset sizes
    size_25 = int(total_records * 0.25)
    size_50 = int(total_records * 0.50)
    size_75 = int(total_records * 0.75)
    size_100 = total_records
    
    print(f"Subset sizes: 25%={size_25}, 50%={size_50}, 75%={size_75}, 100%={size_100}")
    
    # Create progressive subsets
    subsets = {
        25: df_shuffled[:size_25],
        50: df_shuffled[:size_50], 
        75: df_shuffled[:size_75],
        100: df_shuffled[:size_100]
    }
    
    # Save each subset
    for percentage, subset_df in subsets.items():
        output_path = f"{output_dir}/groundtruth_subset_{percentage}.csv"
        subset_df.to_csv(output_path, index=False)
        print(f"Created {output_path}: {len(subset_df)} records ({percentage}%)")
    
    print("\nProgressive subset creation complete!")
    print("Each subset includes all records from the previous smaller subset plus additional random records.")

if __name__ == "__main__":
    main()