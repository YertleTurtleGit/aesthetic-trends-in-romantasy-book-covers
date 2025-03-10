import pandas as pd
import glob

"""" After this script run duplication_delete.py to remove duplicates and get the earliest release date."""

def merge_csv_files():
    # Get all CSV files that match the pattern
    files = sorted(glob.glob('amazon_romantic_fantasy_with_release_dates*.csv'))
    
    # List to store all dataframes
    dfs = []
    
    # Current ID counter
    current_id = 1
    
    # Process each file
    for file in files:
        print(f"Processing {file}...")
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Update IDs
        num_rows = len(df)
        df['ID'] = range(current_id, current_id + num_rows)
        
        # Add to list of dataframes
        dfs.append(df)
        
        # Update current_id for next file
        current_id += num_rows
        
        print(f"Added {num_rows} rows. Next ID will be {current_id}")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged dataframe
    merged_df.to_csv('merged_amazon_romantic_fantasy.csv', index=False, encoding='utf-8-sig')
    print(f"\nMerged file saved with {len(merged_df)} total rows")
    
    return merged_df

if __name__ == "__main__":
    merged_df = merge_csv_files()