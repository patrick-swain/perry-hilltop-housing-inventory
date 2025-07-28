#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np # Retained for potential use if NaNs are involved.

def standardize_join_key(df, standard_key_name):
    """
    Standardizes common variations of a parcel ID column ('pin', 'PIN', 'PARID', 'parcel_id')
    to a single consistent name.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        standard_key_name (str): The desired final name for the key column (e.g., 'pin').
        
    Returns:
        pd.DataFrame: The DataFrame with the standardized key column.
    """
    # Define possible variations of the join key, in order of preference
    key_variations = [
        standard_key_name, # The ideal name
        'pin',
        'PIN',
        'PARID',
        'parcel_id'
    ]
    # Remove duplicates while preserving order
    key_variations = list(dict.fromkeys(key_variations))
    
    found_key = None
    
    for key in key_variations:
        if key in df.columns:
            found_key = key
            break
            
    if found_key and found_key != standard_key_name:
        print(f"Renaming join key column '{found_key}' to '{standard_key_name}'.")
        df.rename(columns={found_key: standard_key_name}, inplace=True)
        
    return df


def add_multiple_columns_from_sources(target_df_path, sources_config, global_join_key_column, output_df_path, filter_column=None, filter_values=None):
    """
    Draws specified columns from multiple source datasets and adds them to a target dataset,
    joined by a specified global key column. Allows for renaming of drawn columns.
    Optionally filters the final dataset.

    Args:
        target_df_path (str): Filepath for the target CSV dataset.
        sources_config (list): A list of dictionaries, where each dictionary defines a source:
            {'path': 'path/to/source.csv', 'columns_to_draw': {'original_name': 'new_name', ...}}
        global_join_key_column (str): The name of the column to use for joining across all files.
        output_df_path (str): Filepath to save the modified target dataset.
        filter_column (str, optional): The column to filter on. Defaults to None.
        filter_values (list or str, optional): The value or list of values to filter for in the filter_column.
    """
    try:
        target_df = pd.read_csv(target_df_path)
        print(f"Loaded target dataset: {target_df_path} (Shape: {target_df.shape})")

        # Standardize join key in target_df
        target_df = standardize_join_key(target_df, global_join_key_column)
        
        if global_join_key_column not in target_df.columns:
            print(f"Error: Global join key '{global_join_key_column}' (or a variant) not found in target '{target_df_path}'.")
            return
        
        target_df[global_join_key_column] = target_df[global_join_key_column].astype(str)

        for i, source_info in enumerate(sources_config):
            source_df_path = source_info['path']
            columns_to_draw_map = source_info['columns_to_draw'] # This is now a dictionary
            
            print(f"\nProcessing source {i+1}/{len(sources_config)}: {source_df_path}")
            try:
                source_df = pd.read_csv(source_df_path)
            except FileNotFoundError:
                print(f"Error: Source file '{source_df_path}' not found. Skipping this source.")
                continue
            except Exception as e:
                print(f"Error loading source file '{source_df_path}': {e}. Skipping this source.")
                continue

            # Standardize join key in source_df
            source_df = standardize_join_key(source_df, global_join_key_column)

            if global_join_key_column not in source_df.columns:
                print(f"Error: Global join key '{global_join_key_column}' (or a variant) not found in source '{source_df_path}'. Skipping this source.")
                continue
            
            original_column_names = list(columns_to_draw_map.keys())
            missing_cols_in_source = [col for col in original_column_names if col not in source_df.columns]
            if missing_cols_in_source:
                print(f"Error: The following columns to draw were not found in source '{source_df_path}': {missing_cols_in_source}. Skipping this source.")
                continue
            
            source_df[global_join_key_column] = source_df[global_join_key_column].astype(str)
            
            # Select only the join key and the columns to draw from the source
            cols_for_subset = [global_join_key_column] + original_column_names
            cols_for_subset = list(dict.fromkeys(cols_for_subset)) 
            
            source_subset_df = source_df[cols_for_subset].copy()
            source_subset_df = source_subset_df.drop_duplicates(subset=[global_join_key_column], keep='first')
            
            # Rename the columns in the subset before merging
            source_subset_df.rename(columns=columns_to_draw_map, inplace=True)
            print(f"Renaming columns: {columns_to_draw_map}")

            # Perform a left merge.
            target_df = pd.merge(target_df, source_subset_df, on=global_join_key_column, how='left')
            print(f"Merged columns from {source_df_path}. Target shape: {target_df.shape}")

        # --- Apply filtering if specified ---
        if filter_column and filter_values is not None: 
            if not isinstance(filter_values, list):
                filter_values_list = [filter_values]
            else:
                filter_values_list = filter_values

            print(f"\nApplying filter: '{filter_column}' is in {filter_values_list}")
            if filter_column in target_df.columns:
                original_row_count = len(target_df)
                target_df = target_df[target_df[filter_column].isin(filter_values_list)]
                print(f"Filtered data. Shape before filter: ({original_row_count}, {target_df.shape[1]}), Shape after filter: {target_df.shape}")
                if target_df.empty:
                    print(f"Warning: Filter resulted in an empty dataset. No rows matched '{filter_column}' in {filter_values_list}.")
            else:
                print(f"Warning: Filter column '{filter_column}' not found in the final dataset. Filtering skipped.")
        elif filter_column or filter_values is not None:
            print(f"Warning: Both filter_column and filter_values must be provided for filtering. Filtering skipped.")


        # Save the result
        target_df.to_csv(output_df_path, index=False)
        print(f"\nSuccessfully processed all sources and saved to '{output_df_path}'.")
        print(f"Final dataset shape: {target_df.shape}")

    except FileNotFoundError:
        print(f"Error: Target file '{target_df_path}' was not found.")
    except Exception as e:
        print(f"An critical error occurred: {e}")

# --- Template for User Input ---
if __name__ == "__main__":
    # ** PLEASE FILL IN YOUR DETAILS BELOW **

    # 1. Path to your target CSV file (the file you want to add columns TO)
    user_target_df_path = "inventory.csv"

    # 2. Configuration for source files and columns.
    #    Each item in the list is a dictionary:
    #    - 'path': path to the source CSV file.
    #    - 'columns_to_draw': a dictionary mapping {'original_name': 'new_name'}
    user_sources_config = [
        {
            'path': "delinquency.csv",
            'columns_to_draw': {"current_delq_tax": "Delinquent_Tax_Amount"}
        },
        {
            'path': "coordinates.csv",
            'columns_to_draw': {"x": "Longitude", "y": "Latitude"}
        },
        {
            'path': "liens.csv",
            'columns_to_draw': {"number": "Lien_Count", "total_amount": "Lien_Total_Amount"}
        },
        {
            'path': "conservatorships.csv",
            'columns_to_draw': {"case_id": "Conservatorship_Case_ID","party_name": "conservatorship_party_name","last_activity": "conservatorship_last_activity"}
        },
        {
            'path': "condemned.csv",
            'columns_to_draw': {"property_type": "Condemned_Property_Type","date": "condemned_date"}
        },
         {
            'path': "pliviolations.csv",
            'columns_to_draw': {"status": "pli_violation_status","investigation_date":"pli_investigation_date","investigation_outcome":"pli_investigation_outcome"}
        },
        {
            'path': "sales.csv",
            'columns_to_draw': {"SALEDATE": "recent_sale_date","PRICE":"recent_sale_price"}
        },
        # Add more source configurations as needed:
        # {
        #     'path': "path/to/another_source.csv",
        #     'columns_to_draw': {"some_other_column": "new_name_1", "yet_another_column": "new_name_2"}
        # },
    ]

    # 3. Name of the common column used for joining across ALL files (e.g., 'pin', 'ID')
    #    The script will look for this name, or 'PIN', or 'PARID', or 'parcel_id' in the CSVs and standardize them.
    user_global_join_key_column = "pin"

    # 4. Path where you want to save the new CSV file (with all added columns)
    user_output_df_path = "inventory_appended.csv"

    # 5. Optional: Specify column and value(s) for filtering the final output.
    #    Set to None if no filtering is needed.
    #    For multiple OR values, provide a list.
    user_filter_column_name = "NEIGHBORHOOD"
    user_filter_column_values = ["Fineview", "Perry South"] # Filter for these two neighborhoods
    user_filter_column_name = None # Example: to disable filtering
    user_filter_column_values = None # Example: to disable filtering


    # --- Do not modify below this line  ---
    
    print(f"Starting script with user-defined parameters...")
    print(f"Target file: {user_target_df_path}")
    print(f"Global Join key: {user_global_join_key_column}")
    print(f"Output file: {user_output_df_path}")
    if user_filter_column_name and user_filter_column_values is not None:
        print(f"Filter: {user_filter_column_name} is in {user_filter_column_values}")
    
    # Basic check if paths still look like placeholders (optional, for user guidance)
    if "path/to/your/" in user_target_df_path or \
       "path/to/your/" in user_output_df_path or \
       any("path/to/" in src['path'] for src in user_sources_config if isinstance(src, dict) and 'path' in src) or \
       any("NameOfColumn" in col for src in user_sources_config if isinstance(src, dict) and 'columns_to_draw' in src for col in src['columns_to_draw']):
        print("\n*********************************************************************")
        print("INFO: Using generic placeholder names for some files/columns.")
        print("Please ensure the actual file paths and column names are correctly set")
        print("if these are not the intended dummy values.")
        print("*********************************************************************\n")
    
    add_multiple_columns_from_sources(
        target_df_path=user_target_df_path,
        sources_config=user_sources_config,
        global_join_key_column=user_global_join_key_column,
        output_df_path=user_output_df_path,
        filter_column=user_filter_column_name,
        filter_values=user_filter_column_values
    )
    print("\nScript execution finished.")


# In[ ]:




