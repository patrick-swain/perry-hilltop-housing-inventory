#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
# Note: This script requires the 'thefuzz' library for fuzzy matching.
# Install it using: pip install thefuzz python-Levenshtein
from thefuzz import process

def standardize_address(address_series):
    """
    Standardizes a series of addresses to improve matching. This version uses
    a more aggressive cleaning process to handle more inconsistencies.
    """
    if not isinstance(address_series, pd.Series):
        return address_series

    std_address = address_series.astype(str).str.upper().str.strip()
    std_address = std_address.str.replace(r'[^\w\s]', '', regex=True)
    
    replacements = {
        r'\bAVENUE\b': 'AVE', r'\bSTREET\b': 'ST', r'\bROAD\b': 'RD',
        r'\bDRIVE\b': 'DR', r'\bLANE\b': 'LN', r'\bBOULEVARD\b': 'BLVD',
        r'\bCOURT\b': 'CT', r'\bSQUARE\b': 'SQ', r'\bTERRACE\b': 'TER',
        r'\bPLACE\b': 'PL', r'\bCIRCLE\b': 'CIR', r'\bNORTH\b': 'N',
        r'\bSOUTH\b': 'S', r'\bEAST\b': 'E', r'\bWEST\b': 'W',
        r'\b(APT|UNIT|#|STE|SUITE)\s*\w*': ''
    }
    
    for old, new in replacements.items():
        std_address = std_address.str.replace(old, new, regex=True)
        
    std_address = std_address.str.replace(r'\s+PITTSBURGH\s*(PA)?\s*\d{5}$', '', regex=True)
    std_address = std_address.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return std_address

def merge_owner_data_fuzzy(inventory_file='inventory_appended.csv', mailing_labels_file='Mailing Labels.csv', output_file='inventory_appended.csv', similarity_threshold=90):
    """
    Merges owner data into the main inventory using fuzzy string matching on addresses.

    Args:
        inventory_file (str): Path to the main inventory CSV.
        mailing_labels_file (str): Path to the CSV with new owner data.
        output_file (str): Name for the final, merged CSV file.
        similarity_threshold (int): The minimum similarity score (0-100) to consider a match.
    """
    print("--- Starting Data Merge Process with Fuzzy Matching ---")
    
    # --- 1. Load both datasets ---
    try:
        df_inventory = pd.read_csv(inventory_file, low_memory=False)
        print(f"Successfully loaded '{inventory_file}' ({len(df_inventory)} rows).")
    except FileNotFoundError:
        print(f"Error: The main inventory file '{inventory_file}' was not found.")
        return

    try:
        df_owners = pd.read_csv(mailing_labels_file)
        print(f"Successfully loaded '{mailing_labels_file}' ({len(df_owners)} rows).")
    except FileNotFoundError:
        print(f"Error: The mailing labels file '{mailing_labels_file}' was not found.")
        return

    # --- 2. Prepare Data and Standardize Addresses ---
    owner_cols = {
        'Property Full Address': 'join_address_temp',
        'Owner 1 Full': 'Verified_Owner',
        'Tax Full Address': 'Verified_Mailing_Address'
    }
    
    if not all(col in df_owners.columns for col in owner_cols.keys()):
        print("Error: The mailing labels file is missing required columns.")
        return
        
    df_owners_subset = df_owners[list(owner_cols.keys())].copy()
    df_owners_subset.rename(columns=owner_cols, inplace=True)

    df_inventory['PROPERTYHOUSENUM'] = pd.to_numeric(df_inventory['PROPERTYHOUSENUM'], errors='coerce').astype('Int64').astype(str)
    df_inventory['PROPERTYFRACTION'] = df_inventory['PROPERTYFRACTION'].astype(str).str.replace(r'\s*\d/\d\s*', '', regex=True).str.strip()
    inventory_address_raw = (df_inventory['PROPERTYHOUSENUM'].str.replace('<NA>', '', regex=False) + ' ' + df_inventory['PROPERTYADDRESS'].astype(str) + ' ' + df_inventory['PROPERTYFRACTION']).str.strip()
    
    df_inventory['join_address'] = standardize_address(inventory_address_raw)
    df_owners_subset['join_address'] = standardize_address(df_owners_subset['join_address_temp'])
    df_owners_subset.drop(columns=['join_address_temp'], inplace=True)
    df_owners_subset.drop_duplicates(subset=['join_address'], inplace=True)
    
    owner_address_choices = df_owners_subset['join_address'].tolist()
    
    # --- 3. Perform Fuzzy Matching ---
    unique_inventory_addresses = df_inventory['join_address'].unique()
    matches = []
    print(f"\nPerforming fuzzy matching on {len(unique_inventory_addresses)} unique addresses... This may take a few moments.")
    
    for inv_addr in unique_inventory_addresses:
        best_match = process.extractOne(inv_addr, owner_address_choices, score_cutoff=similarity_threshold)
        
        if best_match:
            # **CHANGE**: Removed 'match_score' from the appended data
            matches.append({'join_address': inv_addr, 'matched_address': best_match[0]})
        else:
            matches.append({'join_address': inv_addr, 'matched_address': None})
            
    df_matches = pd.DataFrame(matches)

    # --- 4. Merge Dataframes ---
    df_merged = pd.merge(df_inventory, df_matches, on='join_address', how='left')
    
    df_merged = pd.merge(
        df_merged, 
        df_owners_subset, 
        left_on='matched_address', 
        right_on='join_address', 
        how='left',
        suffixes=('', '_owner')
    )
    
    # Clean up temporary columns
    df_merged.drop(columns=['join_address', 'join_address_owner', 'matched_address'], inplace=True)

    num_matched = df_merged['Verified_Owner'].notna().sum()
    print(f"\nMerge complete. Matched {num_matched} of {len(df_inventory)} properties using fuzzy matching.")
    print(f"The final file will have {len(df_merged)} rows.")

    # --- 5. Save the Final File ---
    try:
        # **CHANGE**: Output file is now the original inventory file, overwriting it.
        df_merged.to_csv(output_file, index=False)
        print(f"--- Successfully saved the merged data back to '{output_file}' ---")
    except Exception as e:
        print(f"Error saving the final file: {e}")

# --- Run the main function ---
if __name__ == '__main__':
    merge_owner_data_fuzzy()


# In[ ]:




