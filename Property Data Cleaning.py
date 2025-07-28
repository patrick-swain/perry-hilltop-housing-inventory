#!/usr/bin/env python
# coding: utf-8

# In[133]:


import pandas as pd

def create_and_append_inventory(
    property_data_path='propertydata.csv',
    geo_data_path='geoid.csv',
    output_csv_path='inventory.csv'
):
    """
    Creates a base inventory, finds properties that were missing from the geographic
    data file, and appends them to create a final, complete inventory.

    Args:
        property_data_path (str): Path to the main property data CSV.
        geo_data_path (str): Path to the geographic data CSV (geoid.csv).
        output_csv_path (str): Path to save the final inventory file.
    """
    print("--- Starting Combined Inventory Creation and Append Process ---")

    try:
        # --- 1. Load both datasets ---
        print(f"Attempting to read '{property_data_path}'...")
        df_property = pd.read_csv(property_data_path, low_memory=False)
        print(f"Successfully read '{property_data_path}'.")

        print(f"Attempting to read '{geo_data_path}'...")
        df_geo = pd.read_csv(geo_data_path)
        print(f"Successfully read '{geo_data_path}'.")

        # --- Part 1: Create the Base Inventory ---
        print("\n--- Part 1: Creating Base Inventory ---")
        
        # Define columns for the final inventory
        columns_to_keep = [
            'PARID', 'PROPERTYHOUSENUM', 'PROPERTYFRACTION', 'PROPERTYADDRESS',
            'PROPERTYCITY', 'PROPERTYSTATE', 'PROPERTYUNIT', 'PROPERTYZIP',
            'MUNICODE', 'MUNIDESC', 'NEIGHDESC', 'OWNERDESC',
            'CLASSDESC', 'USEDESC', 'LOTAREA', 'SALEDATE', 'SALEPRICE',
            'SALEDESC', 'PREVSALEDATE', 'PREVSALEPRICE', 'PREVSALEDATE2',
            'PREVSALEPRICE2', 'FAIRMARKETBUILDING', 'HOMESTEADFLAG','FAIRMARKETLAND',
            'FAIRMARKETTOTAL', 'STYLEDESC', 'STORIES', 'YEARBLT',
            'EXTFINISH_DESC', 'ROOFDESC', 'BASEMENTDESC', 'CONDITIONDESC',
            'TOTALROOMS', 'BEDROOMS', 'FULLBATHS', 'HALFBATHS',
            'HEATINGCOOLINGDESC', 'FIREPLACES', 'FINISHEDLIVINGAREA', 'ASOFDATE',
            'CHANGENOTICEADDRESS1'
        ]
        columns_to_append = ['PARID', 'NEIGHBORHOOD', 'PGH_COUNCIL_DISTRICT', 'PGH_WARD']
        neighborhood_filters = ['Fineview', 'Perry South']

        # Merge property data with geo data
        df_property['PARID'] = df_property['PARID'].astype(str)
        df_geo['PARID'] = df_geo['PARID'].astype(str)
        df_merged = pd.merge(df_property, df_geo[columns_to_append], on='PARID', how='left')
        
        # Filter for properties with reliable neighborhood data
        df_inventory_base = df_merged[df_merged['NEIGHBORHOOD'].isin(neighborhood_filters)].copy()
        print(f"Base inventory created with {len(df_inventory_base)} properties.")

        # --- Part 2: Find Properties Missing from GeoID File ---
        print("\n--- Part 2: Finding Properties Missing from GeoID File ---")
        property_parids = set(df_property['PARID'])
        geo_parids = set(df_geo['PARID'])
        missing_parids = property_parids - geo_parids
        
        df_missing_all = df_property[df_property['PARID'].isin(missing_parids)].copy()
        
        # Filter missing properties for Wards 25 & 26
        df_missing_all['extracted_ward'] = df_missing_all['MUNIDESC'].str.extract(r'(\d+)').astype(float)
        wards_to_check = [25, 26]
        df_missing_ward_filtered = df_missing_all[df_missing_all['extracted_ward'].isin(wards_to_check)]
        
        # Exclude specific neighborhoods
        neighborhoods_to_exclude = ['NORTHVIEW HEIGHTS', 'SUMMER HILL', 'PERRY NORTH']
        exclude_pattern = '|'.join(neighborhoods_to_exclude)
        df_missing_final = df_missing_ward_filtered[~df_missing_ward_filtered['NEIGHDESC'].str.contains(exclude_pattern, case=False, na=False)]
        print(f"Found {len(df_missing_final)} relevant properties that were missing from the geo file.")

        # --- Part 3: Combine, Finalize, and Save ---
        print("\n--- Part 3: Appending Missing Properties to Base Inventory ---")
        
        # Combine the base inventory with the missing properties
        df_combined = pd.concat([df_inventory_base, df_missing_final], ignore_index=True)
        
        # Drop duplicates just in case there's any overlap
        df_combined.drop_duplicates(subset=['PARID'], keep='first', inplace=True)
        print(f"Total properties after combining and removing duplicates: {len(df_combined)}")

        # Filter for only RESIDENTIAL properties
        if 'CLASSDESC' in df_combined.columns:
            print(f"\nFiltering for properties where CLASSDESC is 'RESIDENTIAL'...")
            df_final_filtered = df_combined[df_combined['CLASSDESC'] == 'RESIDENTIAL'].copy()
            print(f"DataFrame shape after filtering by CLASSDESC: {df_final_filtered.shape}")
        else:
            df_final_filtered = df_combined

        # Select and order the final columns
        final_columns_ordered = columns_to_keep + ['NEIGHBORHOOD', 'PGH_COUNCIL_DISTRICT', 'PGH_WARD']
        existing_final_columns = [col for col in final_columns_ordered if col in df_final_filtered.columns]
        df_to_save = df_final_filtered[existing_final_columns]
        
        # Save the final, complete inventory
        df_to_save.to_csv(output_csv_path, index=False)
        print(f"\nComplete inventory successfully saved to '{output_csv_path}'.")
        print(f"Final DataFrame shape: {df_to_save.shape}")

    except FileNotFoundError as e:
        print(f"\nError: A required file was not found. Please make sure the file paths are correct.")
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# --- Run the main function ---
if __name__ == '__main__':
    create_and_append_inventory()


# In[ ]:




