#!/usr/bin/env python
# coding: utf-8

# In[90]:


# =============================================================================
# ANNOTATED SCRIPT FOR GENERATING PHFCC REPORT VISUALIZATIONS
#
# Description:
# This script loads the 'inventory_appended.csv' file and systematically
# generates all the graphs and maps featured in the Perry South & Fineview
# Community Housing Inventory report. It is designed to be replicable,
# assuming you have the source CSV file.
#
#
# Instructions for use:
# 1. Make sure you have the required libraries installed. Open your terminal
#    or command prompt and run:
#    pip install pandas matplotlib seaborn numpy geopandas contextily
#
# 2. Place your data file, 'inventory_appended.csv', in the same folder
#    as this Python script.
#
# 3. Run the script from your terminal: python your_script_name.py
#
# 4. All output images and maps will be saved to a folder named
#    'Documents/PHFCC/' which will be created automatically.
# =============================================================================


# Import necessary libraries for data handling, plotting, and mapping
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import to_hex
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
import os

# --- 0. Setup and Data Loading ---
# This section loads the data and sets up foundational variables like
# file paths, color schemes, and inflation data.

# Load the master inventory file from the specified path
try:
    df = pd.read_csv('Documents/PHFCC/inventory_appended.csv')
    print("Successfully loaded 'inventory_appended.csv'.")
except FileNotFoundError:
    print("Error: 'inventory_appended.csv' not found. Please make sure the file is in the same directory as this script.")
    exit()

# Define the directory where all output files will be saved.
# The script will create this folder if it doesn't already exist.
output_dir = 'Documents/PHFCC/'
os.makedirs(output_dir, exist_ok=True)
print(f"All output files will be saved to: {output_dir}")


# Consumer Price Index (CPI) data for adjusting sales prices for inflation.
# The goal is to compare all sale prices in constant 2018 dollars.
cpi_data = {
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
    2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120,
    2018: 251.107
}
cpi_2018 = cpi_data[2018] # The reference year's CPI value

# Define a standard, colorblind-friendly color palette for ownership categories.
# This ensures that graphs are consistent and readable.
ownership_palette = {
    'Owner-Occupied': '#0077BB',             # Blue
    'Other Non-Owner-Occupied': '#009E73',     # Green
    'Corporate Non-Owner-Occupied': '#CC3311' # Red/Orange
}
# Define the order for the legend and axes to ensure logical presentation.
ownership_order = ['Owner-Occupied', 'Other Non-Owner-Occupied', 'Corporate Non-Owner-Occupied']

# Define the logical order for property conditions, from best to worst.
condition_order = ['EXCELLENT', 'VERY GOOD', 'GOOD', 'AVERAGE', 'FAIR', 'POOR', 'VERY POOR', 'UNSOUND']
# Create a color scale that maps this order to a green-to-red gradient.
# 'RdYlGn_r' is a color map where '_r' reverses it, making high values red and low values green.
# Here we reverse the input `linspace` to make green "good" (lower index) and red "bad" (higher index).
condition_colormap = plt.cm.get_cmap('RdYlGn', len(condition_order))
condition_colors_rgba = condition_colormap(np.linspace(1, 0, len(condition_order)))
# Convert the colors to a more web-friendly Hex format.
condition_palette_hex = {cond: to_hex(color) for cond, color in zip(condition_order, condition_colors_rgba)}


# --- 1. Data Processing and Categorization ---
# This section cleans the data and creates new categorical columns
# that are used in the analysis and visualizations.

# Remove any leading/trailing whitespace from the condition descriptions
# to prevent issues with categorization (e.g., " GOOD" vs "GOOD").
df['CONDITIONDESC'] = df['CONDITIONDESC'].str.strip()

# This function defines the logic for categorizing properties into ownership groups.
# The order of checks is important (Homestead -> Corporate -> Other).
def categorize_property_corrected(row):
    # If a property has a Homestead Exemption, it's considered Owner-Occupied.
    if row['HOMESTEADFLAG'] == 'HOM':
        return 'Owner-Occupied'
    # If not, check if the owner name contains corporate identifiers.
    elif any(keyword in str(row['OWNERDESC']).upper() for keyword in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
        return 'Corporate Non-Owner-Occupied'
    # Otherwise, it's a non-owner-occupied property owned by an individual.
    else:
        return 'Other Non-Owner-Occupied'
# Apply this function to every row in the dataframe to create the new 'ownership_category' column.
df['ownership_category'] = df.apply(categorize_property_corrected, axis=1)

# This function defines a broader set of ownership categories for the neighborhood comparison graph.
def categorize_ownership_broad(row):
    owner_desc_upper = str(row['OWNERDESC']).upper()
    # Check for public entities first.
    if any(keyword in owner_desc_upper for keyword in ['CITY OF PITTSBURGH', 'URA', 'HOUSING AUTH']):
        return 'Public'
    # Then check for corporate entities.
    elif 'CORPORATION' in owner_desc_upper:
        return 'Corporate/Investor'
    # All others are classified as Private.
    else:
        return 'Private'
df['ownership_type_broad'] = df.apply(categorize_ownership_broad, axis=1)

# --- 2. Standard Chart Generation ---
# This section contains the code to create and save all the non-map graphs.

# This is a helper function to standardize saving plots. It sets a title,
# ensures the layout is neat, and saves the figure to the output directory.
def save_plot(filename, title):
    full_path = os.path.join(output_dir, filename)
    plt.title(title, fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close() # Close the plot to free up memory
    print(f"Generated: {full_path}")

# --- Distribution Histograms ---
# This set of graphs shows the distribution of key numerical features.

# Distribution of Finished Living Area
plt.figure(figsize=(10, 6))
sns.histplot(df['FINISHEDLIVINGAREA'].dropna(), bins=50, kde=True)
plt.xlabel('Finished Living Area (sq. ft.)')
save_plot('dist_living_area.png', 'Distribution of Finished Living Area')

# Distribution of Lot Area
# Filter out the top 5% of largest lots (extreme outliers) to make the main distribution more visible.
lot_area_cap = df['LOTAREA'].quantile(0.95)
filtered_lot_area = df[df['LOTAREA'] < lot_area_cap]['LOTAREA']
plt.figure(figsize=(10, 6))
sns.histplot(filtered_lot_area.dropna(), bins=50, kde=True)
plt.xlabel('Lot Area (sq. ft.)')
save_plot('dist_lot_area.png', 'Distribution of Lot Area (Bottom 95%)')

# Distribution of Property Stories
plt.figure(figsize=(8, 6))
df['STORIES'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.xlabel('Number of Stories')
plt.ylabel('Number of Properties')
plt.xticks(rotation=0)
save_plot('dist_stories.png', 'Distribution of Property Stories')

# --- NEW: Year Built Distribution Graph with Median ---
plt.figure(figsize=(10, 6))
sns.histplot(df['YEARBLT'].dropna(), bins=30, kde=True)
# Add a vertical dashed line to indicate the median year built.
plt.axvline(df['YEARBLT'].median(), color='red', linestyle='--', label=f"Median: {int(df['YEARBLT'].median())}")
plt.legend()
plt.xlabel('Year Built')
plt.ylabel('Number of Properties')
save_plot('dist_year_built.png', 'Distribution of Year Built')


# --- Top 5 Property Use Types Graph ---
plt.figure(figsize=(10, 6))
# Get the 5 most common property uses
top_5_use = df['USEDESC'].value_counts().nlargest(5)
sns.barplot(x=top_5_use.index, y=top_5_use.values, palette='viridis')
plt.ylabel('Number of Properties')
plt.xlabel('Property Use Description')
plt.xticks(rotation=45, ha='right')
save_plot('top_5_property_use.png', 'Top 5 Property Use Types')


# --- Heatmap: Property Condition by Age Group (Logically Ordered) ---
# Create age brackets for properties.
df['age_group'] = pd.cut(df['YEARBLT'], bins=[0, 1940, 1970, 2000, 2025], labels=['Pre-1940', '1940-1969', '1970-1999', '2000+'])
# Get only the condition types that actually exist in the data.
unique_conditions_in_data = df['CONDITIONDESC'].unique()
present_ordered_conditions = [cond for cond in condition_order if cond in unique_conditions_in_data]
# Create a crosstab to calculate the proportion of conditions within each age group.
condition_age_crosstab = pd.crosstab(df['age_group'], df['CONDITIONDESC'], normalize='index') * 100
# Ensure the columns are in the correct logical order (best to worst).
condition_age_crosstab = condition_age_crosstab[present_ordered_conditions]
plt.figure(figsize=(12, 8))
sns.heatmap(condition_age_crosstab, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False)
save_plot('heatmap_condition_by_age.png', 'Proportion of Property Conditions by Age Group (%)')

# --- Financial Distress Bar Charts ---
# Calculate delinquency rate and average amount owed for each ownership group.
delinquency_rate = df.groupby('ownership_category')['Delinquent_Tax_Amount'].apply(lambda x: (x > 0).sum() / len(x) * 100).reindex(ownership_order)
avg_delinquent_amount = df[df['Delinquent_Tax_Amount'] > 0].groupby('ownership_category')['Delinquent_Tax_Amount'].mean().reindex(ownership_order)
# Create a figure with two subplots, side-by-side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.supxlabel('Ownership Category') # Add a shared x-axis label for the whole figure.

# Plot Delinquency Rate
delinquency_rate.plot(kind='bar', ax=ax1, color=[ownership_palette.get(c) for c in delinquency_rate.index])
ax1.set_title('Delinquency Rate by Ownership')
ax1.set_ylabel('Properties with Delinquent Taxes (%)')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45)

# Plot Average Delinquent Amount
avg_delinquent_amount.plot(kind='bar', ax=ax2, color=[ownership_palette.get(c) for c in avg_delinquent_amount.index])
ax2.set_title('Average Delinquent Amount by Ownership')
ax2.set_ylabel('Avg. Delinquent Tax Amount ($)')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45)
save_plot('financials_delinquency.png', 'Tax Delinquency Analysis by Ownership Category')

# Calculate lien rate and average lien amount for each ownership group.
lien_rate = df.groupby('ownership_category')['Lien_Count'].apply(lambda x: (x > 0).sum() / len(x) * 100).reindex(ownership_order)
avg_lien_amount = df[df['Lien_Count'] > 0].groupby('ownership_category')['Lien_Total_Amount'].mean().reindex(ownership_order)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.supxlabel('Ownership Category') # Add a shared x-axis label.

# Plot Lien Rate
lien_rate.plot(kind='bar', ax=ax1, color=[ownership_palette.get(c) for c in lien_rate.index])
ax1.set_title('Lien Rate by Ownership')
ax1.set_ylabel('Rate of Properties with Liens (%)')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45)

# Plot Average Lien Amount
avg_lien_amount.plot(kind='bar', ax=ax2, color=[ownership_palette.get(c) for c in avg_lien_amount.index])
ax2.set_title('Average Lien Amount by Ownership')
ax2.set_ylabel('Avg. Lien Amount on Properties ($)')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45)
save_plot('financials_liens.png', 'Property Lien Analysis by Ownership Category')

# Calculate PLI Violation Rate
df['has_pli_violation'] = df['pli_violation_status'].notna()
pli_violation_rate = df.groupby('ownership_category')['has_pli_violation'].mean() * 100
plt.figure(figsize=(10, 6))
pli_violation_rate.reindex(ownership_order).plot(kind='bar', color=[ownership_palette.get(c) for c in pli_violation_rate.reindex(ownership_order).index])
plt.xticks(rotation=45, ha='right')
save_plot('pli_violations_by_ownership.png', 'Rate of PLI Violations by Ownership')

# --- Condemnation Rate Graph ---
# A property is considered condemned if it has any entry in the 'Condemned_Property_Type' column.
df['is_condemned'] = df['Condemned_Property_Type'].notna()
condemnation_rate = df.groupby('ownership_category')['is_condemned'].mean() * 100
plt.figure(figsize=(10, 6))
condemnation_rate.reindex(ownership_order).plot(kind='bar', color=[ownership_palette.get(c) for c in condemnation_rate.reindex(ownership_order).index])
plt.ylabel('Condemnation Rate (%)')
plt.xlabel('Ownership Category')
plt.xticks(rotation=45, ha='right')
save_plot('condemnation_rate_by_ownership.png', 'Condemnation Rate by Ownership')


# --- Ownership Proportions by Neighborhood ---
neighborhood_crosstab = pd.crosstab(df['NEIGHBORHOOD'], df['ownership_type_broad'], normalize='index') * 100
neighborhood_crosstab.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
plt.xticks(rotation=0)
save_plot('ownership_by_neighborhood.png', 'Proportion of Ownership Types by Neighborhood')

# --- Sales Trends (Adjusted for Inflation) ---
# Filter for valid sales (price >= $10,000)
sales_df = df[(df['SALEPRICE'] >= 10000)].copy()
# Extract the year from the sale date.
sales_df['SALEYEAR'] = pd.to_datetime(sales_df['SALEDATE']).dt.year
# Filter for years where we have CPI data.
sales_df_recent = sales_df[sales_df['SALEYEAR'].isin(cpi_data.keys())].copy()
# Adjust the sale price to 2018 dollars.
sales_df_recent['ADJUSTED_PRICE_2018'] = sales_df_recent.apply(lambda row: row['SALEPRICE'] * (cpi_2018 / cpi_data[row['SALEYEAR']]), axis=1)
# Calculate sales volume and median price per year.
annual_sales_volume = sales_df['SALEYEAR'].value_counts().sort_index()
annual_median_price_adj = sales_df_recent.groupby('SALEYEAR')['ADJUSTED_PRICE_2018'].median()
# Create a figure with two subplots, one above the other.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
ax1.plot(annual_sales_volume.index, annual_sales_volume.values, marker='o')
ax1.set_title('Annual Sales Volume (Sales >= $10,000)')
ax2.plot(annual_median_price_adj.index, annual_median_price_adj.values, marker='o', color='green')
ax2.set_title('Annual Median Sale Price (Adjusted to 2018 Dollars)')
save_plot('sales_trends_inflation_adjusted.png', 'Sales Market Trends')


# --- 3. Static Map Generation with Street Basemap ---
# This section creates static scatter plots overlayed on a street map.

# Drop rows where location data is missing.
map_df = df.dropna(subset=['Latitude', 'Longitude']).copy()

# This is a helper function to create the static maps.
def create_static_map_with_basemap(filename, title, data, label_col_name, master_palette, master_order_list, fallback_color='#B0B0B0'):
    # Convert the pandas DataFrame to a GeoDataFrame, which is required for spatial plotting.
    gdf = gpd.GeoDataFrame(
        data, 
        geometry=gpd.points_from_xy(data.Longitude, data.Latitude),
        crs="EPSG:4326" # Set the coordinate reference system to WGS84 (standard for lat/lon)
    ).to_crs(epsg=3857) # Re-project to Web Mercator, which is compatible with most web maps.

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Map the labels in the data (e.g., 'AVERAGE') to their assigned color from the palette.
    # If a value is not in the palette, use the gray 'fallback_color'.
    colors = data[label_col_name].map(master_palette).fillna(fallback_color)
    
    # Plot the points on the map.
    gdf.plot(ax=ax, color=colors, markersize=20, alpha=0.7, edgecolor='k', linewidth=0.5)
    
    # Add the underlying street map from a public provider.
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Create the legend.
    unique_labels_in_data = data[label_col_name].dropna().unique()
    legend_handles = [
        mpatches.Patch(color=master_palette[label], label=label)
        for label in master_order_list if label in unique_labels_in_data
    ]
    
    # Add 'Not Available' to the legend if any gray points were plotted.
    if fallback_color in colors.values:
        legend_handles.append(mpatches.Patch(color=fallback_color, label='Not Available'))

    ax.legend(handles=legend_handles, title=label_col_name.replace('_', ' ').title(), loc='upper right', frameon=True)
    ax.set_axis_off() # Turn off the x/y axis labels (longitude/latitude) for a cleaner look.
    
    # Use the save_plot helper function to save the map.
    save_plot(filename, title)

# --- Map 1: Property Conditions ---
create_static_map_with_basemap(
    'map_static_conditions.png', 
    'Map of Property Conditions',
    map_df,
    label_col_name='CONDITIONDESC',
    master_palette=condition_palette_hex,
    master_order_list=condition_order
)

# --- Map 2: Ownership Types ---
create_static_map_with_basemap(
    'map_static_ownership.png',
    'Map of Ownership Types',
    map_df,
    label_col_name='ownership_category',
    master_palette=ownership_palette,
    master_order_list=ownership_order
)

print("\n--- Script Finished ---")


# In[91]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

# Note: The new visualizations require additional libraries.
# You may need to install them using pip:
# pip install geopandas contextily
import geopandas
import contextily as ctx


def generate_sales_graphs(file_path='Documents/PHFCC/inventory_appended.csv'):
    """
    Loads property data, calculates sales volume, inflation-adjusted
    median prices, and sale-to-assessed value ratio. Also generates a
    property condition comparison chart and a geospatial property condition map.
    This script will save five separate PNG files.

    Args:
        file_path (str): The path to the inventory CSV file.
    """
    # --- 1. Setup and Configuration ---
    # Use a clean and professional style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- IMPORTANT: Column Name Configuration ---
    # The script uses the column names you specified.
    # If your CSV uses different names, change them here.
    date_col = 'recent_sale_date'
    price_col = 'recent_sale_price'
    assessed_col = 'FAIRMARKETTOTAL'
    condition_col = 'CONDITIONDESC'
    owner_desc_col = 'OWNERDESC'
    homestead_col = 'HOMESTEADFLAG'
    lat_col = 'Latitude'
    lon_col = 'Longitude'

    # --- Consumer Price Index (CPI-U) Data for Inflation Adjustment ---
    # Source: Annual average CPI for All Urban Consumers (CPI-U), not seasonally adjusted.
    # A reasonable projection for 2024 and 2025 is included based on recent trends.
    cpi_data = {
        2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
        2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120,
        2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.970,
        2022: 292.655, 2023: 304.702,
        2024: 317.587,  # Projected based on ~4.2% YoY increase
        2025: 329.890   # Projected based on ~3.9% YoY increase
    }
    cpi_target_year = cpi_data[2025]  # Target for inflation adjustment

    # --- 2. Data Loading and Cleaning ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory as the script.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Verify that the necessary columns exist
    required_cols = [date_col, price_col, assessed_col, condition_col, owner_desc_col, homestead_col, lat_col, lon_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: One or more required columns are missing from the CSV file.")
        print("Please check the column names in your CSV or update the configuration at the top of the script.")
        print(f"Required: {required_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # --- 3. Feature Engineering and Data Preparation ---

    # A) Ownership Profiles
    def categorize_ownership(row):
        if row[homestead_col] == 'HOM':
            return 'Owner-Occupied'
        elif any(keyword in str(row[owner_desc_col]).upper() for keyword in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
            return 'Corporate Non-Owner-Occupied'
        else:
            return 'Other Non-Owner-Occupied'
    df['ownership_category'] = df.apply(categorize_ownership, axis=1)
    
    # B) Clean Condition Data
    df[condition_col] = df[condition_col].str.strip()


    # C) Sales Data Preparation
    sales_df = df[required_cols].copy()
    sales_df.dropna(subset=[date_col, price_col, assessed_col], inplace=True)
    sales_df[date_col] = pd.to_datetime(sales_df[date_col], errors='coerce')
    sales_df[price_col] = pd.to_numeric(sales_df[price_col], errors='coerce')
    sales_df[assessed_col] = pd.to_numeric(sales_df[assessed_col], errors='coerce')
    sales_df.dropna(subset=[date_col, price_col, assessed_col], inplace=True)
    sales_df = sales_df[(sales_df[price_col] > 1000) & (sales_df[assessed_col] > 0)].copy()

    sales_df['sale_year'] = sales_df[date_col].dt.year
    sales_df = sales_df[sales_df['sale_year'].isin(cpi_data.keys())].copy()

    def adjust_for_inflation(row):
        sale_year = int(row['sale_year'])
        sale_price = row[price_col]
        cpi_sale_year = cpi_data.get(sale_year)
        if cpi_sale_year:
            return sale_price * (cpi_target_year / cpi_sale_year)
        return np.nan

    sales_df['adjusted_price_2025'] = sales_df.apply(adjust_for_inflation, axis=1)
    sales_df['sale_to_assessed_ratio'] = sales_df['adjusted_price_2025'] / sales_df[assessed_col]

    # --- 4. Data Aggregation ---
    annual_sales_volume = sales_df['sale_year'].value_counts().sort_index()
    annual_median_price = sales_df.groupby('sale_year')['adjusted_price_2025'].median()
    annual_median_ratio = sales_df.groupby('sale_year')['sale_to_assessed_ratio'].median()

    # --- 5. Visualization (Separate Graphs) ---

    # Plot 1: Sales Volume
    plt.figure(figsize=(12, 7))
    plt.plot(annual_sales_volume.index, annual_sales_volume.values, marker='o', linestyle='-', color='b')
    plt.title('Annual Sales Volume in Perry South & Fineview', fontsize=16)
    plt.ylabel('Number of Properties Sold', fontsize=12)
    plt.xlabel('Year of Sale', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(annual_sales_volume.index.astype(int), rotation=45)
    plt.tight_layout()
    try:
        output_filename = 'sales_volume_trend.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Graph successfully saved as '{output_filename}'")
    except Exception as e:
        print(f"Could not save the sales volume graph. Error: {e}")
    plt.show()

    # Plot 2: Median Sales Price (Adjusted for Inflation)
    plt.figure(figsize=(12, 7))
    plt.plot(annual_median_price.index, annual_median_price.values, marker='s', linestyle='-', color='g')
    plt.title('Annual Median Sale Price (Adjusted to 2025 Dollars)', fontsize=16)
    plt.ylabel('Median Sale Price ($)', fontsize=12)
    plt.xlabel('Year of Sale', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax = plt.gca()
    formatter = mticker.FuncFormatter(lambda x, p: f'${int(x):,}')
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(annual_median_price.index.astype(int), rotation=45)
    plt.tight_layout()
    try:
        output_filename = 'median_price_trend.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Graph successfully saved as '{output_filename}'")
    except Exception as e:
        print(f"Could not save the median price graph. Error: {e}")
    plt.show()

    # Plot 3: Sale Price to Assessed Value Ratio
    plt.figure(figsize=(12, 7))
    plt.plot(annual_median_ratio.index, annual_median_ratio.values, marker='d', linestyle='-', color='purple')
    plt.title('Annual Median Ratio of Sale Price to Assessed Value', fontsize=16)
    plt.ylabel('Sale Price / Assessed Value Ratio', fontsize=12)
    plt.xlabel('Year of Sale', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='Sale Price = Assessed Value')
    plt.legend()
    plt.xticks(annual_median_ratio.index.astype(int), rotation=45)
    plt.tight_layout()
    try:
        output_filename = 'sale_to_assessed_ratio_trend.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Graph successfully saved as '{output_filename}'")
    except Exception as e:
        print(f"Could not save the ratio graph. Error: {e}")
    plt.show()
    
    # --- Other Visualizations ---

    # Plot 4: Condition by Ownership Profile (Stacked Bar Chart)
    condition_order = ['EXCELLENT', 'VERY GOOD', 'GOOD', 'AVERAGE', 'FAIR', 'POOR', 'VERY POOR', 'UNSOUND']
    ownership_order = ['Owner-Occupied', 'Other Non-Owner-Occupied', 'Corporate Non-Owner-Occupied']
    
    # Use a reversed colorblind-friendly palette that is logically ordered from good (green) to bad (red)
    color_map_palette = sns.color_palette("RdYlGn_r", len(condition_order))
    
    # Calculate proportions
    condition_proportions = pd.crosstab(df['ownership_category'], df[condition_col], normalize='index') * 100
    
    # Ensure all possible conditions are columns, fill missing with 0
    for cond in condition_order:
        if cond not in condition_proportions.columns:
            condition_proportions[cond] = 0
    condition_proportions = condition_proportions[condition_order] # Enforce logical order
    
    # Reorder the ownership categories for plotting
    condition_proportions = condition_proportions.reindex(ownership_order)

    ax = condition_proportions.plot(kind='bar', stacked=True, figsize=(14, 8), color=color_map_palette)
    plt.title('Proportion of Property Conditions by Ownership Profile', fontsize=16)
    plt.ylabel('Percentage of Properties (%)', fontsize=12)
    plt.xlabel('Ownership Profile', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Condition', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    try:
        output_filename = 'condition_by_ownership.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Graph successfully saved as '{output_filename}'")
    except Exception as e:
        print(f"Could not save the condition graph. Error: {e}")
    plt.show()

    # Plot 5: Geospatial Scatter Plot of Property Condition
    map_df = df[[lon_col, lat_col, condition_col]].copy()
    # Drop rows missing coordinates OR condition description
    map_df.dropna(subset=[lon_col, lat_col, condition_col], inplace=True)
    
    gdf = geopandas.GeoDataFrame(
        map_df, geometry=geopandas.points_from_xy(map_df[lon_col], map_df[lat_col]), crs="EPSG:4326"
    ).to_crs(epsg=3857) # Project to Web Mercator for basemap compatibility
    
    # Create the color palette dictionary for the map
    # Uses the same logical order and colors as the bar chart
    condition_palette_dict = {cond: color for cond, color in zip(condition_order, color_map_palette)}
    
    # Map condition values to colors for each point
    point_colors = gdf[condition_col].map(condition_palette_dict)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    gdf.plot(color=point_colors, ax=ax, markersize=20, alpha=0.8, edgecolor='k', linewidth=0.5)
    
    # Create legend handles manually to ensure correct order and labels
    legend_handles = [
        mpatches.Patch(color=condition_palette_dict[label], label=label)
        for label in condition_order if label in gdf[condition_col].unique()
    ]
    
    ax.legend(handles=legend_handles, title='Property Condition', loc='upper right', frameon=True)
    
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    plt.title('Property Condition in Perry South & Fineview', fontsize=18)
    ax.set_axis_off()
    plt.tight_layout()
    try:
        output_filename = 'property_condition_map.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Graph successfully saved as '{output_filename}'")
    except Exception as e:
        print(f"Could not save the map. Error: {e}")
    plt.show()


# --- Run the main function ---
if __name__ == '__main__':
    # Assuming 'inventory_appended.csv' is in the same directory as this script.
    # If it's located elsewhere, provide the full path.
    # For example: generate_sales_graphs(r'C:\Users\YourUser\Documents\inventory_appended.csv')
    generate_sales_graphs()


# In[92]:


import pandas as pd
import folium
from folium.map import MacroElement
from jinja2 import Template
import branca.colormap as cm

def create_interactive_condition_map(file_path='Documents/PHFCC/inventory_appended.csv', output_file='interactive_condition_map.html'):
    """
    Loads property data from a CSV file and generates an interactive HTML map
    displaying properties colored by their condition.

    Args:
        file_path (str): The path to the inventory CSV file.
        output_file (str): The filename for the output HTML map.
    """
    print("--- Creating Property Condition Map ---")
    # --- 1. Configuration ---
    condition_config = {
        'EXCELLENT': '#2ca02c', 'VERY GOOD': '#98df8a', 'GOOD': '#a2d64b',
        'AVERAGE': '#ffffbf', 'FAIR': '#fee08b', 'POOR': '#fdae61',
        'VERY POOR': '#f46d43', 'UNSOUND': '#d73027'
    }
    condition_order = ['EXCELLENT', 'VERY GOOD', 'GOOD', 'AVERAGE', 'FAIR', 'POOR', 'VERY POOR', 'UNSOUND']

    # --- 2. Data Loading and Cleaning ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    required_cols = ['Latitude', 'Longitude', 'CONDITIONDESC', 'PROPERTYADDRESS', 'NEIGHBORHOOD', 'HOMESTEADFLAG', 'OWNERDESC']
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing one or more required columns for the condition map.")
        return
    
    df.dropna(subset=['Latitude', 'Longitude', 'CONDITIONDESC'], inplace=True)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    df['CONDITIONDESC'] = df['CONDITIONDESC'].str.strip()

    # --- 3. Map Initialization ---
    m = folium.Map(location=[40.465, -80.015], zoom_start=14, tiles='CartoDB positron')

    # --- 4. Data Processing and Plotting ---
    def get_ownership_category(row):
        if row['HOMESTEADFLAG'] == 'HOM': return 'Owner-Occupied'
        if isinstance(row['OWNERDESC'], str) and any(kw in row['OWNERDESC'].upper() for kw in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
            return 'Corporate Non-Owner-Occupied'
        return 'Other Non-Owner-Occupied'

    plotted_count = 0
    for idx, row in df.iterrows():
        condition = row['CONDITIONDESC']
        if condition in condition_config:
            popup_html = f"<b>Address:</b> {row['PROPERTYADDRESS'] or 'N/A'}<br><b>Neighborhood:</b> {row['NEIGHBORHOOD'] or 'N/A'}<br><b>Condition:</b> {condition}<br><b>Ownership Type:</b> {get_ownership_category(row) or 'N/A'}"
            popup = folium.Popup(popup_html, max_width=300)
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']], radius=6, popup=popup, color="#000",
                weight=0.5, fill=True, fill_color=condition_config[condition], fill_opacity=0.8
            ).add_to(m)
            plotted_count += 1
    print(f"Successfully processed and plotted {plotted_count} properties for condition map.")

    # --- 5. Custom Map Legend ---
    legend_html_template = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 50px; right: 50px; width: 150px; height: auto; border:2px solid grey; z-index:9999; font-size:14px; background-color:rgba(255, 255, 255, 0.85); padding: 8px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <div style="text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 5px;">Property Condition</div>
        {% for condition in this.conditions %}
        <div style="margin-bottom: 4px;">
            <i style="background:{{ this.colors[condition] }}; width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.9; border: 1px solid #555; border-radius: 50%;"></i>
            {{ condition }}
        </div>
        {% endfor %}
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(legend_html_template)
    macro.conditions = condition_order
    macro.colors = condition_config
    m.add_child(macro)

    # --- 6. Save Map ---
    try:
        m.save(output_file)
        print(f"Successfully created interactive map: '{output_file}'\n")
    except Exception as e:
        print(f"An error occurred while saving the condition map: {e}\n")


def create_interactive_value_map(file_path='Documents/PHFCC/inventory_appended.csv', output_file='interactive_value_map.html'):
    """
    Loads property data from a CSV file and generates an interactive HTML map
    displaying properties colored by their assessed value.

    Args:
        file_path (str): The path to the inventory CSV file.
        output_file (str): The filename for the output HTML map.
    """
    print("--- Creating Property Value Map ---")
    # --- 1. Data Loading and Cleaning ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    required_cols = ['Latitude', 'Longitude', 'FAIRMARKETTOTAL', 'PROPERTYADDRESS', 'NEIGHBORHOOD']
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing one or more required columns for the value map.")
        return

    df.dropna(subset=['Latitude', 'Longitude', 'FAIRMARKETTOTAL'], inplace=True)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['FAIRMARKETTOTAL'] = pd.to_numeric(df['FAIRMARKETTOTAL'], errors='coerce')
    df.dropna(subset=['Latitude', 'Longitude', 'FAIRMARKETTOTAL'], inplace=True)
    
    # --- 2. Color Scale for Values ---
    # To handle outliers, we'll cap the values at the 95th percentile for the colormap
    cap = df['FAIRMARKETTOTAL'].quantile(0.95)
    df_filtered = df[df['FAIRMARKETTOTAL'] <= cap]

    # Create a continuous color map from red (low value) to green (high value)
    colormap = cm.LinearColormap(colors=['red', 'yellow', 'green'],
                                 vmin=df_filtered['FAIRMARKETTOTAL'].min(),
                                 vmax=cap)
    colormap.caption = 'Assessed Property Value ($)'
    
    # --- 3. Map Initialization ---
    m = folium.Map(location=[40.465, -80.015], zoom_start=14, tiles='CartoDB positron')

    # --- 4. Data Processing and Plotting ---
    plotted_count = 0
    for idx, row in df.iterrows():
        value = row['FAIRMARKETTOTAL']
        popup_html = f"<b>Address:</b> {row['PROPERTYADDRESS'] or 'N/A'}<br><b>Neighborhood:</b> {row['NEIGHBORHOOD'] or 'N/A'}<br><b>Assessed Value:</b> ${value:,.0f}"
        popup = folium.Popup(popup_html, max_width=300)

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            popup=popup,
            color="#000",
            weight=0.5,
            fill=True,
            fill_color=colormap(value),
            fill_opacity=0.8
        ).add_to(m)
        plotted_count += 1
    print(f"Successfully processed and plotted {plotted_count} properties for value map.")

    # --- 5. Add Colormap Legend ---
    m.add_child(colormap)

    # --- 6. Save Map ---
    try:
        m.save(output_file)
        print(f"Successfully created interactive map: '{output_file}'\n")
    except Exception as e:
        print(f"An error occurred while saving the value map: {e}\n")


# --- Run the main functions ---
# The functions are called directly to ensure the script runs when executed.
create_interactive_condition_map()
create_interactive_value_map()


# In[93]:


import pandas as pd
import folium

def create_interactive_vacant_map(file_path='Documents/PHFCC/inventory_appended.csv', output_file='interactive_vacant_map.html'):
    """
    Loads property data from a CSV file and generates an interactive HTML map
    displaying the locations of vacant parcels.

    Args:
        file_path (str): The path to the inventory CSV file.
        output_file (str): The filename for the output HTML map.
    """
    print("--- Creating Interactive Map of Vacant Parcels ---")
    
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
        return

    # Check for required columns
    required_cols = ['Latitude', 'Longitude', 'USEDESC', 'PROPERTYADDRESS', 'NEIGHBORHOOD', 'OWNERDESC']
    if not all(col in df.columns for col in required_cols):
        print("Error: The CSV file is missing one or more required columns.")
        print(f"Required: {required_cols}")
        return

    # Filter for vacant parcels, handling potential missing values in USEDESC
    df_vacant = df[df['USEDESC'].str.contains('VACANT', na=False, case=False)].copy()

    # Clean coordinate data
    df_vacant.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    df_vacant['Latitude'] = pd.to_numeric(df_vacant['Latitude'], errors='coerce')
    df_vacant['Longitude'] = pd.to_numeric(df_vacant['Longitude'], errors='coerce')
    df_vacant.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    print(f"Found {len(df_vacant)} vacant parcels to map.")

    # --- 2. Map Initialization ---
    # Center the map on the approximate location of the neighborhoods
    map_center = [40.465, -80.015]
    m = folium.Map(location=map_center, zoom_start=14, tiles='CartoDB positron')

    # --- 3. Data Processing and Plotting ---
    # Iterate over the vacant parcels DataFrame to add a circle marker for each
    for idx, row in df_vacant.iterrows():
        
        # Create the HTML content for the popup with property information
        popup_html = f"""
        <b>Address:</b> {row['PROPERTYADDRESS'] or 'N/A'}<br>
        <b>Neighborhood:</b> {row['NEIGHBORHOOD'] or 'N/A'}<br>
        <b>Owner Description:</b> {row['OWNERDESC'] or 'N/A'}<br>
        <b>Use:</b> {row['USEDESC'] or 'N/A'}
        """
        popup = folium.Popup(popup_html, max_width=300)

        # Add a circle marker to the map for the vacant lot
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=popup,
            color="#d95f02",  # A distinct orange color
            weight=1,
            fill=True,
            fill_color="#d95f02",
            fill_opacity=0.7
        ).add_to(m)

    # --- 4. Save Map to HTML File ---
    try:
        m.save(output_file)
        print(f"\nSuccessfully created interactive map: '{output_file}'")
    except Exception as e:
        print(f"An error occurred while saving the map: {e}")

# --- Run the main function ---
create_interactive_vacant_map()


# In[ ]:




