#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import numpy as np

def generate_report_analysis(file_path='inventory_appended.csv'):
    """
    Loads property data from the specified CSV file and runs a series of
    analyses to generate the key numbers and tables from the PHFCC housing inventory report.

    Args:
        file_path (str): The path to the inventory CSV file.
    """
    # --- 1. Column Name Configuration ---
    # Standardizing column names used throughout the script for easy updates.
    assessed_col = 'FAIRMARKETTOTAL'
    condition_col = 'CONDITIONDESC'
    delinquent_tax_col = 'Delinquent_Tax_Amount'
    homestead_col = 'HOMESTEADFLAG'
    hvac_col = 'HEATINGCOOLINGDESC'
    lien_count_col = 'Lien_Count'
    lien_total_col = 'Lien_Total_Amount'
    neighborhood_col = 'NEIGHBORHOOD'
    owner_desc_col = 'OWNERDESC'
    address_col = 'PROPERTYADDRESS'
    price_col = 'recent_sale_price'
    sale_date_col = 'recent_sale_date'
    year_built_col = 'YEARBLT'
    fireplace_col = 'FIREPLACES'
    condemned_col = 'Condemned_Property_Type'

    # --- 2. Data Loading and Initial Cleaning ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"--- Successfully loaded '{file_path}' ---\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- 3. Feature Engineering ---
    # A) Create Ownership Category
    def categorize_ownership(row):
        if row[homestead_col] == 'HOM':
            return 'Owner-Occupied'
        elif any(keyword in str(row[owner_desc_col]).upper() for keyword in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
            return 'Corporate Non-Owner-Occupied'
        else:
            return 'Other Non-Owner-Occupied'
    df['ownership_category'] = df.apply(categorize_ownership, axis=1)

    # B) Clean numeric and categorical columns
    df[year_built_col] = pd.to_numeric(df[year_built_col], errors='coerce').replace(0, np.nan)
    df[condition_col] = df[condition_col].str.strip()
    df[delinquent_tax_col] = pd.to_numeric(df[delinquent_tax_col], errors='coerce').fillna(0)
    df[lien_count_col] = pd.to_numeric(df[lien_count_col], errors='coerce').fillna(0)
    df[lien_total_col] = pd.to_numeric(df[lien_total_col], errors='coerce').fillna(0)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[sale_date_col] = pd.to_datetime(df[sale_date_col], errors='coerce')

    # C) Create Street Name for block-level analysis
    # This regex removes leading numbers (house number) and fractions to isolate the street name.
    df['street_name'] = df[address_col].str.replace(r'^\d+\s*(\d/\d\s*)?', '', regex=True).str.strip()


    # --- 4. Report Analysis Generation ---
    print(">>> SECTION: SCOPE")
    fineview_count = df[df[neighborhood_col] == 'Fineview'].shape[0]
    perry_south_count = df[df[neighborhood_col] == 'Perry South'].shape[0]
    print(f"Total Properties: {df.shape[0]}")
    print(f"  - Fineview: {fineview_count}")
    print(f"  - Perry South: {perry_south_count}")
    print(f"Total Data Fields (Columns): {df.shape[1]}\n")

    print(">>> SECTION: AGING HOUSING STOCK")
    median_year = df[year_built_col].median()
    typical_age = 2025 - median_year
    pre_1940_count = df[df[year_built_col] < 1940].shape[0]
    valid_year_count = df[year_built_col].notna().sum()
    pre_1940_percent = (pre_1940_count / valid_year_count) * 100 if valid_year_count > 0 else 0
    print(f"Median Year Built: {int(median_year)}")
    print(f"Typical Home Age (as of 2025): {int(typical_age)} years")
    print(f"Percentage of Homes Built Before 1940: {pre_1940_percent:.1f}%\n")

    print(">>> TABLE: HVAC SYSTEM DESCRIPTION")
    hvac_counts = df[hvac_col].value_counts().reset_index()
    hvac_counts.columns = ['HVAC System Description', 'Number of Properties']
    print(hvac_counts.to_string(index=False))
    fireplace_percent = (df[fireplace_col] > 0).sum() / df[fireplace_col].notna().sum() * 100
    print(f"\nPercentage with one or more fireplaces: {fireplace_percent:.1f}%\n")

    print(">>> TABLE: NEIGHBORHOOD SUMMARY")
    summary_data = []
    for name, group_df in [('Fineview', df[df[neighborhood_col] == 'Fineview']),
                           ('Perry South', df[df[neighborhood_col] == 'Perry South']),
                           ('Overall', df)]:
        total_props = group_df.shape[0]
        if total_props == 0: continue
        summary = {
            '': name,
            'Total Properties': total_props,
            'Median Assessed Value': f"${group_df[assessed_col].median():,.0f}",
            'Median Year Built': int(group_df[year_built_col].median()),
            '% Owner-Occupied (Proxy)': f"{(group_df[homestead_col] == 'HOM').sum() / total_props * 100:.1f}%",
            'Tax Delinquency Rate': f"{(group_df[delinquent_tax_col] > 0).sum() / total_props * 100:.1f}%"
        }
        summary_data.append(summary)
    summary_df = pd.DataFrame(summary_data).set_index('')
    print(summary_df.to_string())
    print("\n")

    print(">>> TABLE: OWNERSHIP PROFILE")
    ownership_counts = df['ownership_category'].value_counts()
    ownership_percent = df['ownership_category'].value_counts(normalize=True) * 100
    ownership_df = pd.DataFrame({'Property Count': ownership_counts, 'Percentage of Total': ownership_percent.map('{:.1f}%'.format)})
    print(ownership_df.to_string())
    print("\n")

    print(">>> TABLE: FINANCIAL DISTRESS - TAX DELINQUENCY")
    distress_data = []
    for name, group_df in df.groupby('ownership_category'):
        delinquent_group = group_df[group_df[delinquent_tax_col] > 0]
        rate = (delinquent_group.shape[0] / group_df.shape[0]) * 100 if group_df.shape[0] > 0 else 0
        avg_amount = delinquent_group[delinquent_tax_col].mean()
        distress_data.append({
            'Ownership Profile': name,
            'Delinquency Rate': f"{rate:.1f}%",
            'Avg. Delinquent Amount': f"${avg_amount:,.2f}"
        })
    distress_df = pd.DataFrame(distress_data).set_index('Ownership Profile')
    print(distress_df.to_string())
    print("\n")

    print(">>> TABLE: FINANCIAL DISTRESS - PROPERTY LIENS")
    lien_data = []
    for name, group_df in df.groupby('ownership_category'):
        lien_group = group_df[group_df[lien_count_col] > 0]
        rate = (lien_group.shape[0] / group_df.shape[0]) * 100 if group_df.shape[0] > 0 else 0
        avg_amount = lien_group[lien_total_col].mean()
        lien_data.append({
            'Property Category': name,
            'Rate of Properties with Liens': f"{rate:.1f}%",
            'Average Lien Amount': f"${avg_amount:,.2f}"
        })
    lien_df = pd.DataFrame(lien_data).set_index('Property Category')
    print(lien_df.to_string())
    print("\n")

    print(">>> TABLE: PHYSICAL CONDITION ANALYSIS BY OWNERSHIP")
    condition_order = ['EXCELLENT', 'VERY GOOD', 'GOOD', 'AVERAGE', 'FAIR', 'POOR', 'VERY POOR', 'UNSOUND']
    condition_proportions = pd.crosstab(df['ownership_category'], df[condition_col], normalize='index') * 100
    for cond in condition_order:
        if cond not in condition_proportions.columns:
            condition_proportions[cond] = 0
    condition_proportions = condition_proportions[condition_order]
    # Format to one decimal place with a '%' sign
    for col in condition_proportions.columns:
        condition_proportions[col] = condition_proportions[col].map('{:.1f}%'.format)
    print(condition_proportions.to_string())
    print("\n")

    print(">>> SECTION: CONDEMNATION AND PLI VIOLATIONS")
    df['is_condemned'] = df[condemned_col].notna()
    condemnation_rate = df.groupby('ownership_category')['is_condemned'].mean() * 100
    print("Condemnation Rate by Ownership:")
    for name, rate in condemnation_rate.items():
        print(f"  - {name}: {rate:.1f}%")
    print("\n")

    print(">>> SECTION: MARKET PRESSURE DEEPER DIVE (Sales since 2020)")
    recent_sales_df = df[df[sale_date_col].dt.year >= 2020].copy()
    recent_sales_df = recent_sales_df[recent_sales_df[price_col] > 1000]
    recent_sales_df['sale_to_assessed_ratio'] = recent_sales_df[price_col] / recent_sales_df[assessed_col]
    
    market_heat_data = []
    for name, group_df in [('Fineview', recent_sales_df[recent_sales_df[neighborhood_col] == 'Fineview']),
                           ('Perry South', recent_sales_df[recent_sales_df[neighborhood_col] == 'Perry South']),
                           ('Overall', recent_sales_df)]:
        total_sales = group_df.shape[0]
        if total_sales == 0: continue
        
        sold_above_assessed = group_df[group_df['sale_to_assessed_ratio'] > 1].shape[0]
        market_heat_data.append({
            '': name,
            'Median Sale-to-Assessed Ratio': f"{group_df['sale_to_assessed_ratio'].median():.2f}",
            '% of Sales Above Assessed': f"{(sold_above_assessed / total_sales) * 100:.1f}%"
        })
    market_heat_df = pd.DataFrame(market_heat_data).set_index('')
    print(market_heat_df.to_string())
    print("\n")

    print(">>> TABLE: TOP 10 PRIVATE OWNERS BY PROPERTY COUNT")
    # Filter out public and quasi-public entities
    private_owners_df = df[~df[owner_desc_col].str.contains('CITY OF|URA|HOUSING AUTH|URBAN REDEVELOPMENT', na=False)]
    top_owners = private_owners_df[owner_desc_col].value_counts().nlargest(10).reset_index()
    top_owners.columns = ['Owner', 'Property Count']
    
    delinquency_rates = []
    for owner in top_owners['Owner']:
        owner_df = private_owners_df[private_owners_df[owner_desc_col] == owner]
        rate = (owner_df[delinquent_tax_col] > 0).sum() / owner_df.shape[0] * 100 if owner_df.shape[0] > 0 else 0
        delinquency_rates.append(f"{rate:.1f}%")
    top_owners['Delinquency Rate of Portfolio'] = delinquency_rates
    print(top_owners.to_string(index=False))
    print("\n")
    
    # --- NEW ANALYSIS SECTIONS: GEOSPATIAL & STREET-LEVEL ---

    print(">>> TABLE: ANALYSIS BY STREET (Top 15 Streets by Property Count)")
    poor_conditions = ['POOR', 'VERY POOR', 'UNSOUND']
    
    # Aggregate data by street name
    street_groups = df.groupby('street_name')
    street_analysis = street_groups.agg(
        property_count=(owner_desc_col, 'size'),
        delinquency_rate=(delinquent_tax_col, lambda x: (x > 0).mean() * 100),
        lien_rate=(lien_count_col, lambda x: (x > 0).mean() * 100),
        poor_condition_rate=(condition_col, lambda x: x.isin(poor_conditions).mean() * 100),
        non_owner_occupied_rate=('ownership_category', lambda x: (x != 'Owner-Occupied').mean() * 100)
    ).round(1)

    # Filter for streets with a meaningful number of properties (e.g., more than 5)
    meaningful_streets = street_analysis[street_analysis['property_count'] > 5]
    top_streets = meaningful_streets.nlargest(15, 'property_count')
    print(top_streets.to_string())
    print("\n")

    print(">>> SECTION: GEOSPATIAL CLUSTERING - DISTRESS HOTSPOT STREETS")
    # Define a mask for highly distressed properties
    distressed_mask = (df[delinquent_tax_col] > 0) & \
                      (df[lien_count_col] > 0) & \
                      (df[condition_col].isin(poor_conditions))
    
    distressed_df = df[distressed_mask]
    total_distressed = distressed_df.shape[0]
    
    print(f"Total properties with high distress indicators (Delinquent, Lien, and Poor Condition): {total_distressed}")
    
    if total_distressed > 0:
        distress_ownership_dist = distressed_df['ownership_category'].value_counts(normalize=True) * 100
        print("\nDistribution of These Distressed Properties by Ownership Category:")
        for category, percentage in distress_ownership_dist.items():
            print(f"  - {category}: {percentage:.1f}%")

        print("\nTop 10 Streets with the Highest Count of Highly Distressed Properties:")
        hotspot_streets = distressed_df['street_name'].value_counts().nlargest(10).reset_index()
        hotspot_streets.columns = ['Street Name', 'Count of Highly Distressed Properties']
        print(hotspot_streets.to_string(index=False))

    print("\n--- End of Analysis ---")


# --- Run the main function ---
if __name__ == '__main__':
    # The script will look for 'inventory_appended.csv' in the same directory.
    # If the file is elsewhere, provide the full path as an argument.
    # Example: generate_report_analysis(r'C:\data\inventory_appended.csv')
    generate_report_analysis()


# In[165]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from scipy.stats import chi2_contingency

def perform_statistical_analysis_less_filtering(file_path='inventory_appended.csv'):
    """
    Loads property data and performs several statistical analyses.
    Each analysis uses the maximum data available by only filtering for
    the columns required for that specific test.

    Args:
        file_path (str): The path to the inventory CSV file.
    """
    print("--- Starting Statistical Analysis (Less Filtering Method) ---")
    
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # A) Define ownership category (consistent with other scripts)
    def get_ownership_category(row):
        if row['HOMESTEADFLAG'] == 'HOM': return 'Owner-Occupied'
        if isinstance(row['OWNERDESC'], str) and any(kw in row['OWNERDESC'].upper() for kw in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
            return 'Corporate Non-Owner-Occupied'
        return 'Other Non-Owner-Occupied'
    df['ownership_category'] = df.apply(get_ownership_category, axis=1)

    # B) Clean up key columns across the entire dataframe once
    df['CONDITIONDESC'] = df['CONDITIONDESC'].str.strip()
    
    # Create a simplified condition category
    def simplify_condition(condition):
        if condition in ['EXCELLENT', 'VERY GOOD', 'GOOD']: return 'Good'
        if condition in ['AVERAGE', 'FAIR']: return 'Average'
        if condition in ['POOR', 'VERY POOR', 'UNSOUND']: return 'Poor'
        return np.nan # Return NaN for unknown so it can be dropped easily
    df['condition_simple'] = df['CONDITIONDESC'].apply(simplify_condition)
    
    # Create a binary variable for delinquency
    df['is_delinquent'] = (df['Delinquent_Tax_Amount'] > 0).astype(int)

    # --- 2. Chi-Squared Test (Ownership vs. Condition) ---
    print(">>> Analysis 1: Chi-Squared Test for Association")
    print("Hypothesis: Is there a statistically significant association between Ownership Category and Property Condition?")
    
    # Use only the data needed for this test
    df_chi2 = df.dropna(subset=['ownership_category', 'condition_simple'])
    print(f"(Using {len(df_chi2)} observations for this test)")
    
    contingency_table = pd.crosstab(df_chi2['ownership_category'], df_chi2['condition_simple'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print("\nContingency Table (Observed Frequencies):")
    print(contingency_table)
    print(f"\nChi-Squared Statistic: {chi2:.2f}")
    print(f"P-value: {p:.4f}")
    
    if p < 0.05:
        print("Result: The p-value is less than 0.05. We reject the null hypothesis.")
        print("Conclusion: There IS a statistically significant association between ownership type and property condition.\n")
    else:
        print("Result: The p-value is greater than 0.05. We fail to reject the null hypothesis.")
        print("Conclusion: There is NO statistically significant association between ownership type and property condition.\n")

    # --- 3. ANOVA Test (Value vs. Ownership) ---
    print(">>> Analysis 2: ANOVA Test for Property Value")
    print("Hypothesis: Is there a statistically significant difference in the mean FAIRMARKETTOTAL across different Ownership Categories?")
    
    df_anova = df.dropna(subset=['FAIRMARKETTOTAL', 'ownership_category'])
    print(f"(Using {len(df_anova)} observations for this test)")
    
    model_anova = ols('FAIRMARKETTOTAL ~ C(ownership_category)', data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model_anova, typ=2)
    
    print("\nANOVA Results:")
    print(anova_table)
    
    p_anova = anova_table['PR(>F)'][0]
    if p_anova < 0.05:
        print("\nResult: The p-value is less than 0.05.")
        print("Conclusion: There IS a statistically significant difference in mean property values among the ownership groups.\n")
    else:
        print("\nResult: The p-value is greater than 0.05.")
        print("Conclusion: There is NO statistically significant difference in mean property values among the ownership groups.\n")

    # --- 4. Linear Regression (Predicting Property Value) ---
    print(">>> Analysis 3: Linear Regression to Predict FAIRMARKETTOTAL")
    print("Goal: Quantify the impact of various features on a property's assessed value.")
    
    lm_vars = ['FAIRMARKETTOTAL', 'FINISHEDLIVINGAREA', 'YEARBLT', 'LOTAREA', 'BEDROOMS', 'FULLBATHS', 'condition_simple', 'ownership_category', 'NEIGHBORHOOD']
    df_lm = df.dropna(subset=lm_vars)
    print(f"(Using {len(df_lm)} observations for this regression)")
    
    formula_lm = "FAIRMARKETTOTAL ~ FINISHEDLIVINGAREA + YEARBLT + LOTAREA + BEDROOMS + FULLBATHS + C(condition_simple, Treatment(reference='Good')) + C(ownership_category, Treatment(reference='Corporate Non-Owner-Occupied')) + C(NEIGHBORHOOD)"
    model_lm = ols(formula_lm, data=df_lm).fit()
    
    print("\nLinear Regression Summary:")
    print(model_lm.summary())
    print("\n")

    # --- 5. Logistic Regression (Predicting Delinquency) ---
    print(">>> Analysis 4: Logistic Regression to Predict Tax Delinquency")
    print("Goal: Identify which factors are significant predictors of a property being tax delinquent.")

    logit_vars = ['is_delinquent', 'FAIRMARKETTOTAL', 'YEARBLT', 'Lien_Count', 'condition_simple', 'ownership_category', 'NEIGHBORHOOD']
    df_logit = df.dropna(subset=logit_vars)
    print(f"(Using {len(df_logit)} observations for this regression)")

    formula_logit = "is_delinquent ~ FAIRMARKETTOTAL + YEARBLT + Lien_Count + C(condition_simple, Treatment(reference='Good')) + C(ownership_category, Treatment(reference='Corporate Non-Owner-Occupied')) + C(NEIGHBORHOOD)"
    model_logit = logit(formula_logit, data=df_logit).fit()

    print("\nLogistic Regression Summary:")
    print(model_logit.summary())
    
    print("\nInterpretation (Odds Ratios):")
    odds_ratios = pd.DataFrame(np.exp(model_logit.params), columns=['Odds Ratio'])
    print(odds_ratios)
    print("\n- Odds Ratios > 1 indicate an increase in the odds of delinquency for each one-unit increase in the variable.")
    print("- Odds Ratios < 1 indicate a decrease in the odds of delinquency.\n")

    print("--- End of Statistical Analysis ---")


# --- Run the main function ---
perform_statistical_analysis_less_filtering()


# In[167]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def generate_statistical_graphs(file_path='inventory_appended.csv'):
    """
    Loads property data and creates visualizations for key statistical tests.
    """
    print("--- Starting Statistical Graph Generation ---")
    
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # A) Define ownership category
    def get_ownership_category(row):
        if row['HOMESTEADFLAG'] == 'HOM': return 'Owner-Occupied'
        if isinstance(row['OWNERDESC'], str) and any(kw in row['OWNERDESC'].upper() for kw in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
            return 'Corporate Non-Owner-Occupied'
        return 'Other Non-Owner-Occupied'
    df['ownership_category'] = df.apply(get_ownership_category, axis=1)

    # B) Clean and prepare data
    df['CONDITIONDESC'] = df['CONDITIONDESC'].str.strip()
    
    def simplify_condition(condition):
        if condition in ['EXCELLENT', 'VERY GOOD', 'GOOD']: return 'Good'
        if condition in ['AVERAGE', 'FAIR']: return 'Average'
        if condition in ['POOR', 'VERY POOR', 'UNSOUND']: return 'Poor'
        return np.nan
    df['condition_simple'] = df['CONDITIONDESC'].apply(simplify_condition)
    df['is_delinquent'] = (df['Delinquent_Tax_Amount'] > 0).astype(int)

    # Graph for ANOVA Test (Box Plot) ---
    print(">>> Generating Graph 2: ANOVA Box Plot")
    df_anova = df.dropna(subset=['FAIRMARKETTOTAL', 'ownership_category'])
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='ownership_category', y='FAIRMARKETTOTAL', data=df_anova, palette='viridis')
    plt.title('Distribution of Property Value by Ownership Category', fontsize=16)
    plt.ylabel('Assessed Property Value (FAIRMARKETTOTAL)', fontsize=12)
    plt.xlabel('Ownership Category', fontsize=12)
    # Capping y-axis to exclude extreme outliers for better visualization
    plt.ylim(0, df_anova['FAIRMARKETTOTAL'].quantile(0.95)) 
    plt.tight_layout()
    try:
        plt.savefig('anova_boxplot.png', dpi=300)
        print("Saved anova_boxplot.png\n")
    except Exception as e:
        print(f"Error saving box plot: {e}\n")
    plt.show()

    # Graph for Linear Regression (Residual Plot) ---
    print(">>> Generating Graph 3: Linear Regression Residuals")
    lm_vars = ['FAIRMARKETTOTAL', 'FINISHEDLIVINGAREA', 'YEARBLT', 'LOTAREA', 'BEDROOMS', 'FULLBATHS', 'condition_simple', 'ownership_category', 'NEIGHBORHOOD']
    df_lm = df.dropna(subset=lm_vars)
    formula_lm = "FAIRMARKETTOTAL ~ FINISHEDLIVINGAREA + YEARBLT + LOTAREA + BEDROOMS + FULLBATHS + C(condition_simple, Treatment(reference='Good')) + C(ownership_category, Treatment(reference='Corporate Non-Owner-Occupied')) + C(NEIGHBORHOOD)"
    model_lm = ols(formula_lm, data=df_lm).fit()
    
    # Create residual plot
    plt.figure(figsize=(10, 7))
    sns.residplot(x=model_lm.fittedvalues, y=model_lm.resid, lowess=True, line_kws={'color': 'red', 'lw': 2, 'alpha': 0.8})
    plt.title('Residuals vs. Fitted Values Plot', fontsize=16)
    plt.xlabel('Fitted Values (Predicted Property Value)', fontsize=12)
    plt.ylabel('Residuals (Prediction Errors)', fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig('linear_regression_residuals.png', dpi=300)
        print("Saved linear_regression_residuals.png\n")
    except Exception as e:
        print(f"Error saving residual plot: {e}\n")
    plt.show()

    # Graph for Logistic Regression (Odds Ratios) ---
    print(">>> Generating Graph 4: Logistic Regression Odds Ratios")
    logit_vars = ['is_delinquent', 'FAIRMARKETTOTAL', 'YEARBLT', 'Lien_Count', 'condition_simple', 'ownership_category', 'NEIGHBORHOOD']
    df_logit = df.dropna(subset=logit_vars)
    formula_logit = "is_delinquent ~ FAIRMARKETTOTAL + YEARBLT + Lien_Count + C(condition_simple, Treatment(reference='Good')) + C(ownership_category, Treatment(reference='Corporate Non-Owner-Occupied')) + C(NEIGHBORHOOD)"
    model_logit = logit(formula_logit, data=df_logit).fit(disp=0)
    
    # Get odds ratios and confidence intervals
    odds_ratios = pd.DataFrame({
        "Odds Ratio": np.exp(model_logit.params),
        "Lower CI": np.exp(model_logit.conf_int()[0]),
        "Upper CI": np.exp(model_logit.conf_int()[1]),
    })
    odds_ratios = odds_ratios.drop('Intercept') # Drop intercept for cleaner plot

    print("--- End of Graph Generation ---")

# --- Run the main function ---
generate_statistical_graphs()


# In[168]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def perform_advanced_analysis(file_path='inventory_appended.csv'):
    """
    Performs three advanced analyses on the property inventory data:
    1. Identifies and analyzes property flips.
    2. Compares distress metrics by length of ownership.
    3. Builds a model to identify properties at high risk of future distress and
       adds the risk score back to the original file.
    """
    print("--- Starting Advanced Property Analysis ---")
    
    # --- Data Loading and Universal Preparation ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Convert date columns to datetime objects, coercing errors
    df['SALEDATE'] = pd.to_datetime(df['SALEDATE'], errors='coerce')
    df['PREVSALEDATE'] = pd.to_datetime(df['PREVSALEDATE'], errors='coerce')
    df['recent_sale_date'] = pd.to_datetime(df['recent_sale_date'], errors='coerce')


    # Clean numeric columns
    numeric_cols = ['SALEPRICE', 'PREVSALEPRICE', 'Delinquent_Tax_Amount', 'recent_sale_price']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create binary flags for distress
    df['is_delinquent'] = (df['Delinquent_Tax_Amount'] > 0).astype(int)
    poor_conditions = ['POOR', 'VERY POOR', 'UNSOUND']
    df['is_poor_condition'] = df['CONDITIONDESC'].str.strip().isin(poor_conditions)

    # Define the ownership category
    def get_ownership_category(row):
        if row['HOMESTEADFLAG'] == 'HOM': return 'Owner-Occupied'
        if isinstance(row['OWNERDESC'], str) and any(kw in row['OWNERDESC'].upper() for kw in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
            return 'Corporate Non-Owner-Occupied'
        return 'Other Non-Owner-Occupied'
    df['ownership_category'] = df.apply(get_ownership_category, axis=1)


    # --- Analysis 1: Property "Flipping" Analysis ---
    print(">>> Analysis 1: Property Flipping (Resold within 24 Months)")
    
    # Create a dataframe specifically for flip analysis
    df_flips = df.dropna(subset=['SALEDATE', 'PREVSALEDATE', 'SALEPRICE', 'PREVSALEPRICE']).copy()
    
    # Calculate time difference in days
    df_flips['days_between_sales'] = (df_flips['SALEDATE'] - df_flips['PREVSALEDATE']).dt.days
    
    # Filter for properties sold within 24 months (approx. 730 days)
    potential_flips = df_flips[
        (df_flips['days_between_sales'] > 0) & 
        (df_flips['days_between_sales'] <= 730) &
        (df_flips['SALEPRICE'] > df_flips['PREVSALEPRICE']) &
        (df_flips['PREVSALEPRICE'] > 1000) # Exclude sales from $1, etc.
    ].copy()

    if not potential_flips.empty:
        potential_flips['profit'] = potential_flips['SALEPRICE'] - potential_flips['PREVSALEPRICE']
        potential_flips['profit_margin_pct'] = (potential_flips['profit'] / potential_flips['PREVSALEPRICE']) * 100
        
        avg_profit = potential_flips['profit'].mean()
        avg_profit_margin = potential_flips['profit_margin_pct'].mean()
        avg_hold_time = potential_flips['days_between_sales'].mean()

        print(f"Identified {len(potential_flips)} potential flips (sold within 24 months for a profit).")
        print(f"Average Holding Time: {avg_hold_time:.0f} days")
        print(f"Average Profit: ${avg_profit:,.2f}")
        print(f"Average Profit Margin: {avg_profit_margin:.2f}%")
        
        print("\nTop 10 Most Profitable Flips:")
        top_flips_table = potential_flips.nlargest(10, 'profit')[['PROPERTYADDRESS', 'PREVSALEDATE', 'SALEDATE', 'PREVSALEPRICE', 'SALEPRICE', 'profit']]
        print(top_flips_table.to_string(index=False))
    else:
        print("No potential property flips found within the 24-month timeframe.")
    
    print("\n" + "="*80 + "\n")


    # --- Analysis 2: Long-Term vs. Short-Term Ownership ---
    print(">>> Analysis 2: Distress by Ownership Duration")
    
    # Use a fixed current date for consistent analysis
    current_date = pd.to_datetime('2025-06-20')
    df_duration = df.dropna(subset=['recent_sale_date']).copy()
    
    df_duration['ownership_years'] = (current_date - df_duration['recent_sale_date']).dt.days / 365.25
    
    # Create ownership duration bins
    bins = [0, 2, 5, 10, np.inf]
    labels = ['0-2 Years (New)', '2-5 Years', '5-10 Years', '10+ Years (Long-Term)']
    df_duration['ownership_duration_group'] = pd.cut(df_duration['ownership_years'], bins=bins, labels=labels, right=False)
    
    # Group by the duration and calculate distress metrics
    duration_analysis = df_duration.groupby('ownership_duration_group', observed=True).agg(
        property_count=('OWNERDESC', 'size'),
        delinquency_rate=('is_delinquent', 'mean'),
        poor_condition_rate=('is_poor_condition', 'mean')
    ).reset_index()

    # Format for printing
    duration_analysis['delinquency_rate'] = (duration_analysis['delinquency_rate'] * 100).map('{:.1f}%'.format)
    duration_analysis['poor_condition_rate'] = (duration_analysis['poor_condition_rate'] * 100).map('{:.1f}%'.format)

    print(duration_analysis.to_string(index=False))
    print("\n" + "="*80 + "\n")


    # --- Analysis 3: Modeling "At-Risk" Properties ---
    print(">>> Analysis 3: Identifying 'At-Risk' Properties")
    print("Building a model to find properties not currently delinquent but with a high predicted risk.")

    # A) Prepare data for the model
    # Include 'pin' and 'PROPERTYHOUSENUM' for final output
    model_features = ['FAIRMARKETTOTAL', 'YEARBLT', 'Lien_Count', 'CONDITIONDESC', 'ownership_category', 'NEIGHBORHOOD']
    df_model = df.dropna(subset=model_features + ['is_delinquent', 'pin', 'PROPERTYHOUSENUM', 'PROPERTYADDRESS']).copy()
    df_model['Lien_Count'] = df_model['Lien_Count'].astype(int)

    # B) Define features (X) and target (y)
    X = df_model[model_features]
    y = df_model['is_delinquent']

    # C) Create a preprocessing pipeline for categorical and numerical data
    categorical_features = ['CONDITIONDESC', 'ownership_category', 'NEIGHBORHOOD']
    numeric_features = ['FAIRMARKETTOTAL', 'YEARBLT', 'Lien_Count']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # D) Create and train the logistic regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])
    
    model.fit(X, y)

    # E) Apply the model to predict delinquency probability for ALL properties in the model dataset
    df_model['risk_probability'] = model.predict_proba(X)[:, 1]

    # F) Identify "at-risk" properties
    at_risk_properties = df_model[df_model['is_delinquent'] == 0].copy()
    
    print(f"\nTop 15 Properties at Highest Risk of Becoming Delinquent:")
    # **CHANGE**: Added 'PROPERTYHOUSENUM' to the output table
    at_risk_table = at_risk_properties.nlargest(15, 'risk_probability')[
        ['pin', 'PROPERTYHOUSENUM', 'PROPERTYADDRESS', 'OWNERDESC', 'FAIRMARKETTOTAL', 'CONDITIONDESC', 'risk_probability']
    ]
    at_risk_table['risk_probability'] = (at_risk_table['risk_probability'] * 100).map('{:.1f}%'.format)
    print(at_risk_table.to_string(index=False))

    # G) **CHANGE**: Add a new column to the original CSV file with risk scores
    try:
        # Isolate the pin and risk score from the model's dataframe
        risk_score_output = df_model[['pin', 'risk_probability']].copy()
        risk_score_output.rename(columns={'risk_probability': 'delinquency_risk_score'}, inplace=True)
        
        # Load the original dataframe again
        original_df = pd.read_csv(file_path, low_memory=False)

        # Merge the risk scores back into the original dataframe
        # First, drop the column if it already exists to avoid duplicates
        if 'delinquency_risk_score' in original_df.columns:
            original_df.drop(columns=['delinquency_risk_score'], inplace=True)
        
        # Merge on 'pin'
        updated_df = pd.merge(original_df, risk_score_output, on='pin', how='left')

        # Save the updated dataframe, overwriting the original file
        updated_df.to_csv(file_path, index=False)
        print(f"\nSuccessfully added 'delinquency_risk_score' column to '{file_path}'")
    except Exception as e:
        print(f"\nCould not update the original CSV file with risk scores. Error: {e}")


    print("\n--- End of Advanced Analysis ---")

# --- Run the main function ---
perform_advanced_analysis()


# In[169]:


import pandas as pd
import numpy as np

def analyze_owner_performance(file_path='inventory_appended.csv', min_properties=5):
    """
    Analyzes property owner performance using the 'Verified_Owner' and 
    'Verified_Mailing_Address' columns for more accurate analysis.

    Args:
        file_path (str): The path to the inventory CSV file, which should 
                         already contain the verified owner columns.
        min_properties (int): The minimum number of properties an owner must have
                              to be included in the 'best' and 'worst' analysis.
    """
    print("--- Starting Owner Performance Analysis using Verified Data ---")
    
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Check for required columns, now including the verified ones
    required_cols = ['PROPERTYADDRESS', 'Verified_Mailing_Address', 'Verified_Owner', 'Delinquent_Tax_Amount', 'Lien_Count', 'CONDITIONDESC', 'USEDESC']
    if not all(col in df.columns for col in required_cols):
        print("Error: The CSV file is missing one or more required columns.")
        print("Please ensure the file has been merged with the mailing label data first.")
        print(f"Required: {required_cols}")
        return

    # A) Data Cleaning and Feature Engineering
    df['is_delinquent'] = (df['Delinquent_Tax_Amount'] > 0).astype(int)
    df['has_liens'] = (df['Lien_Count'] > 0).astype(int)
    
    poor_conditions = ['POOR', 'VERY POOR', 'UNSOUND']
    df['is_poor_condition'] = df['CONDITIONDESC'].str.strip().isin(poor_conditions).astype(int)
    
    # B) Create Improved Owner-Occupancy Metric using Verified Address
    df['prop_address_norm'] = df['PROPERTYADDRESS'].str.strip().str.upper()
    # Use the verified mailing address for the comparison
    df['owner_address_norm'] = df['Verified_Mailing_Address'].str.strip().str.upper()
    
    def check_address_contains(row):
        prop_addr = row['prop_address_norm']
        owner_addr = row['owner_address_norm']
        if pd.notna(prop_addr) and pd.notna(owner_addr):
            # Check if the property address is contained within the owner's mailing address
            return prop_addr in owner_addr
        return False
    df['is_owner_occupied'] = df.apply(check_address_contains, axis=1).astype(int)

    print(f"Identified {df['is_owner_occupied'].sum()} owner-occupied properties based on verified address matching.\n")

    # --- 2. Largest Private Owners Analysis ---
    print(">>> TABLE: Top 15 Largest Private Owners by Property Count (Verified)")
    
    # Filter out public entities based on OWNERDESC (original field is still useful here)
    private_owners_df = df[~df['OWNERDESC'].str.contains('CITY OF|URA|HOUSING AUTH|URBAN REDEVELOPMENT', na=False, case=False)].copy()
    
    # Drop rows where the verified owner name is blank
    private_owners_df.dropna(subset=['Verified_Owner'], inplace=True)
    
    # Group by the verified owner name to get property counts
    owner_portfolio_counts = private_owners_df['Verified_Owner'].value_counts().reset_index()
    owner_portfolio_counts.columns = ['Verified Owner', 'Property Count']
    
    print(owner_portfolio_counts.nlargest(15, 'Property Count').to_string(index=False))
    print("\n" + "="*80 + "\n")

    # --- 3. Largest Owners of Vacant Land ---
    print(">>> TABLE: Top 15 Largest Owners of Vacant Land by Lot Count (Verified)")
    
    # Filter for vacant land
    vacant_df = private_owners_df[private_owners_df['USEDESC'].str.contains('VACANT', na=False, case=False)].copy()
    
    vacant_owner_counts = vacant_df['Verified_Owner'].value_counts().reset_index()
    vacant_owner_counts.columns = ['Verified Owner', 'Vacant Lot Count']

    print(vacant_owner_counts.nlargest(15, 'Vacant Lot Count').to_string(index=False))
    print("\n" + "="*80 + "\n")

    # --- 4. Best and Worst Owner Analysis ---
    # **CHANGE**: Group by both owner name and mailing address for more precise analysis
    owner_metrics = private_owners_df.groupby(['Verified_Owner', 'Verified_Mailing_Address']).agg(
        property_count=('Verified_Owner', 'size'),
        delinquency_rate=('is_delinquent', 'mean'),
        lien_rate=('has_liens', 'mean'),
        poor_condition_rate=('is_poor_condition', 'mean')
    )

    # Filter for owners with a significant number of properties
    significant_owners = owner_metrics[owner_metrics['property_count'] >= min_properties].copy()
    
    # Format percentages for readability
    for col in ['delinquency_rate', 'lien_rate', 'poor_condition_rate']:
        significant_owners[col] = (significant_owners[col] * 100).round(1)

    # Rename columns for clarity in the final tables
    significant_owners.rename(columns={
        'delinquency_rate': '% Delinquent',
        'lien_rate': '% With Liens',
        'poor_condition_rate': '% In Poor Condition'
    }, inplace=True)
    significant_owners.reset_index(inplace=True)
    # **CHANGE**: Rename columns to reflect the new grouping
    significant_owners.rename(columns={'Verified_Owner': 'Owner Name', 'Verified_Mailing_Address': 'Mailing Address'}, inplace=True)

    # A) "Best" Owners (Lowest rates of negative metrics)
    print(f">>> TABLE: 'Best' Owners (Portfolios of {min_properties}+ properties, Verified)")
    print("--- (Sorted by lowest delinquency, lien, and poor condition rates) ---\n")
    
    best_owners = significant_owners.sort_values(
        by=['% Delinquent', '% With Liens', '% In Poor Condition', 'property_count'],
        ascending=[True, True, True, False]
    )
    # **CHANGE**: Include Mailing Address in the output table
    print(best_owners.head(10)[['Owner Name', 'Mailing Address', 'property_count', '% Delinquent', '% With Liens', '% In Poor Condition']].to_string(index=False))
    print("\n" + "="*80 + "\n")

    # B) "Worst" Owners (Highest rates of negative metrics)
    print(f">>> TABLE: 'Worst' Owners (Portfolios of {min_properties}+ properties, Verified)")
    print("--- (Sorted by highest delinquency, lien, and poor condition rates) ---\n")
    
    worst_owners = significant_owners.sort_values(
        by=['% Delinquent', '% With Liens', '% In Poor Condition', 'property_count'],
        ascending=[False, False, False, False]
    )
    # **CHANGE**: Include Mailing Address in the output table
    print(worst_owners.head(10)[['Owner Name', 'Mailing Address', 'property_count', '% Delinquent', '% With Liens', '% In Poor Condition']].to_string(index=False))
    print("\n--- End of Analysis ---")

# --- Run the main function ---
if __name__ == '__main__':
    analyze_owner_performance()


# In[171]:


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def analyze_owner_location_performance(file_path='inventory_appended.csv', local_zips=['15214', '15233']):
    """
    Performs a robust statistical analysis to determine if local owners
    have better-performing property portfolios than out-of-town owners.

    Args:
        file_path (str): The path to the inventory CSV file with verified owner data.
        local_zips (list): A list of ZIP codes to be considered "local".
    """
    print("--- Starting Analysis: Owner Location vs. Portfolio Performance ---")
    
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Check for required columns
    required_cols = ['Verified_Mailing_Address', 'Verified_Owner', 'Delinquent_Tax_Amount', 'Lien_Count', 'CONDITIONDESC']
    if not all(col in df.columns for col in required_cols):
        print("Error: The CSV file is missing one or more required columns.")
        return

    # A) Filter for private owners with verified mailing addresses
    df_analysis = df[~df['OWNERDESC'].str.contains('CITY OF|URA|HOUSING AUTH|URBAN REDEVELOPMENT', na=False, case=False)].copy()
    df_analysis.dropna(subset=['Verified_Mailing_Address', 'Verified_Owner'], inplace=True)

    # B) Create "is_local_owner" flag
    def check_if_local(address):
        if not isinstance(address, str):
            return False
        # Check if any of the local ZIP codes are in the address string
        return any(zip_code in address for zip_code in local_zips)
        
    df_analysis['is_local_owner'] = df_analysis['Verified_Mailing_Address'].apply(check_if_local)

    # C) Create a combined "is_distressed" flag
    # A property is considered distressed if it has any of the three negative indicators
    poor_conditions = ['POOR', 'VERY POOR', 'UNSOUND']
    df_analysis['is_delinquent'] = (df_analysis['Delinquent_Tax_Amount'] > 0)
    df_analysis['has_liens'] = (df_analysis['Lien_Count'] > 0)
    df_analysis['is_poor_condition'] = df_analysis['CONDITIONDESC'].str.strip().isin(poor_conditions)
    
    df_analysis['is_distressed'] = (
        df_analysis['is_delinquent'] | 
        df_analysis['has_liens'] | 
        df_analysis['is_poor_condition']
    )

    print(f"Total private properties with verified owners being analyzed: {len(df_analysis)}")
    print(f"Number of properties owned by local owners: {df_analysis['is_local_owner'].sum()}")
    print(f"Number of properties owned by out-of-town owners: {len(df_analysis) - df_analysis['is_local_owner'].sum()}\n")

    # --- 2. Statistical Test (Chi-Squared) ---
    print(">>> Analysis: Is there a statistically significant association between owner location and property distress?")
    
    # Create a contingency table
    contingency_table = pd.crosstab(df_analysis['is_local_owner'], df_analysis['is_distressed'])
    contingency_table.index = ['Out-of-Town Owner', 'Local Owner']
    contingency_table.columns = ['Not Distressed', 'Distressed']
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print("\nContingency Table (Observed Property Counts):")
    print(contingency_table)
    print(f"\nChi-Squared Statistic: {chi2:.2f}")
    print(f"P-value: {p:.4f}")
    
    if p < 0.05:
        print("\nConclusion: The p-value is less than 0.05, which means the result is statistically significant.")
        print("There IS a real association between an owner being local and the condition of their properties.\n")
    else:
        print("\nConclusion: The p-value is greater than 0.05, which means the result is not statistically significant.")
        print("There is NO clear association between an owner being local and the condition of their properties.\n")

    # --- 3. Compare Distress Rates ---
    print(">>> Comparison: Average Distress Rates by Owner Location")
    
    # Group by owner location and calculate the mean of the distress flags
    distress_rates = df_analysis.groupby('is_local_owner').agg(
        total_properties=('is_distressed', 'size'),
        avg_delinquency_rate=('is_delinquent', 'mean'),
        avg_lien_rate=('has_liens', 'mean'),
        avg_poor_condition_rate=('is_poor_condition', 'mean'),
        overall_distress_rate=('is_distressed', 'mean') # % of properties with at least one issue
    )
    distress_rates.index = ['Out-of-Town Owner', 'Local Owner']
    
    # Format for readability
    for col in distress_rates.columns:
        if 'rate' in col:
            distress_rates[col] = (distress_rates[col] * 100).map('{:.1f}%'.format)

    print(distress_rates.to_string())
    print("\n--- End of Analysis ---")

# --- Run the main function ---
if __name__ == '__main__':
    analyze_owner_location_performance()


# In[175]:


import pandas as pd
import numpy as np

def analyze_demolition_risk(file_path='inventory_appended.csv'):
    """
    Analyzes properties from the inventory to identify and categorize those
    at risk of demolition based on condition and condemnation status.

    Args:
        file_path (str): The path to the inventory CSV file.
    """
    print("--- Starting Demolition Risk Analysis ---")
    
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded '{file_path}'.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Check for required columns
    required_cols = ['pin', 'PROPERTYADDRESS', 'CONDITIONDESC', 'Condemned_Property_Type', 'NEIGHBORHOOD', 'OWNERDESC', 'HOMESTEADFLAG']
    if not all(col in df.columns for col in required_cols):
        print("Error: The CSV file is missing one or more required columns.")
        print(f"Required: {required_cols}")
        return

    # --- 2. Define Risk Categories ---
    def assess_demolition_risk(row):
        """Categorizes a property's demolition risk based on its attributes."""
        condition = str(row['CONDITIONDESC']).strip().upper()
        condemned_status = row['Condemned_Property_Type']

        if pd.notna(condemned_status) or condition == 'UNSOUND':
            return 'Critical Risk'
        elif condition == 'VERY POOR':
            return 'High Risk'
        elif condition == 'POOR':
            return 'Medium Risk'
        else:
            return 'No Apparent Risk'

    df['demolition_risk'] = df.apply(assess_demolition_risk, axis=1)

    # Create ownership category for breakdown analysis
    def get_ownership_category(row):
        if row['HOMESTEADFLAG'] == 'HOM': return 'Owner-Occupied'
        if isinstance(row['OWNERDESC'], str) and any(kw in row['OWNERDESC'].upper() for kw in ['LLC', 'INC', 'CORP', 'LP', 'TRUST']):
            return 'Corporate Non-Owner-Occupied'
        return 'Other Non-Owner-Occupied'
    df['ownership_category'] = df.apply(get_ownership_category, axis=1)

    # --- 3. Generate Summary Tables ---
    
    # A) Overall Risk Summary
    print(">>> TABLE: Overall Demolition Risk Summary")
    risk_summary = df['demolition_risk'].value_counts().reset_index()
    risk_summary.columns = ['Risk Category', 'Property Count']
    # Define a logical order for the categories
    category_order = ['Critical Risk', 'High Risk', 'Medium Risk', 'No Apparent Risk']
    risk_summary['Risk Category'] = pd.Categorical(risk_summary['Risk Category'], categories=category_order, ordered=True)
    print(risk_summary.sort_values('Risk Category').to_string(index=False))
    print("\n" + "="*80 + "\n")

    # B) Risk Breakdown by Neighborhood
    print(">>> TABLE: Demolition Risk by Neighborhood")
    risk_by_neighborhood = pd.crosstab(df['NEIGHBORHOOD'], df['demolition_risk'])
    # Reorder columns for logical presentation
    risk_by_neighborhood = risk_by_neighborhood[category_order]
    print(risk_by_neighborhood.to_string())
    print("\n" + "="*80 + "\n")

    # C) Risk Breakdown by Ownership Category
    print(">>> TABLE: Demolition Risk by Ownership Category")
    risk_by_ownership = pd.crosstab(df['ownership_category'], df['demolition_risk'])
    # Reorder columns for logical presentation
    risk_by_ownership = risk_by_ownership[category_order]
    print(risk_by_ownership.to_string())
    print("\n" + "="*80 + "\n")

    # D) List of Top "Critical Risk" Properties
    print(">>> TABLE: Top 15 Properties at 'Critical Risk'")
    critical_risk_properties = df[df['demolition_risk'] == 'Critical Risk'].copy()
    
    if not critical_risk_properties.empty:
        output_cols = ['pin', 'PROPERTYADDRESS', 'NEIGHBORHOOD', 'OWNERDESC', 'CONDITIONDESC', 'Condemned_Property_Type']
        print(critical_risk_properties.head(15)[output_cols].to_string(index=False))
    else:
        print("No properties were identified to be at 'Critical Risk'.")

    print("\n--- End of Analysis ---")

# --- Run the main function ---
if __name__ == '__main__':
    analyze_demolition_risk()


# In[ ]:




