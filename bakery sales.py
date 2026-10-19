# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:05:51 2026

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure clean plotting styles and scannable visual configurations
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14
})

# =====================================================================
# 1. LOAD DATA & DATA CLEANING
# =====================================================================
# Replace 'bakery_sales_data.csv' with your actual path
df = pd.read_csv("C:/Users/User/Documents/PYTHON AND STOCKS/archive (1)/bakery_sales_dataset.csv")

# Ensure proper data types
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Clean missing or conditional values safely
df['Festival'] = df['Festival'].fillna('Regular Day')
df['Promotion_Type'] = df['Promotion_Type'].fillna('No Promotion')

# Recalculate metrics if discrepancies exist from synthetic rounding
df['Calculated_Revenue'] = df['Quantity'] * df['Selling_Price']
df['Calculated_Cost'] = df['Quantity'] * df['Cost_Price']
df['Calculated_Profit'] = df['Calculated_Revenue'] - df['Calculated_Cost']

# =====================================================================
# ANALYSIS 1: PRODUCTION EFFICIENCY & SPOILAGE RISK
# =====================================================================
print("--- Evaluating Spoilage Risk & Waste Valuation ---")

# Calculate direct financial loss of inventory that spoiled or went unsold
df['Financial_Waste_Loss'] = df['Unsold_Units'] * df['Cost_Price']

# Group by category & perishability to find production forecasting flaws
waste_profile = df.groupby(['Category', 'Perishability']).agg(
    Avg_Waste_Pct=('Waste_Pct', 'mean'),
    Total_Waste_Loss=('Financial_Waste_Loss', 'sum'),
    Total_Unsold_Units=('Unsold_Units', 'sum'),
    Total_Units_Produced=('Units_Produced', 'sum')
).reset_index()

# Calculate structural overproduction ratio
waste_profile['Overproduction_Ratio'] = (waste_profile['Total_Unsold_Units'] / waste_profile['Total_Units_Produced']) * 100
print(waste_profile.sort_values(by='Total_Waste_Loss', ascending=False))

# Plotting: Spoilage Hotspots Across Product Tiers
sns.barplot(data=waste_profile, x='Category', y='Avg_Waste_Pct', hue='Perishability', palette='autumn')
plt.title('Average Waste Percentage by Product Category and Perishability Level', fontweight='bold')
plt.xticks(rotation=15)
plt.ylabel('Average Waste (%)')
plt.tight_layout()
plt.show()

# =====================================================================
# ANALYSIS 2: CUSTOMER SEGMENTATION & MARGIN DILUTION
# =====================================================================
print("\n--- Evaluating Loyalty Value vs Promotion Margin Dilution ---")

# Investigate if loyalty and promotions are giving away margins unnecessarily
loyalty_promo_mix = df.groupby(['Loyalty_Member', 'Promotion_Applied']).agg(
    Avg_Discount_Pct=('Discount_Percentage', 'mean'),
    Avg_Ticket_Size=('Total_Bill', 'mean'),
    Avg_Customer_Rating=('Customer_Rating', 'mean'),
    Total_Net_Profit=('Profit', 'sum'),
    Transaction_Count=('Order_ID', 'count')
).reset_index()

print(loyalty_promo_mix)

# Plotting: Ticket Size vs Loyalty Ratings
sns.lmplot(data=df, x='Customer_Age', y='Total_Bill', hue='Loyalty_Member', col='Customer_Segment', 
           scatter_kws={'alpha':0.3}, line_kws={'linewidth':2}, height=4, aspect=1)
plt.subplots_adjust(top=0.85)
plt.suptitle('Customer Age vs. Total Bill Across Segments and Loyalty Membership', fontweight='bold')
plt.show()

# =====================================================================
# ANALYSIS 3: EXTERNAL ENVIRONMENT DEMAND PATTERNS (WEATHER & TIMES)
# =====================================================================
print("\n--- Constructing Environmental Demand Heatmap ---")

# Group by hour and weather conditions to map bakery peak operations
weather_time_demand = df.pivot_table(
    index='Hour', 
    columns='Weather', 
    values='Quantity', 
    aggfunc='sum'
).fillna(0)

# Generate an operational rush-hour heatmap
sns.heatmap(weather_time_demand, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Total Units Sold'})
plt.title('Bakery Demand Heatmap: Hour of Day vs Weather Conditions', fontweight='bold')
plt.xlabel('Weather Condition')
plt.ylabel('Hour of Day (24h)')
plt.tight_layout()
plt.show()
