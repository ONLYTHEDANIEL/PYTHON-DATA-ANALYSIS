# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:04:23 2026

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style 
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset 
df = pd.read_csv("C:/Users/User/Documents/PYTHON AND STOCKS/archive/accident_data.csv", low_memory=False)

# Inspection
print(df.info())
print(df.head())

# 2. DATA CLEANING & PREPARATION
# Handle missing rows for Road Surface Conditions (~742 missing entries)
df['Road_Surface_Conditions'] = df['Road_Surface_Conditions'].fillna('Unknown')

# Drop rows with critical structural anomalies in geospatial data
df = df.dropna(subset=['Latitude', 'Longitude'])

# Parse strings into true DateTime objects to unlock temporal indexing
df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')
df['Year'] = df['Accident Date'].dt.year
df['Month'] = df['Accident Date'].dt.month
df['Day_of_Week'] = df['Accident Date'].dt.day_name()

# ANALYSIS 1: TEMPORAL TRENDS OVER TIME (2019 - 2025)
# =====================================================================
print("--- Generating Temporal Trends ---")
yearly_trends = df.groupby('Year')['Number_of_Casualties'].agg(['count', 'sum']).rename(
    columns={'count': 'Total_Accidents', 'sum': 'Total_Casualties'}
)
print(yearly_trends)

# Plotting Macro Trends
fig, ax1 = plt.subplots()
color = '#1f77b4'
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Accidents', color=color)
ax1.plot(yearly_trends.index, yearly_trends['Total_Accidents'], color=color, marker='o', linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Total Casualties', color=color)
ax2.plot(yearly_trends.index, yearly_trends['Total_Casualties'], color=color, marker='s', linestyle='--', linewidth=2.5)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('UK Road Accidents & Casualties Macro Trends (2019 - 2025)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ANALYSIS 2: ENVIRONMENTAL EFFECTS ON SEVERITY
# =====================================================================
print("\n--- Generating Environmental Cross-Tabulations ---")
# Build a normalized cross-tabulation table to inspect proportional risk changes
road_severity_pct = pd.crosstab(df['Road_Surface_Conditions'], df['Accident_Severity'], normalize='index') * 100
print(road_severity_pct)

# Plotting Stacked Percentage Bar Chart
road_severity_pct.plot(kind='bar', stacked=True, color=['#e74c3c', '#f39c12', '#3498db'])
plt.title('Proportional Distribution of Accident Severity by Road Surface', fontsize=14, fontweight='bold')
plt.xlabel('Road Surface Condition')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Severity')
plt.tight_layout()
plt.show()

# ANALYSIS 3: GEOGRAPHIC AND DISTRICT PROFILE (SEVERITY INDEX)
# =====================================================================
print("\n--- Calculating High-Risk District Profiles ---")

# Step 1: Map severity labels to numerical penalty weights 
severity_weights = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
df['Severity_Weight'] = df['Accident_Severity'].map(severity_weights).fillna(1)

# Step 2: Aggregate values by district area
district_stats = df.groupby('District Area').agg(
    Total_Accidents=('Accident_Severity', 'count'),
    Total_Casualties=('Number_of_Casualties', 'sum'),
    Weighted_Severity_Sum=('Severity_Weight', 'sum')
)
# Step 3: Equation execution for structural Severity Score normalization
district_stats['Severity_Score'] = district_stats['Weighted_Severity_Sum'] / district_stats['Total_Accidents']

# Filter out extreme low-sample outliers (districts with fewer than 50 total accidents over 6 years)
high_volume_districts = district_stats[district_stats['Total_Accidents'] >= 50]

# Display the Top 10 most critical high-risk districts by weighted risk profile
top_10_dangerous = high_volume_districts.sort_values(by='Severity_Score', ascending=False).head(10)
print(top_10_dangerous[['Total_Accidents', 'Total_Casualties', 'Severity_Score']])



