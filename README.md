

# Overview
This project creates a comprehensive sales analytics dashboard using Power BI with Python integration for advanced analytics. The dashboard provides insights into sales performance, profitability, customer behavior, and geographic trends.

# Prerequisites

- Power BI Desktop (free version)
- Python 3.7+ with the following packages:
  
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
  ```

## Data Preparation
Python Cleaning Script
Create a data_preparation.py file:

```python

import pandas as pd

# Load and clean data
df = pd.read_csv('sales-data-sample.csv')
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df['ProfitMargin'] = df['Profit'] / df['Sales']

# Customer segmentation
def segment_customer(row):
    if row['Sales'] > 5000: return 'VIP'
    elif row['Sales'] > 2000: return 'Premium'
    else: return 'Regular'
    
df['CustomerSegment'] = df.apply(segment_customer, axis=1)
df.to_csv('processed_sales.csv', index=False)

```
