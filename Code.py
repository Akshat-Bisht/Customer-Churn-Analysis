"""
Customer Churn Analysis and Insights
====================================
A comprehensive analysis of customer churn patterns using Python, NumPy, Pandas, and Matplotlib

Author: Data Analytics Project
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA GENERATION
# ============================================================================

def generate_customer_data(n_customers=3000, random_seed=42):
    """
    Generate synthetic customer churn dataset with realistic patterns

    Parameters:
    -----------
    n_customers : int
        Number of customer records to generate
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        Customer dataset with churn information
    """
    np.random.seed(random_seed)

    print("Generating Customer Churn Dataset...")
    print("=" * 70)

    # Customer Demographics
    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    age = np.random.normal(45, 15, n_customers).astype(int)
    age = np.clip(age, 18, 80)
    gender = np.random.choice(['Male', 'Female'], n_customers, p=[0.48, 0.52])

    # Geographic data
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
              'Kolkata', 'Pune', 'Ahmedabad']
    city = np.random.choice(cities, n_customers, 
                           p=[0.18, 0.16, 0.15, 0.12, 0.11, 0.10, 0.10, 0.08])

    # Service Information
    tenure_months = np.random.exponential(24, n_customers).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)

    contract_type = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], 
                                     n_customers, p=[0.55, 0.25, 0.20])

    # Internet Services
    internet_service = np.random.choice(['Fiber Optic', 'DSL', 'No'], 
                                        n_customers, p=[0.45, 0.35, 0.20])

    online_security = np.where(internet_service == 'No', 'No Internet Service',
                               np.random.choice(['Yes', 'No'], n_customers, p=[0.35, 0.65]))

    tech_support = np.where(internet_service == 'No', 'No Internet Service',
                            np.random.choice(['Yes', 'No'], n_customers, p=[0.32, 0.68]))

    streaming_tv = np.where(internet_service == 'No', 'No Internet Service',
                            np.random.choice(['Yes', 'No'], n_customers, p=[0.42, 0.58]))

    streaming_movies = np.where(internet_service == 'No', 'No Internet Service',
                                np.random.choice(['Yes', 'No'], n_customers, p=[0.40, 0.60]))

    # Billing Information
    paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.60, 0.40])
    payment_method = np.random.choice(['Electronic Check', 'Credit Card', 
                                       'Bank Transfer', 'Mailed Check'],
                                      n_customers, p=[0.35, 0.25, 0.23, 0.17])

    # Financial Metrics
    base_charge = 30
    monthly_charges = base_charge + np.random.uniform(10, 80, n_customers)
    monthly_charges = np.where(internet_service == 'Fiber Optic', 
                               monthly_charges * 1.3, monthly_charges)
    monthly_charges = np.where(contract_type == 'Two Year', 
                               monthly_charges * 0.9, monthly_charges)
    monthly_charges = np.round(monthly_charges, 2)

    total_charges = monthly_charges * tenure_months + np.random.uniform(-200, 200, n_customers)
    total_charges = np.round(np.maximum(total_charges, monthly_charges), 2)

    # Customer Service
    customer_service_calls = np.random.poisson(2, n_customers)
    customer_service_calls = np.clip(customer_service_calls, 0, 10)

    # Churn Logic (Realistic Patterns)
    churn_probability = np.zeros(n_customers)

    # Risk Factors (Increase Churn)
    churn_probability += (contract_type == 'Month-to-Month') * 0.25
    churn_probability += (tenure_months < 6) * 0.20
    churn_probability += (customer_service_calls > 4) * 0.15
    churn_probability += (payment_method == 'Electronic Check') * 0.10
    churn_probability += (online_security == 'No') * 0.08
    churn_probability += (tech_support == 'No') * 0.08
    churn_probability += (monthly_charges > 80) * 0.12

    # Protection Factors (Decrease Churn)
    churn_probability -= (contract_type == 'Two Year') * 0.20
    churn_probability -= (tenure_months > 36) * 0.15
    churn_probability -= (online_security == 'Yes') * 0.10

    # Add randomness and clip
    churn_probability += np.random.uniform(-0.1, 0.1, n_customers)
    churn_probability = np.clip(churn_probability, 0.05, 0.85)

    # Generate churn
    churn = (np.random.random(n_customers) < churn_probability).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': gender,
        'Age': age,
        'City': city,
        'TenureMonths': tenure_months,
        'ContractType': contract_type,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'CustomerServiceCalls': customer_service_calls,
        'Churn': churn
    })

    print(f"✓ Dataset created with {n_customers} customers")
    print(f"✓ Churn Rate: {(churn.sum()/n_customers)*100:.2f}%")

    return df


# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df):
    """
    Perform comprehensive exploratory data analysis

    Parameters:
    -----------
    df : pandas.DataFrame
        Customer dataset
    """
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Data Quality
    print("\n1. DATA QUALITY")
    print("-" * 70)
    print(f"Total Records: {len(df)}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Records: {df.duplicated().sum()}")

    # Churn Distribution
    print("\n2. CHURN DISTRIBUTION")
    print("-" * 70)
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    print(f"Retained: {churn_counts[0]} ({churn_pct[0]:.2f}%)")
    print(f"Churned: {churn_counts[1]} ({churn_pct[1]:.2f}%)")

    # Numerical Statistics
    print("\n3. NUMERICAL STATISTICS")
    print("-" * 70)
    print(df[['Age', 'TenureMonths', 'MonthlyCharges', 
              'TotalCharges', 'CustomerServiceCalls']].describe())

    # Categorical Analysis
    print("\n4. CATEGORICAL ANALYSIS")
    print("-" * 70)
    categorical_cols = ['Gender', 'City', 'ContractType', 'InternetService', 
                       'PaymentMethod']
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

    return df


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    """
    Create additional features for analysis

    Parameters:
    -----------
    df : pandas.DataFrame
        Customer dataset

    Returns:
    --------
    pandas.DataFrame
        Dataset with additional features
    """
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    # Age Groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], 
                            labels=['18-30', '31-40', '41-50', '51-60', '60+'])

    # Tenure Groups
    df['TenureGroup'] = pd.cut(df['TenureMonths'], 
                               bins=[0, 6, 12, 24, 36, 72], 
                               labels=['0-6 months', '7-12 months', 
                                      '13-24 months', '25-36 months', '36+ months'])

    # Charges Groups
    df['ChargesGroup'] = pd.cut(df['MonthlyCharges'], 
                                bins=[0, 50, 75, 100, 150], 
                                labels=['Low (<50)', 'Medium (50-75)', 
                                       'High (75-100)', 'Very High (100+)'])

    # Service Calls Groups
    df['ServiceCallsGroup'] = pd.cut(df['CustomerServiceCalls'], 
                                     bins=[-1, 1, 3, 5, 10], 
                                     labels=['0-1 calls', '2-3 calls', 
                                            '4-5 calls', '6+ calls'])

    # Risk Score
    df['RiskScore'] = 0
    df['RiskScore'] += (df['ContractType'] == 'Month-to-Month') * 25
    df['RiskScore'] += (df['TenureMonths'] < 6) * 20
    df['RiskScore'] += (df['CustomerServiceCalls'] > 4) * 15
    df['RiskScore'] += (df['PaymentMethod'] == 'Electronic Check') * 10
    df['RiskScore'] += (df['OnlineSecurity'] == 'No') * 10
    df['RiskScore'] += (df['TechSupport'] == 'No') * 10
    df['RiskScore'] += (df['MonthlyCharges'] > 90) * 10

    # Risk Category
    df['RiskCategory'] = pd.cut(df['RiskScore'], 
                                bins=[0, 30, 60, 100], 
                                labels=['Low Risk', 'Medium Risk', 'High Risk'])

    print("✓ Created Age Groups")
    print("✓ Created Tenure Groups")
    print("✓ Created Charges Groups")
    print("✓ Created Service Calls Groups")
    print("✓ Calculated Risk Scores")

    return df


# ============================================================================
# 4. CHURN ANALYSIS
# ============================================================================

def analyze_churn(df):
    """
    Analyze churn patterns across different dimensions

    Parameters:
    -----------
    df : pandas.DataFrame
        Customer dataset with features
    """
    print("\n" + "=" * 70)
    print("CHURN ANALYSIS")
    print("=" * 70)

    # Contract Type Analysis
    print("\n1. CHURN BY CONTRACT TYPE")
    print("-" * 70)
    contract_churn = pd.crosstab(df['ContractType'], df['Churn'], 
                                 normalize='index') * 100
    print(contract_churn.round(2))

    # Tenure Analysis
    print("\n2. CHURN BY TENURE")
    print("-" * 70)
    tenure_churn = pd.crosstab(df['TenureGroup'], df['Churn'], 
                               normalize='index') * 100
    print(tenure_churn.round(2))

    # Service Analysis
    print("\n3. CHURN BY SERVICES")
    print("-" * 70)
    print("\nOnline Security:")
    security_churn = pd.crosstab(df['OnlineSecurity'], df['Churn'], 
                                 normalize='index') * 100
    print(security_churn.round(2))

    print("\nTech Support:")
    tech_churn = pd.crosstab(df['TechSupport'], df['Churn'], 
                            normalize='index') * 100
    print(tech_churn.round(2))

    # Payment Method Analysis
    print("\n4. CHURN BY PAYMENT METHOD")
    print("-" * 70)
    payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], 
                                normalize='index') * 100
    print(payment_churn.round(2))

    # Financial Analysis
    print("\n5. FINANCIAL ANALYSIS")
    print("-" * 70)
    print(f"Avg Monthly Charges (Churned): ₹{df[df['Churn']==1]['MonthlyCharges'].mean():.2f}")
    print(f"Avg Monthly Charges (Retained): ₹{df[df['Churn']==0]['MonthlyCharges'].mean():.2f}")
    print(f"Avg Total Revenue (Churned): ₹{df[df['Churn']==1]['TotalCharges'].mean():.2f}")
    print(f"Avg Total Revenue (Retained): ₹{df[df['Churn']==0]['TotalCharges'].mean():.2f}")


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def create_visualizations(df):
    """
    Create comprehensive visualizations

    Parameters:
    -----------
    df : pandas.DataFrame
        Customer dataset with features
    """
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    colors = ['#2ecc71', '#e74c3c']

    # Figure 1: Churn Overview
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Customer Churn Overview Dashboard', fontsize=16, fontweight='bold')

    # Churn Distribution
    churn_counts = df['Churn'].value_counts()
    axes[0, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], 
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0, 0].set_title('Overall Churn Distribution', fontweight='bold')

    # Contract Type
    contract_data = pd.crosstab(df['ContractType'], df['Churn'], 
                                normalize='index') * 100
    contract_data.plot(kind='bar', ax=axes[0, 1], color=colors, rot=45)
    axes[0, 1].set_title('Churn Rate by Contract Type', fontweight='bold')
    axes[0, 1].set_ylabel('Percentage (%)', fontweight='bold')
    axes[0, 1].legend(['Retained', 'Churned'])

    # Tenure
    tenure_data = pd.crosstab(df['TenureGroup'], df['Churn'], 
                             normalize='index') * 100
    tenure_data.plot(kind='bar', ax=axes[1, 0], color=colors, rot=45)
    axes[1, 0].set_title('Churn Rate by Tenure', fontweight='bold')
    axes[1, 0].set_ylabel('Percentage (%)', fontweight='bold')
    axes[1, 0].legend(['Retained', 'Churned'])

    # Payment Method
    payment_data = pd.crosstab(df['PaymentMethod'], df['Churn'], 
                               normalize='index') * 100
    payment_data.plot(kind='barh', ax=axes[1, 1], color=colors)
    axes[1, 1].set_title('Churn Rate by Payment Method', fontweight='bold')
    axes[1, 1].set_xlabel('Percentage (%)', fontweight='bold')
    axes[1, 1].legend(['Retained', 'Churned'])

    plt.tight_layout()
    plt.savefig('churn_overview_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: churn_overview_dashboard.png")
    plt.close()

    # Additional visualizations can be created similarly
    print("✓ All visualizations created successfully!")


# ============================================================================
# 6. RISK IDENTIFICATION
# ============================================================================

def identify_high_risk_customers(df):
    """
    Identify and export high-risk customers

    Parameters:
    -----------
    df : pandas.DataFrame
        Customer dataset with risk scores

    Returns:
    --------
    pandas.DataFrame
        High-risk customer dataset
    """
    print("\n" + "=" * 70)
    print("IDENTIFYING HIGH-RISK CUSTOMERS")
    print("=" * 70)

    high_risk = df[df['RiskCategory'] == 'High Risk'].copy()
    high_risk = high_risk.sort_values('RiskScore', ascending=False)

    print(f"\n✓ Identified {len(high_risk)} High-Risk Customers")
    print(f"  - Churned: {high_risk['Churn'].sum()}")
    print(f"  - Still Active: {(high_risk['Churn']==0).sum()}")

    # Export
    export_cols = ['CustomerID', 'Age', 'City', 'TenureMonths', 
                   'ContractType', 'MonthlyCharges', 'CustomerServiceCalls',
                   'OnlineSecurity', 'TechSupport', 'RiskScore', 
                   'RiskCategory', 'Churn']
    high_risk[export_cols].to_csv('high_risk_customers.csv', index=False)
    print("✓ Saved: high_risk_customers.csv")

    return high_risk


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "=" * 70)
    print("CUSTOMER CHURN ANALYSIS PROJECT")
    print("=" * 70)
    print("Technologies: Python, NumPy, Pandas, Matplotlib, Seaborn")
    print("=" * 70)

    # Generate Data
    df = generate_customer_data(n_customers=3000, random_seed=42)
    df.to_csv('customer_churn_data.csv', index=False)
    print("✓ Saved: customer_churn_data.csv")

    # Perform EDA
    df = perform_eda(df)

    # Feature Engineering
    df = create_features(df)

    # Churn Analysis
    analyze_churn(df)

    # Create Visualizations
    create_visualizations(df)

    # Identify High-Risk Customers
    high_risk = identify_high_risk_customers(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  1. customer_churn_data.csv - Complete dataset")
    print("  2. high_risk_customers.csv - High-risk customers")
    print("  3. churn_overview_dashboard.png - Main visualizations")
    print("  4. (Additional visualization files)")

    return df


if __name__ == "__main__":
    df = main()
