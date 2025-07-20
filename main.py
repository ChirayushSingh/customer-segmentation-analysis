import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statistics as st
from os import system, name
from termcolor import colored
from check import check
from plots import plot_customer_segments
import warnings
warnings.filterwarnings('ignore')

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

def create_sample_customer_data():
    """Create a sample customer database for segmentation analysis"""
    np.random.seed(42)
    n_customers = 1000
    
    # Generate customer data
    customer_data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(40, 15, n_customers).astype(int),
        'annual_income': np.random.normal(50000, 20000, n_customers),
        'spending_score': np.random.randint(1, 101, n_customers),
        'recency_days': np.random.exponential(30, n_customers).astype(int),
        'frequency': np.random.poisson(5, n_customers),
        'monetary_value': np.random.lognormal(6, 1, n_customers),
        'loyalty_years': np.random.exponential(2, n_customers)
    }
    
    # Ensure realistic ranges
    customer_data['age'] = np.clip(customer_data['age'], 18, 80)
    customer_data['annual_income'] = np.clip(customer_data['annual_income'], 20000, 150000)
    customer_data['recency_days'] = np.clip(customer_data['recency_days'], 1, 365)
    customer_data['frequency'] = np.clip(customer_data['frequency'], 1, 50)
    customer_data['monetary_value'] = np.clip(customer_data['monetary_value'], 10, 10000)
    customer_data['loyalty_years'] = np.clip(customer_data['loyalty_years'], 0.1, 15)
    
    return pd.DataFrame(customer_data)

def perform_rfm_analysis(data):
    """Perform RFM (Recency, Frequency, Monetary) analysis"""
    print(colored("=== RFM ANALYSIS ===", 'cyan', attrs=['bold']))
    
    # Calculate RFM scores (1-5 scale)
    data['R_score'] = pd.qcut(data['recency_days'], 5, labels=[5,4,3,2,1])  # Lower recency = higher score
    data['F_score'] = pd.qcut(data['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    data['M_score'] = pd.qcut(data['monetary_value'], 5, labels=[1,2,3,4,5])
    
    # Create RFM segments
    data['RFM_Score'] = data['R_score'].astype(str) + data['F_score'].astype(str) + data['M_score'].astype(str)
    data['RFM_Value'] = data['R_score'].astype(int) + data['F_score'].astype(int) + data['M_score'].astype(int)
    
    # Define customer segments based on RFM
    def segment_customers(row):
        if row['RFM_Value'] >= 9:
            return 'Champions'
        elif row['RFM_Value'] >= 7:
            return 'Loyal Customers'
        elif row['RFM_Value'] >= 5:
            return 'Potential Loyalists'
        elif row['R_score'] >= 4:
            return 'New Customers'
        elif row['F_score'] >= 3:
            return 'At Risk'
        else:
            return 'Lost Customers'
    
    data['RFM_Segment'] = data.apply(segment_customers, axis=1)
    
    # Display RFM statistics
    rfm_summary = data.groupby('RFM_Segment').agg({
        'customer_id': 'count',
        'recency_days': 'mean',
        'frequency': 'mean',
        'monetary_value': 'mean'
    }).round(2)
    
    print("\nRFM Segment Summary:")
    print(rfm_summary)
    
    return data

def perform_kmeans_clustering(data, n_clusters=5):
    """Perform K-means clustering on customer data"""
    print(colored(f"\n=== K-MEANS CLUSTERING (k={n_clusters}) ===", 'cyan', attrs=['bold']))
    
    # Select features for clustering
    features = ['age', 'annual_income', 'spending_score', 'recency_days', 'frequency', 'monetary_value']
    X = data[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['K_Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Display cluster statistics
    cluster_summary = data.groupby('K_Cluster')[features].mean().round(2)
    cluster_counts = data['K_Cluster'].value_counts().sort_index()
    
    print(f"\nCluster Distribution:")
    for i in range(n_clusters):
        print(f"Cluster {i}: {cluster_counts[i]} customers")
    
    print(f"\nCluster Characteristics:")
    print(cluster_summary)
    
    return data, kmeans, scaler

def calculate_customer_statistics(data):
    """Calculate comprehensive customer statistics"""
    print(colored("\n=== CUSTOMER DATABASE STATISTICS ===", 'cyan', attrs=['bold']))
    
    total_customers = len(data)
    
    # Basic statistics
    stats = {
        'Total Customers': total_customers,
        'Average Age': round(data['age'].mean(), 1),
        'Average Annual Income': f"${data['annual_income'].mean():,.0f}",
        'Average Spending Score': round(data['spending_score'].mean(), 1),
        'Average Recency (days)': round(data['recency_days'].mean(), 1),
        'Average Frequency': round(data['frequency'].mean(), 1),
        'Average Monetary Value': f"${data['monetary_value'].mean():,.2f}",
        'Average Loyalty (years)': round(data['loyalty_years'].mean(), 1)
    }
    
    # Display statistics
    print("\nCustomer Overview:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Age distribution
    age_ranges = {
        '18-25': len(data[(data['age'] >= 18) & (data['age'] <= 25)]),
        '26-35': len(data[(data['age'] >= 26) & (data['age'] <= 35)]),
        '36-45': len(data[(data['age'] >= 36) & (data['age'] <= 45)]),
        '46-55': len(data[(data['age'] >= 46) & (data['age'] <= 55)]),
        '56+': len(data[data['age'] >= 56])
    }
    
    print(f"\nAge Distribution:")
    for range_name, count in age_ranges.items():
        percentage = (count / total_customers) * 100
        print(f"{range_name}: {count} customers ({percentage:.1f}%)")
    
    # Income segments
    income_segments = {
        'Low Income (<$35k)': len(data[data['annual_income'] < 35000]),
        'Middle Income ($35k-$65k)': len(data[(data['annual_income'] >= 35000) & (data['annual_income'] < 65000)]),
        'High Income ($65k+)': len(data[data['annual_income'] >= 65000])
    }
    
    print(f"\nIncome Segments:")
    for segment, count in income_segments.items():
        percentage = (count / total_customers) * 100
        print(f"{segment}: {count} customers ({percentage:.1f}%)")

def main():
    clear()
    print(colored("CUSTOMER SEGMENTATION ANALYSIS SYSTEM", 'green', attrs=['bold']))
    print("=" * 50)
    
    # Load or create customer data
    try:
        print("Loading customer database...")
        customer_data = create_sample_customer_data()
        print(f"Successfully loaded {len(customer_data)} customer records.")
        
        # Calculate basic statistics
        calculate_customer_statistics(customer_data)
        
        # Perform RFM Analysis
        customer_data = perform_rfm_analysis(customer_data)
        
        # Perform K-means clustering
        customer_data, kmeans_model, scaler = perform_kmeans_clustering(customer_data)
        
        # Display segment analysis
        print(colored("\n=== SEGMENT ANALYSIS ===", 'cyan', attrs=['bold']))
        
        print("\nRFM Segment Distribution:")
        rfm_dist = customer_data['RFM_Segment'].value_counts()
        for segment, count in rfm_dist.items():
            percentage = (count / len(customer_data)) * 100
            print(f"{segment}: {count} customers ({percentage:.1f}%)")
        
        # Customer Lifetime Value estimation
        customer_data['CLV_Estimate'] = (
            customer_data['monetary_value'] * 
            customer_data['frequency'] * 
            customer_data['loyalty_years']
        )
        
        print(f"\nAverage Customer Lifetime Value: ${customer_data['CLV_Estimate'].mean():,.2f}")
        print(f"Top 10% CLV Average: ${customer_data['CLV_Estimate'].quantile(0.9):,.2f}")
        
        # Churn risk analysis
        high_risk_customers = len(customer_data[
            (customer_data['recency_days'] > 90) & 
            (customer_data['frequency'] < 3)
        ])
        churn_risk_percentage = (high_risk_customers / len(customer_data)) * 100
        print(f"\nChurn Risk Analysis:")
        print(f"High-risk customers: {high_risk_customers} ({churn_risk_percentage:.1f}%)")
        
        # Generate visualizations
        print(colored("\nGenerating customer segmentation plots...", 'yellow'))
        plot_customer_segments(customer_data)
        
        # Save processed data
        customer_data.to_csv('customer_segmentation_results.csv', index=False)
        print(colored("\nResults saved to 'customer_segmentation_results.csv'", 'green'))
        
        # List all generated files for download
        print(colored("\n=== FILES READY FOR DOWNLOAD ===", 'cyan', attrs=['bold']))
        import os
        files_to_download = [
            'customer_segmentation_analysis.png',
            'customer_detailed_analysis.png', 
            'customer_segmentation_results.csv',
            'main.py'
        ]
        
        for file in files_to_download:
            if os.path.exists(file):
                file_size = os.path.getsize(file)
                print(f"✓ {file} ({file_size:,} bytes)")
            else:
                print(f"✗ {file} (not found)")
        
        print(colored("\nTo download files: Right-click on each file in the file explorer and select 'Download'", 'yellow'))
        
    except Exception as e:
        print(colored(f"Error: {str(e)}", 'red', attrs=['bold']))

if __name__ == "__main__":
    main()
