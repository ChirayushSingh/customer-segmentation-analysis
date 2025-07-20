import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_customer_segments(data):
    """Create comprehensive customer segmentation visualizations"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. RFM Segment Distribution
    plt.subplot(3, 3, 1)
    rfm_counts = data['RFM_Segment'].value_counts()
    plt.pie(rfm_counts.values, labels=rfm_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('RFM Segment Distribution', fontsize=12, fontweight='bold')
    
    # 2. Age vs Income by RFM Segment
    plt.subplot(3, 3, 2)
    for segment in data['RFM_Segment'].unique():
        segment_data = data[data['RFM_Segment'] == segment]
        plt.scatter(segment_data['age'], segment_data['annual_income'], 
                   label=segment, alpha=0.6, s=30)
    plt.xlabel('Age')
    plt.ylabel('Annual Income ($)')
    plt.title('Age vs Income by RFM Segment')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. K-means Clusters
    plt.subplot(3, 3, 3)
    scatter = plt.scatter(data['annual_income'], data['spending_score'], 
                         c=data['K_Cluster'], cmap='viridis', alpha=0.6, s=30)
    plt.xlabel('Annual Income ($)')
    plt.ylabel('Spending Score')
    plt.title('K-means Clusters (Income vs Spending)')
    plt.colorbar(scatter, label='Cluster')
    
    # 4. RFM Heatmap
    plt.subplot(3, 3, 4)
    # Convert categorical scores to numeric for heatmap calculation
    data_numeric = data.copy()
    data_numeric['R_score_num'] = data['R_score'].astype(int)
    data_numeric['F_score_num'] = data['F_score'].astype(int)
    data_numeric['M_score_num'] = data['M_score'].astype(int)
    
    rfm_pivot = data_numeric.groupby(['R_score_num', 'F_score_num'])['M_score_num'].mean().unstack()
    sns.heatmap(rfm_pivot, annot=True, cmap='RdYlBu_r', fmt='.1f')
    plt.title('RFM Heatmap (R vs F, values = M)')
    plt.xlabel('Frequency Score')
    plt.ylabel('Recency Score')
    
    # 5. Customer Lifetime Value Distribution
    plt.subplot(3, 3, 5)
    plt.hist(data['CLV_Estimate'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Customer Lifetime Value ($)')
    plt.ylabel('Number of Customers')
    plt.title('CLV Distribution')
    plt.axvline(data['CLV_Estimate'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${data["CLV_Estimate"].mean():,.0f}')
    plt.legend()
    
    # 6. Recency vs Frequency
    plt.subplot(3, 3, 6)
    plt.scatter(data['recency_days'], data['frequency'], 
               c=data['monetary_value'], cmap='plasma', alpha=0.6, s=30)
    plt.xlabel('Recency (days)')
    plt.ylabel('Frequency')
    plt.title('Recency vs Frequency (Color = Monetary Value)')
    plt.colorbar(label='Monetary Value ($)')
    
    # 7. Age Distribution by Segment
    plt.subplot(3, 3, 7)
    segments = data['RFM_Segment'].unique()
    age_data = [data[data['RFM_Segment'] == seg]['age'].values for seg in segments]
    plt.boxplot(age_data, labels=segments)
    plt.xlabel('RFM Segment')
    plt.ylabel('Age')
    plt.title('Age Distribution by RFM Segment')
    plt.xticks(rotation=45)
    
    # 8. Spending Patterns
    plt.subplot(3, 3, 8)
    spending_by_segment = data.groupby('RFM_Segment')['spending_score'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(spending_by_segment)), spending_by_segment.values, 
                   color='lightcoral', alpha=0.7)
    plt.xlabel('RFM Segment')
    plt.ylabel('Average Spending Score')
    plt.title('Average Spending Score by Segment')
    plt.xticks(range(len(spending_by_segment)), spending_by_segment.index, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 9. Monthly Revenue by Segment
    plt.subplot(3, 3, 9)
    revenue_by_segment = data.groupby('RFM_Segment')['monetary_value'].sum().sort_values(ascending=False)
    plt.pie(revenue_by_segment.values, labels=revenue_by_segment.index, 
            autopct='%1.1f%%', startangle=90)
    plt.title('Revenue Distribution by RFM Segment')
    
    plt.tight_layout()
    try:
        plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Main analysis plots saved as 'customer_segmentation_analysis.png'")
    except Exception as e:
        print(f"Warning: Could not save main plots - {e}")
    
    try:
        plt.show()
    except:
        pass  # In case display is not available
    
    # Create additional detailed plots
    create_detailed_analysis_plots(data)

def create_detailed_analysis_plots(data):
    """Create additional detailed analysis plots"""
    
    # Customer Journey Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loyalty vs CLV
    ax1.scatter(data['loyalty_years'], data['CLV_Estimate'], alpha=0.6, c='green', s=30)
    ax1.set_xlabel('Loyalty Years')
    ax1.set_ylabel('Customer Lifetime Value ($)')
    ax1.set_title('Customer Loyalty vs CLV')
    
    # Add trend line
    z = np.polyfit(data['loyalty_years'], data['CLV_Estimate'], 1)
    p = np.poly1d(z)
    ax1.plot(data['loyalty_years'], p(data['loyalty_years']), "r--", alpha=0.8)
    
    # 2. Income vs Spending by Age Groups
    age_groups = pd.cut(data['age'], bins=[18, 30, 45, 60, 80], labels=['18-30', '31-45', '46-60', '60+'])
    for group in age_groups.unique():
        if pd.notna(group):
            group_data = data[age_groups == group]
            ax2.scatter(group_data['annual_income'], group_data['spending_score'], 
                       label=group, alpha=0.6, s=30)
    ax2.set_xlabel('Annual Income ($)')
    ax2.set_ylabel('Spending Score')
    ax2.set_title('Income vs Spending by Age Groups')
    ax2.legend()
    
    # 3. Churn Risk Analysis
    churn_risk = data.copy()
    churn_risk['Churn_Risk'] = np.where(
        (churn_risk['recency_days'] > 90) & (churn_risk['frequency'] < 3), 'High Risk', 'Low Risk'
    )
    risk_counts = churn_risk['Churn_Risk'].value_counts()
    ax3.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
            colors=['red', 'lightgreen'], startangle=90)
    ax3.set_title('Churn Risk Distribution')
    
    # 4. Segment Performance Matrix
    segment_matrix = data.groupby('RFM_Segment').agg({
        'monetary_value': 'mean',
        'frequency': 'mean'
    })
    
    for i, segment in enumerate(segment_matrix.index):
        ax4.scatter(segment_matrix.loc[segment, 'frequency'], 
                   segment_matrix.loc[segment, 'monetary_value'],
                   s=200, alpha=0.7, label=segment)
        ax4.annotate(segment, 
                    (segment_matrix.loc[segment, 'frequency'], 
                     segment_matrix.loc[segment, 'monetary_value']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Average Frequency')
    ax4.set_ylabel('Average Monetary Value ($)')
    ax4.set_title('RFM Segment Performance Matrix')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig('customer_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Detailed analysis plots saved as 'customer_detailed_analysis.png'")
    except Exception as e:
        print(f"Warning: Could not save detailed plots - {e}")
    
    try:
        plt.show()
    except:
        pass  # In case display is not available

def plot(data, bins, show_bar_graph):
    """Legacy function for backward compatibility with original statistical analysis"""
    if isinstance(data, list):
        # Original statistical plotting
        fig, axs = plt.subplots(3 if show_bar_graph else 2, figsize=(10, 8))
        fig.suptitle("Statistical Analysis Plots")
        
        axs[0].boxplot(data, vert=False)
        axs[0].set_title("Boxplot")
        
        axs[1].hist(data, bins=int(bins), edgecolor='black')
        axs[1].set_title("Histogram")
        
        if show_bar_graph:
            unique_values = list(set(data))
            counts = [data.count(x) for x in unique_values]
            axs[2].bar(unique_values, counts, edgecolor="black")
            axs[2].set_title("Bar Graph")
        
        plt.tight_layout()
        plt.show()
    else:
        # If it's customer data, redirect to customer segmentation plots
        plot_customer_segments(data)
