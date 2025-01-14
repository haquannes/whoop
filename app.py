import os
import sys

# Add the project directory to Python path
project_dir = "/Users/hakandurgut/Documents/code/whoop"
if project_dir not in sys.path:
    sys.path.append(project_dir)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Whoop Data Analysis", layout="wide")

st.title("Whoop Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Choose your Whoop data CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Convert time columns to datetime
    time_columns = ['Cycle start time', 'Cycle end time', 'Sleep onset', 'Wake onset']
    for col in time_columns:
        df[col] = pd.to_datetime(df[col])

    # Calculate sleep duration in hours
    df['Sleep Duration'] = (df['Wake onset'] - df['Sleep onset']).dt.total_seconds() / 3600

    # Display basic statistics
    st.header("Data Overview")
    st.write(f"Total number of records: {len(df)}")
    
    # Calculate date range
    date_range = df['Cycle end time'].max() - df['Cycle start time'].min()
    st.write(f"Date range: {date_range.days} days")
    
    # Display the first few rows of the data
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    # Seasonality Analysis
    st.header("Seasonality Analysis")
    
    # Add day of week and month information
    df['Day_of_Week'] = df['Cycle start time'].dt.day_name()
    df['Month'] = df['Cycle start time'].dt.month_name()
    df['Is_Weekend'] = df['Cycle start time'].dt.dayofweek.isin([5, 6])
    
    # Order days of week correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Weekly Patterns
    st.subheader("Weekly Patterns")
    
    # Create tabs for different metrics
    weekly_tabs = st.tabs(["Sleep Performance", "Recovery Score", "Day Strain", "Sleep Duration"])
    
    with weekly_tabs[0]:
        # Sleep Performance by Day of Week
        fig = px.box(df, x='Day_of_Week', y='Sleep performance %',
                    category_orders={'Day_of_Week': day_order},
                    title='Sleep Performance Distribution by Day of Week')
        st.plotly_chart(fig)
        
        # Weekend vs Weekday comparison
        weekend_sleep = df[df['Is_Weekend']]['Sleep performance %'].mean()
        weekday_sleep = df[~df['Is_Weekend']]['Sleep performance %'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekend Sleep Performance", f"{weekend_sleep:.1f}%")
        with col2:
            st.metric("Weekday Sleep Performance", f"{weekday_sleep:.1f}%")
    
    with weekly_tabs[1]:
        # Recovery Score by Day of Week
        fig = px.box(df, x='Day_of_Week', y='Recovery score %',
                    category_orders={'Day_of_Week': day_order},
                    title='Recovery Score Distribution by Day of Week')
        st.plotly_chart(fig)
        
        # Weekend vs Weekday comparison
        weekend_recovery = df[df['Is_Weekend']]['Recovery score %'].mean()
        weekday_recovery = df[~df['Is_Weekend']]['Recovery score %'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekend Recovery Score", f"{weekend_recovery:.1f}%")
        with col2:
            st.metric("Weekday Recovery Score", f"{weekday_recovery:.1f}%")
    
    with weekly_tabs[2]:
        # Day Strain by Day of Week
        fig = px.box(df, x='Day_of_Week', y='Day Strain',
                    category_orders={'Day_of_Week': day_order},
                    title='Day Strain Distribution by Day of Week')
        st.plotly_chart(fig)
        
        # Weekend vs Weekday comparison
        weekend_strain = df[df['Is_Weekend']]['Day Strain'].mean()
        weekday_strain = df[~df['Is_Weekend']]['Day Strain'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekend Day Strain", f"{weekend_strain:.1f}")
        with col2:
            st.metric("Weekday Day Strain", f"{weekday_strain:.1f}")
    
    with weekly_tabs[3]:
        # Sleep Duration by Day of Week
        fig = px.box(df, x='Day_of_Week', y='Sleep Duration',
                    category_orders={'Day_of_Week': day_order},
                    title='Sleep Duration Distribution by Day of Week')
        st.plotly_chart(fig)
        
        # Weekend vs Weekday comparison
        weekend_duration = df[df['Is_Weekend']]['Sleep Duration'].mean()
        weekday_duration = df[~df['Is_Weekend']]['Sleep Duration'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekend Sleep Duration", f"{weekend_duration:.1f} hrs")
        with col2:
            st.metric("Weekday Sleep Duration", f"{weekday_duration:.1f} hrs")
    
    # Monthly Patterns
    st.subheader("Monthly Patterns")
    
    # Create tabs for different metrics
    monthly_tabs = st.tabs(["Sleep Performance", "Recovery Score", "Day Strain", "Sleep Duration"])
    
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    with monthly_tabs[0]:
        # Monthly Sleep Performance
        monthly_sleep = df.groupby('Month')['Sleep performance %'].agg(['mean', 'std']).reset_index()
        monthly_sleep = monthly_sleep.sort_values('Month', key=lambda x: pd.Categorical(x, categories=month_order))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_sleep['Month'],
            y=monthly_sleep['mean'],
            error_y=dict(type='data', array=monthly_sleep['std']),
            mode='lines+markers',
            name='Sleep Performance'
        ))
        fig.update_layout(title='Monthly Sleep Performance Trends',
                         xaxis_title='Month',
                         yaxis_title='Sleep Performance %')
        st.plotly_chart(fig)
    
    with monthly_tabs[1]:
        # Monthly Recovery Score
        monthly_recovery = df.groupby('Month')['Recovery score %'].agg(['mean', 'std']).reset_index()
        monthly_recovery = monthly_recovery.sort_values('Month', key=lambda x: pd.Categorical(x, categories=month_order))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_recovery['Month'],
            y=monthly_recovery['mean'],
            error_y=dict(type='data', array=monthly_recovery['std']),
            mode='lines+markers',
            name='Recovery Score'
        ))
        fig.update_layout(title='Monthly Recovery Score Trends',
                         xaxis_title='Month',
                         yaxis_title='Recovery Score %')
        st.plotly_chart(fig)
    
    with monthly_tabs[2]:
        # Monthly Day Strain
        monthly_strain = df.groupby('Month')['Day Strain'].agg(['mean', 'std']).reset_index()
        monthly_strain = monthly_strain.sort_values('Month', key=lambda x: pd.Categorical(x, categories=month_order))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_strain['Month'],
            y=monthly_strain['mean'],
            error_y=dict(type='data', array=monthly_strain['std']),
            mode='lines+markers',
            name='Day Strain'
        ))
        fig.update_layout(title='Monthly Day Strain Trends',
                         xaxis_title='Month',
                         yaxis_title='Day Strain')
        st.plotly_chart(fig)
    
    with monthly_tabs[3]:
        # Monthly Sleep Duration
        monthly_duration = df.groupby('Month')['Sleep Duration'].agg(['mean', 'std']).reset_index()
        monthly_duration = monthly_duration.sort_values('Month', key=lambda x: pd.Categorical(x, categories=month_order))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_duration['Month'],
            y=monthly_duration['mean'],
            error_y=dict(type='data', array=monthly_duration['std']),
            mode='lines+markers',
            name='Sleep Duration'
        ))
        fig.update_layout(title='Monthly Sleep Duration Trends',
                         xaxis_title='Month',
                         yaxis_title='Sleep Duration (hours)')
        st.plotly_chart(fig)
    
    # Statistical Tests
    st.subheader("Statistical Analysis")
    
    # Perform t-tests for weekend vs weekday differences
    metrics = {
        'Sleep performance %': 'Sleep Performance',
        'Recovery score %': 'Recovery Score',
        'Day Strain': 'Day Strain',
        'Sleep Duration': 'Sleep Duration'
    }
    
    for metric, name in metrics.items():
        weekend_data = df[df['Is_Weekend']][metric]
        weekday_data = df[~df['Is_Weekend']][metric]
        
        t_stat, p_value = stats.ttest_ind(weekend_data, weekday_data)
        
        st.write(f"\n{name} - Weekend vs Weekday:")
        st.write(f"t-statistic: {t_stat:.2f}")
        st.write(f"p-value: {p_value:.4f}")
        if p_value < 0.05:
            st.write(f"**Significant difference detected** between weekend and weekday {name.lower()}")
        else:
            st.write(f"No significant difference detected between weekend and weekday {name.lower()}")

    # Sleep Profile Clustering Analysis
    st.header("Sleep Profile Analysis")
    
    # Prepare data for clustering
    sleep_features = ['Sleep Duration', 'Sleep performance %', 'Heart rate variability (ms)', 
                     'Resting heart rate (bpm)', 'Recovery score %']
    
    # Create a copy of the features for clustering
    X = df[sleep_features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    K = range(2, 6)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    optimal_k = K[np.argmax(silhouette_scores)]
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Sleep Profile'] = kmeans.fit_predict(X_scaled)
    
    # Analyze cluster characteristics
    cluster_stats = df.groupby('Sleep Profile')[sleep_features].mean()
    
    # Determine cluster labels based on characteristics
    profile_labels = {}
    for cluster in range(optimal_k):
        stats = cluster_stats.loc[cluster]
        if stats['Sleep Duration'] > cluster_stats['Sleep Duration'].mean():
            if stats['Sleep performance %'] > cluster_stats['Sleep performance %'].mean():
                label = "Optimal Sleepers"
            else:
                label = "Long but Inefficient Sleepers"
        else:
            if stats['Sleep performance %'] > cluster_stats['Sleep performance %'].mean():
                label = "Short but Efficient Sleepers"
            else:
                label = "Poor Sleepers"
        profile_labels[cluster] = label
    
    # Add profile labels to dataframe
    df['Sleep Profile Label'] = df['Sleep Profile'].map(profile_labels)
    
    # Display cluster characteristics
    st.subheader("Sleep Profile Characteristics")
    for cluster in range(optimal_k):
        st.write(f"\nProfile: {profile_labels[cluster]}")
        cluster_data = cluster_stats.loc[cluster]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Sleep Duration", f"{cluster_data['Sleep Duration']:.1f} hrs")
            st.metric("Avg Sleep Performance", f"{cluster_data['Sleep performance %']:.1f}%")
        with col2:
            st.metric("Avg HRV", f"{cluster_data['Heart rate variability (ms)']:.1f} ms")
            st.metric("Avg RHR", f"{cluster_data['Resting heart rate (bpm)']:.1f} bpm")
        with col3:
            st.metric("Avg Recovery Score", f"{cluster_data['Recovery score %']:.1f}%")
    
    # Visualize clusters
    st.subheader("Sleep Profile Visualization")
    
    # Create scatter plot of Sleep Duration vs Sleep Performance
    fig = px.scatter(df, 
                    x='Sleep Duration', 
                    y='Sleep performance %',
                    color='Sleep Profile Label',
                    hover_data=['Recovery score %', 'Heart rate variability (ms)'],
                    title='Sleep Profiles: Duration vs Performance')
    st.plotly_chart(fig)
    
    # Analyze relationship between sleep profiles and next day metrics
    st.subheader("Sleep Profile Impact Analysis")
    
    # Calculate next day metrics
    df['Next_Day_Recovery'] = df['Recovery score %'].shift(-1)
    df['Next_Day_Strain'] = df['Day Strain'].shift(-1)
    df['Next_Day_RHR'] = df['Resting heart rate (bpm)'].shift(-1)
    
    # Create box plots for next day metrics by sleep profile
    metrics = {
        'Next_Day_Recovery': 'Next Day Recovery Score',
        'Next_Day_Strain': 'Next Day Strain',
        'Next_Day_RHR': 'Next Day Resting Heart Rate'
    }
    
    for metric, title in metrics.items():
        fig = px.box(df, 
                    x='Sleep Profile Label',
                    y=metric,
                    title=f'{title} by Sleep Profile')
        st.plotly_chart(fig)
    
    # 1. Day Strain vs Next Day Recovery Score
    df['Next_Day_Recovery'] = df['Recovery score %'].shift(-1)
    strain_recovery_corr = df['Day Strain'].corr(df['Next_Day_Recovery'])

    # 2. Sleep Performance vs Resting HR
    sleep_hr_corr = df['Sleep performance %'].corr(df['Resting heart rate (bpm)'])

    # 3. HRV vs Sleep Performance
    hrv_sleep_corr = df['Heart rate variability (ms)'].corr(df['Sleep performance %'])

    st.header("Correlation Analysis Results")
    st.write(f"1. Day Strain vs Next Day Recovery correlation: {strain_recovery_corr:.2f}")
    st.write(f"2. Sleep Performance vs Resting HR correlation: {sleep_hr_corr:.2f}")
    st.write(f"3. HRV vs Sleep Performance correlation: {hrv_sleep_corr:.2f}")

    # Create visualization functions
    def create_correlation_plots():
        # Day Strain vs Next Day Recovery
        fig1 = px.scatter(df, x='Day Strain', y='Next_Day_Recovery',
                         title=f'Day Strain vs Next Day Recovery (Correlation: {strain_recovery_corr:.2f})')
        st.plotly_chart(fig1)
        
        # Sleep Performance vs Resting HR
        fig2 = px.scatter(df, x='Sleep performance %', y='Resting heart rate (bpm)',
                         title=f'Sleep Performance vs Resting HR (Correlation: {sleep_hr_corr:.2f})')
        st.plotly_chart(fig2)
        
        # HRV vs Sleep Performance
        fig3 = px.scatter(df, x='Sleep performance %', y='Heart rate variability (ms)',
                         title=f'HRV vs Sleep Performance (Correlation: {hrv_sleep_corr:.2f})')
        st.plotly_chart(fig3)

    def analyze_hrv_by_sleep_efficiency():
        # Define high/low sleep efficiency threshold (median split)
        median_sleep = df['Sleep performance %'].median()
        high_sleep = df[df['Sleep performance %'] > median_sleep]['Heart rate variability (ms)'].mean()
        low_sleep = df[df['Sleep performance %'] <= median_sleep]['Heart rate variability (ms)'].mean()
        
        st.subheader("HRV Analysis by Sleep Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average HRV with high sleep performance", f"{high_sleep:.2f} ms")
        with col2:
            st.metric("Average HRV with low sleep performance", f"{low_sleep:.2f} ms")

    create_correlation_plots()
    analyze_hrv_by_sleep_efficiency()
