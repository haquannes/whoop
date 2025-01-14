import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import plotly.io as pio

# Set dark theme as default for all plotly figures
pio.templates.default = "plotly_dark"

# Set page config with dark theme
st.set_page_config(page_title="Whoop Data Visualization", layout="wide")

# Set dark theme for Streamlit
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .streamlit-expanderHeader {
            background-color: #262730;
            color: #FAFAFA;
        }
        .css-1d391kg {
            background-color: #262730;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Whoop Data Visualization")

# File upload
uploaded_file = st.file_uploader("Upload your Whoop physiological cycles CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Convert time columns to datetime
    time_columns = ['Cycle start time', 'Cycle end time', 'Sleep onset', 'Wake onset']
    for col in time_columns:
        df[col] = pd.to_datetime(df[col])
    
    # Set the index to cycle start time for time series analysis
    df.set_index('Cycle start time', inplace=True)
    df.sort_index(inplace=True)
    
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create a container for the controls
    with st.sidebar:
        st.header("Controls")
        
        # Create sections for different types of metrics
        st.subheader("Select Metrics to Display")
        
        # Group metrics by type
        sleep_metrics = [col for col in numerical_cols if any(term in col.lower() for term in ['sleep', 'bed'])]
        strain_metrics = [col for col in numerical_cols if any(term in col.lower() for term in ['strain', 'cal', 'hr', 'heart'])]
        recovery_metrics = [col for col in numerical_cols if any(term in col.lower() for term in ['recovery', 'hrv', 'spo2', 'temp'])]
        other_metrics = [col for col in numerical_cols if col not in sleep_metrics + strain_metrics + recovery_metrics]
        
        # Dictionary to store checkbox states
        selected_fields = []
        
        # Sleep Metrics
        if sleep_metrics:
            st.write("Sleep Metrics")
            for field in sleep_metrics:
                if st.checkbox(field, key=f"cb_{field}"):
                    selected_fields.append(field)
            st.markdown("---")
        
        # Strain Metrics
        if strain_metrics:
            st.write("Strain & Activity Metrics")
            for field in strain_metrics:
                if st.checkbox(field, key=f"cb_{field}"):
                    selected_fields.append(field)
            st.markdown("---")
        
        # Recovery Metrics
        if recovery_metrics:
            st.write("Recovery Metrics")
            for field in recovery_metrics:
                if st.checkbox(field, key=f"cb_{field}"):
                    selected_fields.append(field)
            st.markdown("---")
        
        # Other Metrics
        if other_metrics:
            st.write("Other Metrics")
            for field in other_metrics:
                if st.checkbox(field, key=f"cb_{field}"):
                    selected_fields.append(field)
            st.markdown("---")
        
        # Smoothing slider
        smoothing_days = st.slider(
            "Smoothing window (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to use for rolling average smoothing"
        )
        
        # Option to show/hide original data
        show_original = st.checkbox("Show original data", value=False)
        
    if selected_fields:
        # Create figure
        fig = px.line(title="Whoop Metrics Over Time")
        
        # Add traces for each selected field
        for field in selected_fields:
            # Get the data and interpolate missing values
            field_data = df[field].copy()
            
            # Interpolate missing values using cubic spline
            field_data = field_data.interpolate(method='cubic', limit_direction='both')
            
            # Normalize the interpolated data
            normalized_data = (field_data - field_data.mean()) / field_data.std()
            
            if show_original:
                # Add original data as scattered points
                fig.add_scatter(
                    x=df.index,
                    y=normalized_data,
                    name=f"{field} (Original)",
                    mode='markers',
                    marker=dict(size=3),
                    opacity=0.3
                )
            
            # Add smoothed line with cubic spline interpolation
            smoothed_data = normalized_data.rolling(
                window=smoothing_days,
                center=True,
                min_periods=1  # Allow partial windows
            ).mean()
            
            # Additional smoothing using Savitzky-Golay filter
            from scipy.signal import savgol_filter
            window_length = min(smoothing_days * 2 + 1, len(smoothed_data) - 1)
            if window_length % 2 == 0:
                window_length += 1
            
            # Only apply Savitzky-Golay filter if window is large enough
            if window_length >= 5:  # Minimum window length for polynomial order 3
                polyorder = min(3, window_length - 2)  # Ensure polyorder is less than window_length
                smoothed_data = pd.Series(
                    savgol_filter(
                        smoothed_data,
                        window_length=window_length,
                        polyorder=polyorder,
                        mode='nearest'
                    ),
                    index=smoothed_data.index
                )
            
            fig.add_scatter(
                x=df.index,
                y=smoothed_data,
                name=f"{field} (Smoothed)",
                line=dict(
                    width=3,
                    shape='spline',  # Make lines curvy
                    smoothing=1.3    # Increase curve smoothness
                )
            )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalized Value (Z-score)",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            plot_bgcolor='#1f2630',  # Dark background
            paper_bgcolor='#1f2630',  # Dark background
            font=dict(color='#ffffff')  # White text
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(255,255,255,0.2)'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(255,255,255,0.2)'
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        if len(selected_fields) > 1:
            st.subheader(f"Correlation Analysis (Smoothing Window: {smoothing_days} days)")
            
            # Create smoothed dataframe for correlation analysis
            smoothed_df = pd.DataFrame()
            for field in selected_fields:
                # Get the data and interpolate missing values
                field_data = df[field].copy()
                field_data = field_data.interpolate(method='cubic', limit_direction='both')
                
                # Apply the same smoothing as in the plot
                smoothed_data = field_data.rolling(
                    window=smoothing_days,
                    center=True,
                    min_periods=1
                ).mean()
                
                # Apply Savitzky-Golay filter
                window_length = min(smoothing_days * 2 + 1, len(smoothed_data) - 1)
                if window_length % 2 == 0:
                    window_length += 1
                
                # Only apply Savitzky-Golay filter if window is large enough
                if window_length >= 5:  # Minimum window length for polynomial order 3
                    polyorder = min(3, window_length - 2)  # Ensure polyorder is less than window_length
                    smoothed_data = pd.Series(
                        savgol_filter(
                            smoothed_data,
                            window_length=window_length,
                            polyorder=polyorder,
                            mode='nearest'
                        ),
                        index=smoothed_data.index
                    )
                
                smoothed_df[field] = smoothed_data
            
            # Calculate correlation matrix for smoothed data
            corr_matrix = smoothed_df.corr()
            
            # Create two columns for correlation analysis
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create correlation heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    zmin=-1,  # Set minimum correlation value
                    zmax=1,   # Set maximum correlation value
                    range_color=[-1, 1]  # Fix the color range
                )
                
                # Update layout for correlation heatmap
                fig_corr.update_layout(
                    title=f"Correlation Matrix (Smoothed over {smoothing_days} days)",
                    width=600,
                    height=600,
                    plot_bgcolor='#1f2630',
                    paper_bgcolor='#1f2630',
                    font=dict(color='#ffffff')
                )
                
                # Update colorbar appearance
                fig_corr.update_traces(
                    colorbar=dict(
                        tickfont=dict(color='#ffffff'),
                        title=dict(
                            font=dict(color='#ffffff')
                        ),
                        tickvals=[-1, -0.5, 0, 0.5, 1],  # Set specific tick values
                        ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]  # Set tick labels
                    )
                )
                
                st.plotly_chart(fig_corr)
            
            with col2:
                # Create detailed correlation table
                st.write("Significant Correlations:")
                
                # Get all pairs of correlations
                correlations = []
                for i in range(len(selected_fields)):
                    for j in range(i+1, len(selected_fields)):
                        field1 = selected_fields[i]
                        field2 = selected_fields[j]
                        corr = corr_matrix.loc[field1, field2]
                        correlations.append({
                            'Field 1': field1,
                            'Field 2': field2,
                            'Correlation': corr,
                            'Strength': abs(corr)
                        })
                
                # Convert to DataFrame and sort by absolute correlation
                corr_df = pd.DataFrame(correlations)
                corr_df = corr_df.sort_values('Strength', ascending=False)
                
                # Format correlation values
                corr_df['Correlation'] = corr_df['Correlation'].apply(lambda x: f"{x:.3f}")
                corr_df['Interpretation'] = corr_df['Strength'].apply(
                    lambda x: "Very Strong" if x >= 0.8 else
                            "Strong" if x >= 0.6 else
                            "Moderate" if x >= 0.4 else
                            "Weak" if x >= 0.2 else
                            "Very Weak"
                )
                
                # Display the correlation table
                st.dataframe(
                    corr_df[['Field 1', 'Field 2', 'Correlation', 'Interpretation']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Add explanation
                st.markdown("""
                **Correlation Interpretation:**
                - ±0.8 to ±1.0: Very Strong
                - ±0.6 to ±0.8: Strong
                - ±0.4 to ±0.6: Moderate
                - ±0.2 to ±0.4: Weak
                - ±0.0 to ±0.2: Very Weak
                
                *Note: Correlations are calculated using smoothed data to focus on longer-term relationships.*
                """)
        
        # Show statistics
        st.subheader("Statistics")
        
        # Create columns for stats
        cols = st.columns(len(selected_fields))
        
        # Show basic statistics for each selected field
        for i, field in enumerate(selected_fields):
            with cols[i]:
                st.metric(
                    label=field,
                    value=f"{df[field].mean():.2f}",
                    delta=f"±{df[field].std():.2f}"
                )
                
                # Additional statistics
                stats = pd.DataFrame({
                    'Statistic': ['Min', 'Max', 'Median'],
                    'Value': [
                        f"{df[field].min():.2f}",
                        f"{df[field].max():.2f}",
                        f"{df[field].median():.2f}"
                    ]
                })
                st.dataframe(stats, hide_index=True)
    else:
        st.info("Please select at least one metric from the sidebar to visualize the data.")
