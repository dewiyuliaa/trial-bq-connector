import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import base64
import os

# Set page config
st.set_page_config(page_title="Audience Insight Dashboard", layout="wide")

# Initialize session states
if 'user_login' not in st.session_state:
    st.session_state.user_login = True

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'overview'

# New functions for audience size estimation
def format_audience_range(estimated_value):
    """Convert estimated audience to a range format with M/K formatting"""
    if estimated_value == 0:
        return "0"
    
    def format_single_value(value):
        if value >= 1000000:
            return f"{value/1000000:.1f}M"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        else:
            return f"{value:,}"
    
    if estimated_value < 10:
        return f"{estimated_value}"
    elif estimated_value < 100:
        base = round(estimated_value / 10) * 10
        lower = max(0, base - 5)
        upper = base + 5
        return f"{lower} - {upper}"
    elif estimated_value < 1000:
        base = round(estimated_value / 50) * 50
        lower = max(0, base - 25)
        upper = base + 25
        return f"{lower:,} - {upper:,}"
    elif estimated_value < 10000:
        base = round(estimated_value / 100) * 100
        lower = max(0, base - 100)
        upper = base + 100
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    elif estimated_value < 100000:
        base = round(estimated_value / 1000) * 1000
        lower = max(0, base - 500)
        upper = base + 500
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    elif estimated_value < 1000000:
        base = round(estimated_value / 5000) * 5000
        lower = max(0, base - 2500)
        upper = base + 2500
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    else:
        value_in_millions = estimated_value / 1000000
        
        if value_in_millions < 10:
            base = round(value_in_millions * 10) / 10
            lower = max(0, base - 0.1)
            upper = base + 0.1
            lower_formatted = f"{lower:.1f}M"
            upper_formatted = f"{upper:.1f}M"
        else:
            base = round(value_in_millions * 2) / 2
            lower = max(0, base - 0.5)
            upper = base + 0.5
            lower_formatted = f"{lower:.1f}M"
            upper_formatted = f"{upper:.1f}M"
        
        return f"{lower_formatted} - {upper_formatted}"

def format_number_display(value):
    """Format numbers for display - show millions as M, thousands as K"""
    if value == 0:
        return "0"
    elif value >= 1000000:
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    else:
        return f"{value:,}"

def predict_users_combined(daily_data, days_to_predict=1, use_last_n_days=30, user_login=True):
    """Predict users for multiple days based on historical data"""
    if len(daily_data) == 0:
        return 0
    
    daily_users_df = daily_data.reset_index()
    if user_login:
        daily_users_df.columns = ['date', 'unique_users']
    else:
        daily_users_df.columns = ['date', 'total_users']
        daily_users_df['unique_users'] = daily_users_df['total_users']
    
    daily_users_df['date'] = pd.to_datetime(daily_users_df['date'])
    daily_users_df = daily_users_df.sort_values('date').copy()
    
    if len(daily_users_df) > use_last_n_days:
        df_last_n = daily_users_df.iloc[-use_last_n_days:].copy()
    else:
        df_last_n = daily_users_df.copy()
    
    if len(df_last_n) == 0:
        return 0
    
    if days_to_predict == 1:
        moving_avg_days = min(7, len(df_last_n))
        last_7_avg = df_last_n['unique_users'].iloc[-moving_avg_days:].mean()
        
        last_7_days = df_last_n['unique_users'].iloc[-moving_avg_days:].reset_index(drop=True)
        
        if moving_avg_days == 7:
            weights = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.25])
        else:
            weights = np.linspace(0.5, 1.5, moving_avg_days)
            weights = weights / weights.sum()
        
        weighted_avg = (last_7_days * weights).sum()
        last_7_median = df_last_n['unique_users'].iloc[-moving_avg_days:].median()
        
        daily_prediction = (last_7_avg * 0.4) + (weighted_avg * 0.4) + (last_7_median * 0.2)
        
        return round(daily_prediction)
    
    else:
        if user_login:
            recent_avg = df_last_n['unique_users'].mean()
            
            if days_to_predict == 2:
                overlap_factor = 0.80
            elif days_to_predict == 3:
                overlap_factor = 0.70
            elif days_to_predict <= 5:
                overlap_factor = 0.60
            elif days_to_predict <= 7:
                overlap_factor = 0.55
            else:
                overlap_factor = 0.50
            
            conservative_estimate = recent_avg * days_to_predict * overlap_factor
            
            if len(daily_users_df) >= days_to_predict:
                similar_periods = []
                for i in range(len(daily_users_df) - days_to_predict + 1):
                    period_sum = daily_users_df['unique_users'].iloc[i:i+days_to_predict].sum()
                    similar_periods.append(period_sum)
                
                if similar_periods:
                    pattern_estimate = np.median(similar_periods) * overlap_factor
                else:
                    pattern_estimate = conservative_estimate
            else:
                pattern_estimate = conservative_estimate
            
            min_daily = df_last_n['unique_users'].min()
            min_estimate = min_daily * days_to_predict * (overlap_factor + 0.1)
            
            if len(df_last_n) >= 3:
                recent_3_avg = df_last_n['unique_users'].iloc[-3:].mean()
                trend_estimate = recent_3_avg * days_to_predict * overlap_factor
            else:
                trend_estimate = conservative_estimate
            
            final_prediction = (
                conservative_estimate * 0.25 +
                pattern_estimate * 0.35 +
                min_estimate * 0.15 +
                trend_estimate * 0.25
            )
            
            max_reasonable = recent_avg * days_to_predict * 0.90
            min_reasonable = min_daily * days_to_predict * 0.40
            
            final_prediction = min(final_prediction, max_reasonable)
            final_prediction = max(final_prediction, min_reasonable)
            
            return round(final_prediction)
        
        else:
            recent_avg = df_last_n['unique_users'].mean()
            avg_based_estimate = recent_avg * days_to_predict
            
            if len(daily_users_df) >= days_to_predict:
                similar_periods = []
                for i in range(len(daily_users_df) - days_to_predict + 1):
                    period_sum = daily_users_df['unique_users'].iloc[i:i+days_to_predict].sum()
                    similar_periods.append(period_sum)
                
                if similar_periods:
                    pattern_estimate = np.median(similar_periods)
                else:
                    pattern_estimate = avg_based_estimate
            else:
                pattern_estimate = avg_based_estimate
            
            if len(df_last_n) >= 3:
                recent_3_avg = df_last_n['unique_users'].iloc[-3:].mean()
                trend_estimate = recent_3_avg * days_to_predict
            else:
                trend_estimate = avg_based_estimate
            
            max_daily = df_last_n['unique_users'].max()
            optimistic_estimate = max_daily * days_to_predict * 0.85
            
            if len(df_last_n) >= 7:
                recent_7_days = df_last_n['unique_users'].iloc[-7:].reset_index(drop=True)
                weights = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.25])
                weighted_recent_avg = (recent_7_days * weights).sum()
                weighted_estimate = weighted_recent_avg * days_to_predict
            else:
                weighted_estimate = avg_based_estimate
            
            final_prediction = (
                avg_based_estimate * 0.15 +
                pattern_estimate * 0.35 +
                trend_estimate * 0.20 +
                optimistic_estimate * 0.15 +
                weighted_estimate * 0.15
            )
            
            min_daily = df_last_n['unique_users'].min()
            max_daily = df_last_n['unique_users'].max()
            
            min_reasonable = min_daily * days_to_predict * 0.8
            max_reasonable = max_daily * days_to_predict * 1.1
            
            avg_min_bound = recent_avg * days_to_predict * 0.85
            avg_max_bound = recent_avg * days_to_predict * 1.15
            
            final_min = min(min_reasonable, avg_min_bound)
            final_max = max(max_reasonable, avg_max_bound)
            
            final_prediction = max(final_prediction, final_min)
            final_prediction = min(final_prediction, final_max)
            
            return round(final_prediction)

def create_trend_chart(filtered_df, user_login=True, days_to_show=30):
    """Create trend chart showing last 30 days of data"""
    # Ensure we have date column
    if 'date' not in filtered_df.columns:
        return go.Figure()
    
    # Get daily data for the specified period
    end_date = pd.to_datetime(filtered_df['date'].max())
    start_date = end_date - timedelta(days=days_to_show-1)
    
    # Filter data for the trend period
    trend_data = filtered_df[pd.to_datetime(filtered_df['date']) >= start_date].copy()
    trend_data['date'] = pd.to_datetime(trend_data['date'])
    
    if user_login:
        # For user login data, use correct column names: 'user_id' and 'page_views'
        agg_dict = {}
        
        # Check for user column (user_id or userid)
        if 'user_id' in trend_data.columns:
            agg_dict['user_id'] = 'nunique'
            user_col = 'user_id'
        elif 'userid' in trend_data.columns:
            agg_dict['userid'] = 'nunique'
            user_col = 'userid'
        else:
            # Fallback: count rows
            daily_stats = trend_data.groupby('date').size().reset_index()
            daily_stats.columns = ['date', 'unique_users']
            daily_stats['total_pageviews'] = daily_stats['unique_users'] * 2
            return create_trend_chart_figure(daily_stats)
        
        # Check for pageviews column (page_views or pageviews)
        if 'page_views' in trend_data.columns:
            agg_dict['page_views'] = 'sum'
            views_col = 'page_views'
        elif 'pageviews' in trend_data.columns:
            agg_dict['pageviews'] = 'sum'
            views_col = 'pageviews'
        else:
            # Estimate pageviews as 2x users
            views_col = None
        
        daily_stats = trend_data.groupby('date').agg(agg_dict).reset_index()
        
        # Rename columns for consistency
        if len(daily_stats.columns) == 3:  # date + user + views
            daily_stats.columns = ['date', 'unique_users', 'total_pageviews']
        else:  # date + user only
            daily_stats.columns = ['date', 'unique_users']
            daily_stats['total_pageviews'] = daily_stats['unique_users'] * 2
            
    else:
        # For non-login data, use Total users column if available
        if 'Total users' in trend_data.columns:
            daily_stats = trend_data.groupby('date').agg({
                'Total users': 'sum',
                'Views': 'sum' if 'Views' in trend_data.columns else 'count'
            }).reset_index()
            daily_stats.columns = ['date', 'unique_users', 'total_pageviews']
        else:
            # Fallback: count rows as proxy for users
            daily_stats = trend_data.groupby('date').size().reset_index()
            daily_stats.columns = ['date', 'unique_users']
            daily_stats['total_pageviews'] = daily_stats['unique_users'] * 2  # Rough estimate
    
    return create_trend_chart_figure(daily_stats)

def create_trend_chart_figure(daily_stats):
    """Helper function to create the actual trend chart figure"""
    # Create combo chart with secondary y-axis
    fig = go.Figure()
    
    # Add audience bars (left y-axis)
    fig.add_trace(
        go.Bar(
            x=daily_stats['date'],
            y=daily_stats['unique_users'],
            name='Audience',
            marker=dict(
                color='rgba(59, 130, 246, 0.7)',
                line=dict(color='rgba(59, 130, 246, 1)', width=1),
                cornerradius=2
            ),
            hovertemplate='Date: %{x}<br>Audience: %{y:,}<extra></extra>',
            yaxis='y'
        )
    )
    
    # Add views line (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['total_pageviews'],
            mode='lines+markers',
            name='Views',
            line=dict(color='#EF4444', width=3),
            marker=dict(size=6, color='#EF4444'),
            hovertemplate='Date: %{x}<br>Views: %{y:,}<extra></extra>',
            yaxis='y2'
        )
    )
    
    # Update layout with dual y-axes
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        font=dict(size=12),
        bargap=0.2,  # Add space between bars
        # Primary y-axis (left) for Audience
        yaxis=dict(
            title=dict(text="Audience", font=dict(color="#3B82F6")),
            tickfont=dict(color="#3B82F6"),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot',  # Change to dotted lines like age chart
            side='left'
        ),
        # Secondary y-axis (right) for Views
        yaxis2=dict(
            title=dict(text="Views", font=dict(color="#EF4444")),
            tickfont=dict(color="#EF4444"),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        # X-axis
        xaxis=dict(
            showgrid=False,  # Remove vertical gridlines
            tickformat='%m/%d'
        )
    )
    
    return fig

# Chart functions
def create_city_chart(df, user_login=True):
    """Create city distribution chart with tosca color"""
    # Get filtered data (top 5 cities)
    if user_login:
        city_data = df['city'].value_counts().head(5)
    else:
        # For User Non Login, exclude "(not set)", get top 5 by total users, and sort by largest
        if 'Total users' in df.columns:
            filtered_cities = df[df['city'] != '(not set)']
            city_data = filtered_cities.groupby('city')['Total users'].sum().nlargest(5)
        else:
            filtered_cities = df[df['city'] != '(not set)']
            city_data = filtered_cities['city'].value_counts().head(5)
    
    # Calculate percentages
    total = city_data.sum()
    percentages = (city_data / total * 100) if total > 0 else city_data * 0
    
    # Create the chart
    fig = go.Figure()
    
    # Add bars with tosca color
    fig.add_trace(go.Bar(
        x=city_data.index,
        y=percentages,
        marker=dict(
            color='rgba(6, 182, 212, 0.8)',
            cornerradius=4,
            line=dict(width=0)
        ),
        text=[f'{pct:.1f}%' for pct in percentages],
        textposition='outside',
        hovertemplate='City: %{x}<br>Percentage: %{y:.1f}%<extra></extra>',
        showlegend=False
    ))
    
    # Get max value for y-axis range
    max_value = max(percentages) if len(percentages) > 0 else 0
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Percentage",
        showlegend=False,
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickangle=45,
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        ),
        yaxis=dict(
            range=[0, max_value * 1.15] if max_value > 0 else [0, 100],
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        )
    )
    
    return fig

def create_age_chart(df, user_login=True):
    """Create age distribution chart with tosca color"""
    # Get filtered data
    if user_login:
        age_data = df['age_group'].value_counts()
    else:
        # For User Non Login, use Total users for weighting
        if 'Total users' in df.columns:
            age_data = df.groupby('age_group')['Total users'].sum()
        else:
            age_data = df['age_group'].value_counts()
    
    # Define age order for consistent display (excluding Unknown and Other)
    age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    # Get all unique age groups from dataset, excluding Unknown and Other
    all_ages = set(age_data.index.tolist())
    ordered_ages = [age for age in age_order if age in all_ages and age not in ['Unknown', 'Other']]
    
    # Reindex to maintain order
    age_data = age_data.reindex(ordered_ages, fill_value=0)
    
    # Calculate percentages
    total = age_data.sum()
    percentages = (age_data / total * 100) if total > 0 else age_data * 0
    
    # Create the chart
    fig = go.Figure()
    
    # Add bars with tosca color
    fig.add_trace(go.Bar(
        x=ordered_ages,
        y=percentages,
        marker=dict(
            color='rgba(6, 182, 212, 0.8)',
            cornerradius=4,
            line=dict(width=0)
        ),
        text=[f'{pct:.1f}%' for pct in percentages],
        textposition='outside',
        hovertemplate='Age Group: %{x}<br>Percentage: %{y:.1f}%<extra></extra>',
        showlegend=False
    ))
    
    # Get max value for y-axis range
    max_value = max(percentages) if len(percentages) > 0 else 0
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Percentage",
        showlegend=False,
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickangle=45,
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        ),
        yaxis=dict(
            range=[0, max_value * 1.15] if max_value > 0 else [0, 100],
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        )
    )
    
    return fig

def create_gender_chart(df, user_login=True):
    """Create gender distribution pie chart"""
    if user_login:
        gender_data = df['sex'].value_counts()
    else:
        # For User Non Login, use Total users for weighting
        if 'Total users' in df.columns:
            gender_data = df.groupby('sex')['Total users'].sum()
        else:
            gender_data = df['sex'].value_counts()
    
    # Exclude unknown values
    gender_data = gender_data.drop(['unknown'], errors='ignore')
    
    # Calculate percentages
    total = gender_data.sum()
    percentages = (gender_data / total * 100) if total > 0 else gender_data * 0
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=[label.title() for label in gender_data.index],
        values=percentages,
        hole=0.5,
        marker=dict(
            colors=['rgba(79, 70, 229, 0.8)', 'rgba(6, 182, 212, 0.8)', 'rgba(16, 185, 129, 0.8)'],
            line=dict(color='rgba(255,255,255,0.8)', width=2)
        ),
        textinfo='none',
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
        showlegend=False
    )])
    
    fig.update_layout(
        height=110,
        margin=dict(l=0, r=100, t=0, b=0),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig, percentages

def create_device_chart(df, user_login=True):
    """Create device category distribution pie chart"""
    if user_login:
        device_data = df['device_category'].value_counts()
    else:
        # For User Non Login, use Total users for weighting
        if 'Total users' in df.columns:
            device_data = df.groupby('device_category')['Total users'].sum()
        else:
            device_data = df['device_category'].value_counts()
    
    # Calculate percentages
    total = device_data.sum()
    percentages = (device_data / total * 100) if total > 0 else device_data * 0
    
    # Create custom labels with percentages
    labels_with_pct = []
    for label, pct in zip(device_data.index, percentages):
        labels_with_pct.append(f"{label}<br>{pct:.1f}%")
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels_with_pct,
        values=percentages,
        hole=0.5,
        marker=dict(
            colors=['rgba(16, 185, 129, 0.8)', 'rgba(16, 185, 129, 0.6)', 'rgba(16, 185, 129, 0.4)'],
            line=dict(color='rgba(255,255,255,0.8)', width=2)
        ),
        textinfo='none',
        hovertemplate='%{label}<extra></extra>',
        showlegend=True
    )])
    
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=100, t=30, b=0),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=0.85,
            font=dict(size=11)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# Load and process data
@st.cache_data
def load_data():
    try:
        # Read file for df1
        df1 = pd.read_csv("cnbc.csv", encoding='utf-8')
        
        # Read and combine two files for df2
        df2_part1 = pd.read_csv("cnbc2-1.csv", encoding='utf-8')
        df2_part2 = pd.read_csv("cnbc2-2.csv", encoding='utf-8')
        
        # Combine the two df2 files
        df2 = pd.concat([df2_part1, df2_part2], ignore_index=True)
        
        # Process df1 (User Login data)
        df1['date'] = pd.to_datetime(df1['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # Add age_group column
        def safe_age_to_group(age_value):
            if pd.isna(age_value) or age_value == 'unknown' or age_value == '':
                return "Unknown"
            
            try:
                age_int = int(float(age_value))
                if 18 <= age_int <= 24:
                    return "18-24"
                elif 25 <= age_int <= 34:
                    return "25-34"
                elif 35 <= age_int <= 44:
                    return "35-44"
                elif 45 <= age_int <= 54:
                    return "45-54"
                elif 55 <= age_int <= 64:
                    return "55-64"
                elif age_int >= 65:
                    return "65+"
                else:
                    return "Other"
            except (ValueError, TypeError):
                return "Unknown"
        
        df1['age_group'] = df1['age'].apply(safe_age_to_group)
        
        # Define kanal_group function
        def categorize_kanal(kanalid, page_url=None):
            # First check page URL if provided
            if page_url == "https://www.cnbcindonesia.com/":
                return "WP"
            
            if pd.isna(kanalid):
                return "Other"
            
            kanalid_str = str(kanalid)
            
            if kanalid_str.startswith("2-3"):
                return "News"
            elif kanalid_str.startswith("2-5"):
                return "Market"
            elif kanalid_str.startswith("2-9"):
                return "Entrepreneur"
            elif kanalid_str.startswith("2-12"):
                return "Tech"
            elif kanalid_str.startswith("2-11"):
                return "Lifestyle"
            elif kanalid_str.startswith("2-10"):
                return "Syariah"
            elif kanalid_str.startswith("2-13"):
                return "Opini"
            elif kanalid_str.startswith("2-71"):
                return "MyMoney"
            elif kanalid_str.startswith("2-78"):
                return "Cuap Cuap Cuan"
            elif kanalid_str.startswith("2-127"):
                return "Research"
            else:
                return "Other"
        
        # Apply kanal categorization with page check if page column exists
        if 'page' in df1.columns:
            df1['kanal_group'] = df1.apply(lambda row: categorize_kanal(row['kanalid'], row.get('page')), axis=1)
        else:
            df1['kanal_group'] = df1['kanalid'].apply(lambda x: categorize_kanal(x))
        
        # Process df2 (User Non Login data)
        df2['date'] = pd.to_datetime(df2['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # Map Gender column to sex for consistency
        if 'Gender' in df2.columns:
            df2['sex'] = df2['Gender'].str.lower()
        
        # Use Age column directly as age_group for df2
        if 'Age' in df2.columns:
            df2['age_group'] = df2['Age']
        
        # Map Device category for consistency
        if 'Device category' in df2.columns:
            df2['device_category'] = df2['Device category']
        
        # Map City for consistency
        if 'City' in df2.columns:
            df2['city'] = df2['City']
        
        # Map Kanal ID and categorize with page check if available
        if 'Kanal ID' in df2.columns:
            # Define the categorize_kanal function for df2 (User Non Login)
            def categorize_kanal_df2(kanalid):
                if pd.isna(kanalid):
                    return "Other"
                
                kanalid_str = str(kanalid)
                
                # Special case for df2: kanalid "2" = "WP"
                if kanalid_str == "2":
                    return "WP"
                elif kanalid_str.startswith("2-3"):
                    return "News"
                elif kanalid_str.startswith("2-5"):
                    return "Market"
                elif kanalid_str.startswith("2-9"):
                    return "Entrepreneur"
                elif kanalid_str.startswith("2-12"):
                    return "Tech"
                elif kanalid_str.startswith("2-11"):
                    return "Lifestyle"
                elif kanalid_str.startswith("2-10"):
                    return "Syariah"
                elif kanalid_str.startswith("2-13"):
                    return "Opini"
                elif kanalid_str.startswith("2-71"):
                    return "MyMoney"
                elif kanalid_str.startswith("2-78"):
                    return "Cuap Cuap Cuan"
                elif kanalid_str.startswith("2-127"):
                    return "Research"
                else:
                    return "Other"
            
            # Apply kanal categorization for df2
            df2['kanal_group'] = df2['Kanal ID'].apply(categorize_kanal_df2)
        
        return df1, df2
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to get last 30 days date range
def get_last_30_days_range(df):
    """Get the date range for the last 30 days from the dataset"""
    max_date_str = df['date'].max()
    max_date = datetime.datetime.strptime(max_date_str, '%Y-%m-%d').date()
    start_date = max_date - timedelta(days=29)  # 30 days including end date
    return start_date, max_date

# Function to auto-download CSV data
def auto_download_csv(filtered_df):
    """Generate CSV data for download"""
    if not filtered_df.empty:
        # Check if contact columns exist in the filtered data
        contact_columns = []
        
        if 'full_name' in filtered_df.columns:
            contact_columns.append('full_name')
        if 'email' in filtered_df.columns:
            contact_columns.append('email') 
        if 'phone_number' in filtered_df.columns:
            contact_columns.append('phone_number')
        
        if contact_columns:
            # Create contact export data
            contact_df = filtered_df[contact_columns].copy()
            
            # Fix phone number format to remove .0
            if 'phone_number' in contact_df.columns:
                contact_df['phone_number'] = contact_df['phone_number'].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) and str(x) != 'nan' else x
                )
            
            # Rename columns to be more readable
            column_mapping = {
                'full_name': 'Full Name',
                'email': 'Email', 
                'phone_number': 'Phone Number'
            }
            contact_df = contact_df.rename(columns=column_mapping)
            
            # Remove duplicates if any
            contact_df = contact_df.drop_duplicates()
            
            # Convert to CSV
            csv_data = contact_df.to_csv(index=False)
            
            return csv_data, len(contact_df)
        else:
            return None, 0
    else:
        return None, 0

# Load data
df1, df2 = load_data()
if df1 is None or df2 is None:
    st.stop()

# Get last 30 days date range based on current dataset
current_df = df1 if st.session_state.user_login else df2
start_date, end_date = get_last_30_days_range(current_df)

# Get unique values for selectors from the appropriate dataframe
def get_filter_options(user_login):
    current_df = df1 if user_login else df2
    
    # Get top 10 cities by count/usage
    if 'city' in current_df.columns:
        if user_login:
            top_cities = current_df['city'].value_counts().head(10).index.tolist()
        else:
            # For User Non Login, exclude "(not set)" and use Total users for weighting
            if 'Total users' in current_df.columns:
                filtered_cities = current_df[current_df['city'] != '(not set)']
                top_cities = filtered_cities.groupby('city')['Total users'].sum().nlargest(10).index.tolist()
            else:
                filtered_cities = current_df[current_df['city'] != '(not set)']
                top_cities = filtered_cities['city'].value_counts().head(10).index.tolist()
        all_cities = sorted(top_cities)
    else:
        all_cities = []
    
    all_age_groups = sorted(current_df['age_group'].dropna().unique().tolist()) if 'age_group' in current_df.columns else []
    all_genders = sorted(current_df['sex'].dropna().unique().tolist()) if 'sex' in current_df.columns else []
    
    # Get kanal groups and exclude "Other"
    if 'kanal_group' in current_df.columns:
        all_kanals_raw = sorted(current_df['kanal_group'].dropna().unique().tolist())
        all_kanals = [kanal for kanal in all_kanals_raw if kanal != 'Other']
    else:
        all_kanals = []
    
    all_devices = sorted(current_df['device_category'].dropna().unique().tolist()) if 'device_category' in current_df.columns else []
    
    all_aws = []
    all_ages_raw = []
    all_paylater_status = []
    
    if user_login:
        if 'aws' in current_df.columns:
            all_aws = sorted(current_df['aws'].dropna().unique().tolist())
        
        if 'age' in current_df.columns:
            all_ages_raw = sorted([str(x) for x in current_df['age'].dropna().unique().tolist()])
        
        if 'paylater_status' in current_df.columns:
            all_paylater_status = sorted(current_df['paylater_status'].dropna().unique().tolist())
    
    all_categories = []
    if 'categoryauto_new_rank1' in current_df.columns:
        categories_data = current_df['categoryauto_new_rank1'].dropna().unique()
        if len(categories_data) > 0:
            all_categories = sorted(categories_data.tolist())
    
    return {
        'cities': all_cities,
        'age_groups': all_age_groups,
        'genders': all_genders,
        'kanals': all_kanals,
        'devices': all_devices,
        'categories': all_categories,
        'aws': all_aws,
        'ages_raw': all_ages_raw,
        'paylater_status': all_paylater_status
    }

# Get initial filter options
filter_options = get_filter_options(st.session_state.user_login)

# Sidebar configuration
st.sidebar.title("Filter audience")

# City selector
st.sidebar.markdown("### 📍 Locations")
selected_cities = st.sidebar.multiselect(
    "",
    filter_options['cities'],
    default=[],
    label_visibility="collapsed",
    key="city_selector",
    placeholder="Choose options"
)

# Age selector
st.sidebar.markdown("### 👤 Demographics")
selected_age = st.sidebar.multiselect(
    "Age",
    filter_options['age_groups'],
    default=[],
    key="age_selector",
    placeholder="Choose options"
)

# Gender selector
selected_genders = []
if len(filter_options['genders']) >= 2:
    gender_options = st.sidebar.multiselect(
        "Gender",
        ['Female', 'Male', 'Unknown'],
        default=[],
        key="gender_multiselect",
        placeholder="Choose options"
    )
    selected_genders = [g.lower() for g in gender_options]

# Kanal selector
st.sidebar.markdown("### 📺 Select kanal")
selected_kanal = st.sidebar.multiselect(
    "",
    filter_options['kanals'],
    default=[],
    label_visibility="collapsed",
    key="kanal_selector",
    placeholder="Choose options"
)

# Device selector
st.sidebar.markdown("### 📱 Select device")
selected_device = st.sidebar.multiselect(
    "",
    filter_options['devices'],
    default=[],
    label_visibility="collapsed",
    key="device_selector",
    placeholder="Choose options"
)

# AWS selector (User Login only)
selected_aws = []
if st.session_state.user_login and filter_options['aws']:
    st.sidebar.markdown("### 💳 Select Allo Wallet Status")
    selected_aws = st.sidebar.multiselect(
        "",
        filter_options['aws'],
        default=[],
        label_visibility="collapsed",
        key="aws_selector",
        placeholder="Choose options"
    )

# Paylater Status selector (User Login only)
selected_paylater = []
if st.session_state.user_login and filter_options['paylater_status']:
    st.sidebar.markdown("### 🏦 Select Paylater Status")
    selected_paylater = st.sidebar.multiselect(
        "",
        filter_options['paylater_status'],
        default=[],
        label_visibility="collapsed",
        key="paylater_selector",
        placeholder="Choose options"
    )

# Category selector
st.sidebar.markdown("### 🏷️ Select category")
selected_categories = st.sidebar.multiselect(
    "",
    filter_options['categories'],
    default=[],
    label_visibility="collapsed",
    key="category_selector",
    placeholder="Choose options"
)

# Reset Filters button
st.sidebar.markdown("<div style='margin-top: 130px;'></div>", unsafe_allow_html=True)
if st.sidebar.button("Reset", use_container_width=True, type="secondary"):
    filter_keys = [
        "city_selector", "age_selector", 
        "gender_multiselect", "kanal_selector", 
        "device_selector", "category_selector", "aws_selector", "paylater_selector"
    ]
    
    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state["city_selector"] = []
    st.session_state["age_selector"] = []
    st.session_state["gender_multiselect"] = []
    st.session_state["kanal_selector"] = []
    st.session_state["device_selector"] = []
    st.session_state["category_selector"] = []
    st.session_state["aws_selector"] = []
    st.session_state["paylater_selector"] = []
        
    st.rerun()

# CSS and Header styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main styling */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Force full width for header */
    .block-container {
        padding-top: 0 !important;
    }
    
    /* Full width header */
    .tiktok-header {
        background-color: #1a1a1a;
        margin: -1rem -100vw 1rem -100vw;
        padding: 12px 100vw;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 60px;
        border-bottom: 1px solid #333;
        position: relative;
    }
    
    .header-title {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
        padding-left: 24px;
    }
    
    .header-nav {
        display: flex;
        gap: 0;
        padding-right: 24px;
    }
    
    .nav-item {
        color: #9ca3af;
        padding: 8px 20px;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .nav-item:hover {
        color: #e5e7eb;
    }
    
    .nav-item.active {
        color: white;
        border-bottom-color: #ff0050;
    }
    
    /* Remove extra spacing in plotly charts */
    .js-plotly-plot {
        margin: 0 !important;
    }
    
    /* Streamlit plotly chart container */
    .stPlotlyChart {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Audience overview tabs */
    .audience-tabs {
        display: flex;
        gap: 2px;
        margin-bottom: 20px;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .audience-tab {
        padding: 12px 24px;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 500;
        color: #6b7280;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .audience-tab.active {
        color: #4f46e5;
        border-bottom-color: #4f46e5;
    }
    
    /* Style the Export contact button to look like text */
    button[title="Download contact data"] {
        background: transparent !important;
        border: none !important;
        color: #6b7280 !important;
        font-weight: 500 !important;
        padding: 12px 0 !important;
        box-shadow: none !important;
        text-decoration: none !important;
        font-size: 0.95rem !important;
    }
    button[title="Download contact data"]:hover {
        color: #4b5563 !important;
        text-decoration: underline !important;
        background: transparent !important;
        border: none !important;
    }
    button[title="Download contact data"]:focus {
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
        border: none !important;
    }
    
    /* Audience Size Card Styling */
    .audience-size-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
    }
    
    .audience-size-title {
        font-size: 1.1rem;
        margin-bottom: 8px;
        opacity: 0.9;
    }
    
    .audience-size-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 16px 0;
    }
    
    .audience-size-subtitle {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 16px;
    }
    
    .audience-size-disclaimer {
        font-size: 0.85rem;
        opacity: 0.7;
        line-height: 1.4;
    }
    
    /* Allo Wallet Cards Styling */
    .allo-wallet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .allo-wallet-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 8px;
        opacity: 0.9;
    }
    
    .allo-wallet-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .allo-wallet-subtitle {
        font-size: 0.8rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Create navigation with black header styling
st.markdown("""
<div style='background-color: #1a1a1a; margin: -1rem -100vw 1rem -100vw; padding: 12px 100vw; min-height: 60px; border-bottom: 1px solid #333; display: flex; align-items: center; justify-content: space-between;'>
    <div style='color: white; font-size: 1.2rem; font-weight: 600; padding-left: 24px;'>
        Prototype Audience Insight
    </div>
</div>
""", unsafe_allow_html=True)

# Create navigation buttons in a container that looks like the header
nav_container = st.container()
with nav_container:
    # Position the buttons in the top right
    col1, col2, col3, col4 = st.columns([5, 1.2, 1.4, 0.5])
    
    with col2:
        if st.button("User Login", 
                     key="login_nav_btn", 
                     use_container_width=True):
            if not st.session_state.user_login:
                st.session_state.user_login = True
                st.rerun()

    with col3:
        if st.button("User Non Login", 
                     key="non_login_nav_btn", 
                     use_container_width=True):
            if st.session_state.user_login:
                st.session_state.user_login = False
                st.rerun()

# Add CSS to style the buttons and position them like the original header
st.markdown(f"""
<style>
/* Position the navigation buttons to overlay on the black header */
.stButton {{
    position: relative;
    margin-top: -80px;
    z-index: 999;
}}

/* Style buttons to look like header navigation tabs - NO RED BORDER */
.stButton > button {{
    background-color: transparent !important;
    border: none !important;
    color: #9ca3af !important;
    padding: 8px 20px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    border-bottom: none !important;
    transition: all 0.2s ease !important;
    height: 45px !important;
}}

.stButton > button:hover {{
    color: #e5e7eb !important;
    background-color: transparent !important;
    border: none !important;
}}

/* Active state styling - NO RED BORDER */
.stButton > button[aria-pressed="true"] {{
    color: white !important;
    border-bottom: none !important;
    background-color: transparent !important;
}}

/* Style for User Login button - First button (col2) */
div[data-testid="column"]:nth-child(2) .stButton > button {{
    color: {'white' if st.session_state.user_login else '#9ca3af'} !important;
    border-bottom: none !important;
}}

/* Style for User Non Login button - Second button (col3) */
div[data-testid="column"]:nth-child(3) .stButton > button {{
    color: {'white' if not st.session_state.user_login else '#9ca3af'} !important;
    border-bottom: none !important;
}}

.stButton > button:focus {{
    outline: none !important;
    box-shadow: none !important;
}}

/* Adjust the main content margin */
.block-container {{
    margin-top: 20px !important;
}}
</style>
""", unsafe_allow_html=True)

# Apply filters to the dataframe
current_df = df1 if st.session_state.user_login else df2
filtered_df = current_df.copy()

# Apply date filter (last 30 days)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
filtered_df = filtered_df[(filtered_df['date'] >= start_date_str) & 
                          (filtered_df['date'] <= end_date_str)]

# Apply other filters
if selected_cities:
    filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

if selected_age:
    filtered_df = filtered_df[filtered_df['age_group'].isin(selected_age)]

if selected_genders:
    filtered_df = filtered_df[filtered_df['sex'].isin(selected_genders)]

if selected_kanal:
    filtered_df = filtered_df[filtered_df['kanal_group'].isin(selected_kanal)]

if selected_device:
    filtered_df = filtered_df[filtered_df['device_category'].isin(selected_device)]

if selected_categories and 'categoryauto_new_rank1' in current_df.columns:
    filtered_df = filtered_df[filtered_df['categoryauto_new_rank1'].isin(selected_categories)]

# Apply User Login specific filters
if st.session_state.user_login:
    if selected_aws and 'aws' in current_df.columns:
        filtered_df = filtered_df[filtered_df['aws'].isin(selected_aws)]
    
    if selected_paylater and 'paylater_status' in current_df.columns:
        filtered_df = filtered_df[filtered_df['paylater_status'].isin(selected_paylater)]

# Export contact section (KEEP POSITION)
export_col1, export_col2, export_spacer, export_col3 = st.columns([1.8, 1.5, 3.7, 1.5])

with export_col3:
    # Only show Export contact for User Login mode
    if st.session_state.user_login:
        # Generate CSV data for potential download
        csv_data, contact_count = auto_download_csv(filtered_df)
        
        if csv_data:
            # Direct download button styled as Export contact button
            st.download_button(
                label="⬇️ Export contact",
                data=csv_data,
                file_name="contact_list.csv",
                mime="text/csv",
                key="direct_download",
                help="Download contact data",
                use_container_width=False
            )
        else:
            # Show regular button if no data available
            if st.button("⬇️ Export contact", key="data_btn", help="Download contact data"):
                if not filtered_df.empty:
                    st.warning("❌ No contact data (full_name, email, phone_number) available in the current dataset.")
                else:
                    st.error("❌ No data available for the selected filters.")
    else:
        # For User Non Login, show empty space or alternative content
        st.markdown("")

# Add bottom border (KEEP POSITION)
st.markdown("""
<div style='border-bottom: 1px solid #e2e8f0; margin-top: -7px; margin-bottom: 20px;'></div>
""", unsafe_allow_html=True)

# Add some spacing before tab buttons
st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

# Audience overview tabs with Data download functionality (MOVED LOWER)
tab_col1, tab_col2, spacer_col = st.columns([2.2, 1.5, 6.3])

with tab_col1:
    # Audience Overview Tab Button
    if st.button("📈 Audience overview", key="overview_tab", use_container_width=True):
        st.session_state.active_tab = 'overview'

with tab_col2:
    # Interests Tab Button
    if st.button("💡 Interests", key="interests_tab", use_container_width=True):
        st.session_state.active_tab = 'interests'

# Add CSS to style the tab buttons
st.markdown(f"""
<style>
/* Style for Overview tab button */
div[data-testid="column"]:nth-child(1) .stButton > button {{
    background: transparent !important;
    border: none !important;
    color: {'#4f46e5' if st.session_state.active_tab == 'overview' else '#6b7280'} !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    border-bottom: {'2px solid #4f46e5' if st.session_state.active_tab == 'overview' else '2px solid transparent'} !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-size: 0.95rem !important;
}}

/* Style for Interests tab button */
div[data-testid="column"]:nth-child(2) .stButton > button {{
    background: transparent !important;
    border: none !important;
    color: {'#4f46e5' if st.session_state.active_tab == 'interests' else '#6b7280'} !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    border-bottom: {'2px solid #4f46e5' if st.session_state.active_tab == 'interests' else '2px solid transparent'} !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-size: 0.95rem !important;
}}

/* Tab button hover effects */
div[data-testid="column"]:nth-child(1) .stButton > button:hover,
div[data-testid="column"]:nth-child(2) .stButton > button:hover {{
    color: #4b5563 !important;
    background: transparent !important;
    border: none !important;
}}

div[data-testid="column"]:nth-child(1) .stButton > button:focus,
div[data-testid="column"]:nth-child(2) .stButton > button:focus {{
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    border: none !important;
}}
</style>
""", unsafe_allow_html=True)

# Initialize download trigger state
if 'download_triggered' not in st.session_state:
    st.session_state.download_triggered = False

# Main content based on active tab
if not filtered_df.empty:
    if st.session_state.active_tab == 'overview':
        # AUDIENCE OVERVIEW TAB CONTENT
        # AUDIENCE SIZE AND TREND SECTION - MOVED TO TOP
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Calculate audience size estimation
        if st.session_state.user_login:
            # For User Login: count unique users using correct column name
            if 'user_id' in filtered_df.columns:
                unique_users = filtered_df['user_id'].nunique()
                # Create daily data for prediction
                daily_users = filtered_df.groupby('date')['user_id'].nunique()
            elif 'userid' in filtered_df.columns:
                unique_users = filtered_df['userid'].nunique()
                daily_users = filtered_df.groupby('date')['userid'].nunique()
            else:
                unique_users = len(filtered_df)
                daily_users = filtered_df.groupby('date').size()
        else:
            # For User Non Login: use Total users if available
            if 'Total users' in filtered_df.columns:
                unique_users = filtered_df['Total users'].sum()
                daily_users = filtered_df.groupby('date')['Total users'].sum()
            else:
                unique_users = len(filtered_df)
                daily_users = filtered_df.groupby('date').size()
        
        # Calculate days in current period (30 days)
        current_period_days = 30
        
        # Predict audience for the period
        if len(daily_users) > 0:
            predicted_audience = predict_users_combined(
                daily_users, 
                days_to_predict=current_period_days, 
                user_login=st.session_state.user_login
            )
        else:
            predicted_audience = unique_users
        
        # Format the audience range
        audience_range = format_audience_range(predicted_audience)
        
        # Create two columns for Audience Size and Trend
        size_col, trend_col = st.columns([1, 2])
        
        with size_col:
            # Audience Size Card - show for both User Login and User Non Login
            st.markdown(f"""
            <div class="audience-size-card">
                <div class="audience-size-title">Audience Size</div>
                <div class="audience-size-subtitle">Estimated Audience Size (30 days)</div>
                <div class="audience-size-value">{audience_range}</div>
                <div class="audience-size-disclaimer">
                    Estimates may vary significantly over time based on your targeting selections and available data.
                    <br><br>
                    {"Reachable Audience is based on audience who have provided valid contact information (email or phone number). This can be downloaded by clicking the button in the top right corner." if st.session_state.user_login else "This audience size is based on the total users data from Google Analytics for the selected filters and time period."}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with trend_col:
            # Trend Chart
            st.markdown("<h3 style='margin: 0 0 10px 0; color: #374151; font-size: 16px;'>Trend: Last 30 Days</h3>", unsafe_allow_html=True)
            trend_fig = create_trend_chart(filtered_df, st.session_state.user_login, days_to_show=30)
            st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")
        
        # Add spacing before other charts
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        
        # Create charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Age chart - now simplified to show only selected audience
            st.markdown("<h3 style='margin: 0 0 10px 0; color: #374151; font-size: 16px;'>Age</h3>", unsafe_allow_html=True)
            age_fig = create_age_chart(filtered_df, st.session_state.user_login)
            st.plotly_chart(age_fig, use_container_width=True, key="age_chart")
        
        with col2:
            # Gender chart with statistics stacked vertically
            st.markdown("<h3 style='margin: 0 0 -30px 0; color: #374151; font-size: 16px;'>Gender</h3>", unsafe_allow_html=True)
            
            if not filtered_df.empty:
                gender_fig, gender_percentages = create_gender_chart(filtered_df, st.session_state.user_login)
                
                # Create custom labels with percentages for Selected audience
                selected_labels_with_pct = []
                for label, pct in zip(gender_percentages.index, gender_percentages):
                    selected_labels_with_pct.append(f"{label.title()}<br>{pct:.1f}%")
                
                # Recreate Selected audience chart with percentages in labels
                selected_gender_fig = go.Figure(data=[go.Pie(
                    labels=selected_labels_with_pct,
                    values=gender_percentages,
                    hole=0.5,
                    marker=dict(
                        colors=['rgba(79, 70, 229, 0.8)', 'rgba(79, 70, 229, 0.5)'],
                        line=dict(color='rgba(255,255,255,0.8)', width=2)
                    ),
                    textinfo='none',
                    hovertemplate='%{label}<extra></extra>',
                    showlegend=True
                )])
                
                selected_gender_fig.update_layout(
                    height=220,
                    margin=dict(l=0, r=100, t=30, b=0),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=0.85,
                        font=dict(size=11)
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(selected_gender_fig, use_container_width=True, key="selected_audience_gender")
        
        # Top Cities and Device Category charts in two columns
        city_col, device_col = st.columns(2)
        
        with city_col:
            # Top Cities chart - simplified to show only selected audience
            st.markdown("<h3 style='margin: 0 0 10px 0; color: #374151; font-size: 16px;'>Top Cities</h3>", unsafe_allow_html=True)
            city_fig = create_city_chart(filtered_df, st.session_state.user_login)
            st.plotly_chart(city_fig, use_container_width=True, key="top_cities_chart")
        
        with device_col:
            # Device Category chart as pie chart
            st.markdown("<h3 style='margin: 0 0 -30px 0; color: #374151; font-size: 16px;'>Device Category</h3>", unsafe_allow_html=True)
            device_fig = create_device_chart(filtered_df, st.session_state.user_login)
            st.plotly_chart(device_fig, use_container_width=True, key="device_category_chart")
        
        # Additional charts if there's kanal data
        if 'kanal_group' in filtered_df.columns and not filtered_df['kanal_group'].isna().all():
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            
            # Create kanal chart (exclude "Other" category)
            if st.session_state.user_login:
                kanal_data = filtered_df[filtered_df['kanal_group'] != 'Other']['kanal_group'].value_counts()
            else:
                # For User Non Login, use Total users for weighting
                if 'Total users' in filtered_df.columns:
                    kanal_data = filtered_df[filtered_df['kanal_group'] != 'Other'].groupby('kanal_group')['Total users'].sum()
                else:
                    kanal_data = filtered_df[filtered_df['kanal_group'] != 'Other']['kanal_group'].value_counts()
            
            # Only show chart if there's data after excluding "Other"
            if len(kanal_data) > 0:
                # Sort data from largest to smallest
                kanal_data = kanal_data.sort_values(ascending=False)
                
                # Calculate percentages
                total = kanal_data.sum()
                percentages = (kanal_data / total * 100) if total > 0 else kanal_data * 0
                
                # Create the chart
                kanal_fig = go.Figure()
                
                # Add bars with gradient effect
                for i, (kanal, pct) in enumerate(zip(kanal_data.index, percentages)):
                    kanal_fig.add_trace(go.Bar(
                        x=[kanal],
                        y=[pct],
                        marker=dict(
                            color=f'rgba(245, 158, 11, {1 - i*0.1})',
                            cornerradius=4,
                            line=dict(width=0),
                            pattern=dict(
                                shape='',
                                bgcolor=f'rgba(245, 158, 11, 0.2)',
                                fgcolor=f'rgba(245, 158, 11, {1 - i*0.05})'
                            )
                        ),
                        text=f'{pct:.1f}%',
                        textposition='outside',
                        hovertemplate=f'Kanal: {kanal}<br>Percentage: {pct:.1f}%<extra></extra>',
                        showlegend=False
                    ))
                
                kanal_fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Percentage",
                    showlegend=False,
                    height=280,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        tickangle=45,
                        gridcolor='rgba(0,0,0,0.1)',
                        griddash='dot'
                    ),
                    yaxis=dict(
                        range=[0, max(percentages) * 1.15] if len(percentages) > 0 else [0, 100],
                        gridcolor='rgba(0,0,0,0.1)',
                        griddash='dot'
                    )
                )
                
                st.markdown("<h3 style='margin: 0 0 10px 0; color: #374151; font-size: 16px;'>Kanal Groups</h3>", unsafe_allow_html=True)
                st.plotly_chart(kanal_fig, use_container_width=True, key="kanal_groups_chart")
        
        # Add Allo Wallet Cards section (User Login only)
        if st.session_state.user_login and 'aws' in filtered_df.columns:
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin: 0 0 20px 0; color: #374151; font-size: 16px;'>Allo Wallet Status</h3>", unsafe_allow_html=True)
            
            # Calculate unique users for each Allo Wallet status in last 30 days
            allo_pay_users = 0
            allo_pay_plus_users = 0
            allo_prime_users = 0
            
            if 'user_id' in filtered_df.columns:
                allo_pay_users = filtered_df[filtered_df['aws'] == 'ALLO_PAY']['user_id'].nunique()
                allo_pay_plus_users = filtered_df[filtered_df['aws'] == 'ALLO_PAY_PLUS']['user_id'].nunique()
                allo_prime_users = filtered_df[filtered_df['aws'] == 'ALLO_PRIME']['user_id'].nunique()
            elif 'userid' in filtered_df.columns:
                allo_pay_users = filtered_df[filtered_df['aws'] == 'ALLO_PAY']['userid'].nunique()
                allo_pay_plus_users = filtered_df[filtered_df['aws'] == 'ALLO_PAY_PLUS']['userid'].nunique()
                allo_prime_users = filtered_df[filtered_df['aws'] == 'ALLO_PRIME']['userid'].nunique()
            else:
                # Fallback: count rows
                allo_pay_users = len(filtered_df[filtered_df['aws'] == 'ALLO_PAY'])
                allo_pay_plus_users = len(filtered_df[filtered_df['aws'] == 'ALLO_PAY_PLUS'])
                allo_prime_users = len(filtered_df[filtered_df['aws'] == 'ALLO_PRIME'])
            
            # Create three columns for the Allo Wallet cards
            allo_col1, allo_col2, allo_col3 = st.columns(3)
            
            with allo_col1:
                st.markdown(f"""
                <div class="allo-wallet-card">
                    <div class="allo-wallet-title">Allo Pay</div>
                    <div class="allo-wallet-value">{format_number_display(allo_pay_users)}</div>
                    <div class="allo-wallet-subtitle">audiences in last 30 days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with allo_col2:
                st.markdown(f"""
                <div class="allo-wallet-card">
                    <div class="allo-wallet-title">Allo Pay Plus</div>
                    <div class="allo-wallet-value">{format_number_display(allo_pay_plus_users)}</div>
                    <div class="allo-wallet-subtitle">audiences in last 30 days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with allo_col3:
                st.markdown(f"""
                <div class="allo-wallet-card">
                    <div class="allo-wallet-title">Allo Prime</div>
                    <div class="allo-wallet-value">{format_number_display(allo_prime_users)}</div>
                    <div class="allo-wallet-subtitle">audiences in last 30 days</div>
                </div>
                """, unsafe_allow_html=True)
    
    elif st.session_state.active_tab == 'interests':
        # INTERESTS TAB CONTENT
        # TOP CATEGORIES SECTION
        if 'categoryauto_new_rank1' in filtered_df.columns and not filtered_df['categoryauto_new_rank1'].isna().all():
            st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
            
            # Get categories data from the FILTERED dataset (changed from current_df to filtered_df)
            if st.session_state.user_login:
                # For User Login: count unique users for each category
                all_categories_data = filtered_df.groupby('categoryauto_new_rank1').size().sort_values(ascending=False)
            else:
                # For User Non Login: use Total users for weighting
                if 'Total users' in filtered_df.columns:
                    all_categories_data = filtered_df.groupby('categoryauto_new_rank1')['Total users'].sum().sort_values(ascending=False)
                else:
                    all_categories_data = filtered_df.groupby('categoryauto_new_rank1').size().sort_values(ascending=False)
            
            # Calculate percentages for filtered audience (for chart display)
            total_all = all_categories_data.sum()
            percentages_all = (all_categories_data / total_all * 100) if total_all > 0 else all_categories_data * 0
            
            # Create top categories chart (top 10, sorted)
            top_categories_data = all_categories_data.head(10)
            top_categories_percentages = percentages_all.head(10)
            
            # Only show chart if there's data
            if len(top_categories_data) > 0:
                # Create the chart
                top_categories_fig = go.Figure()
                
                # Add bars with purple gradient effect
                for i, (category, pct) in enumerate(zip(top_categories_data.index, top_categories_percentages.values)):
                    top_categories_fig.add_trace(go.Bar(
                        x=[category],
                        y=[pct],
                        marker=dict(
                            color=f'rgba(139, 92, 246, {1 - i*0.08})',
                            cornerradius=4,
                            line=dict(width=0)
                        ),
                        text=f'{pct:.1f}%',
                        textposition='outside',
                        hovertemplate=f'Category: {category}<br>Percentage: {pct:.1f}%<extra></extra>',
                        showlegend=False
                    ))
                
                top_categories_fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Percentage",
                    showlegend=False,
                    height=280,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        tickangle=45,
                        gridcolor='rgba(0,0,0,0.1)',
                        griddash='dot'
                    ),
                    yaxis=dict(
                        range=[0, max(top_categories_percentages) * 1.15] if len(top_categories_percentages) > 0 else [0, 100],
                        gridcolor='rgba(0,0,0,0.1)',
                        griddash='dot'
                    )
                )
                
                st.markdown("<h3 style='margin: 0 0 10px 0; color: #374151; font-size: 16px;'>Top Categories</h3>", unsafe_allow_html=True)
                st.plotly_chart(top_categories_fig, use_container_width=True, key="top_categories_chart")
        
        # ALL CATEGORIES TABLE SECTION
        if 'categoryauto_new_rank1' in filtered_df.columns and not filtered_df['categoryauto_new_rank1'].isna().all():
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            
            # Get categories data from the FILTERED dataset (changed from current_df to filtered_df)
            if st.session_state.user_login:
                # For User Login: count unique users for each category
                all_categories_data = filtered_df.groupby('categoryauto_new_rank1').size().sort_values(ascending=False)
            else:
                # For User Non Login: use Total users for weighting
                if 'Total users' in filtered_df.columns:
                    all_categories_data = filtered_df.groupby('categoryauto_new_rank1')['Total users'].sum().sort_values(ascending=False)
                else:
                    all_categories_data = filtered_df.groupby('categoryauto_new_rank1').size().sort_values(ascending=False)
            
            # Calculate percentages for filtered audience (for table display)
            total_all = all_categories_data.sum()
            percentages_all = (all_categories_data / total_all * 100) if total_all > 0 else all_categories_data * 0
            
            # TOP 10 CATEGORIES LAYOUT - MOVED ABOVE TABLE
            st.markdown("<h4 style='margin: 0 0 20px 0; color: #374151; font-size: 16px; font-weight: 600;'>Top 10 categories</h4>", unsafe_allow_html=True)
            
            # Get top 10 categories for the layout
            top_10_categories = all_categories_data.head(10)
            
            # Only show the top 10 layout if there are categories
            if len(top_10_categories) > 0:
                # Create two rows of 5 containers each
                row1_cols = st.columns(5)
                row2_cols = st.columns(5)
                
                # First row (categories 1-5)
                for i, (category, user_count) in enumerate(zip(top_10_categories.index[:5], top_10_categories.values[:5])):
                    with row1_cols[i]:
                        st.markdown(f"""
                        <div style='
                            background-color: #f8fafc;
                            border: 1px solid #e2e8f0;
                            border-radius: 8px;
                            padding: 16px 12px;
                            text-align: left;
                            min-height: 80px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            margin-bottom: 12px;
                        '>
                            <div style='
                                font-weight: 600;
                                color: #4f46e5;
                                font-size: 18px;
                                margin-bottom: 4px;
                            '>{i+1}</div>
                            <div style='
                                font-weight: 500;
                                color: #374151;
                                font-size: 14px;
                                line-height: 1.3;
                                margin-bottom: 4px;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                display: -webkit-box;
                                -webkit-line-clamp: 2;
                                -webkit-box-orient: vertical;
                            '>{category}</div>
                            <div style='
                                font-size: 12px;
                                color: #6b7280;
                                font-weight: 500;
                            '>{user_count:,} Users</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Second row (categories 6-10) - only if there are more than 5 categories
                if len(top_10_categories) > 5:
                    for i, (category, user_count) in enumerate(zip(top_10_categories.index[5:10], top_10_categories.values[5:10]), 5):
                        with row2_cols[i-5]:
                            st.markdown(f"""
                            <div style='
                                background-color: #f8fafc;
                                border: 1px solid #e2e8f0;
                                border-radius: 8px;
                                padding: 16px 12px;
                                text-align: left;
                                min-height: 80px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                                margin-bottom: 12px;
                            '>
                                <div style='
                                    font-weight: 600;
                                    color: #4f46e5;
                                    font-size: 18px;
                                    margin-bottom: 4px;
                                '>{i+1}</div>
                                <div style='
                                    font-weight: 500;
                                    color: #374151;
                                    font-size: 14px;
                                    line-height: 1.3;
                                    margin-bottom: 4px;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                    display: -webkit-box;
                                    -webkit-line-clamp: 2;
                                    -webkit-box-orient: vertical;
                                '>{category}</div>
                                <div style='
                                    font-size: 12px;
                                    color: #6b7280;
                                    font-weight: 500;
                                '>{user_count:,} Users</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Add spacing before the table
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin: 0 0 15px 0; color: #374151; font-size: 16px;'>All Categories</h3>", unsafe_allow_html=True)
            
            # Create a dataframe for display
            table_data = []
            for i, (category, count) in enumerate(all_categories_data.items(), 1):
                percentage = percentages_all[category]
                table_data.append({
                    'Rank': i,
                    'Categories': category,
                    'All Audience': percentage  # Keep as numeric for bar chart
                })
            
            # Create DataFrame and display with Streamlit
            df_table = pd.DataFrame(table_data)
            
            # Display the table with custom styling and bar chart
            st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank",
                        width="small",
                    ),
                    "Categories": st.column_config.TextColumn(
                        "Categories",
                        width="large",
                    ),
                    "All Audience": st.column_config.ProgressColumn(
                        "All Audience",
                        width="medium",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    ),
                }
            )
        else:
            # Show message when no category data is available after filtering
            st.info("No category data available for the selected filters.")

else:
    st.info("No data available for the selected filters and date range.")
