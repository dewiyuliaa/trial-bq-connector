import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="User Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_bigquery():
    """Load data from BigQuery"""
    try:
        # Create credentials from Streamlit secrets
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        
        # Initialize BigQuery client
        client = bigquery.Client(credentials=credentials)
        
        # Query to get unique users per date for the last 30 days
        query = """
        SELECT 
            date,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(*) as total_pageviews
        FROM `pm-data-217109.dewi_audience_poc.detik_lookup`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            AND user_id IS NOT NULL
        GROUP BY date
        ORDER BY date ASC
        """
        
        # Execute query
        df = client.query(query).to_dataframe()
        
        # Convert date to datetime if it's not already
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        st.error(f"Error connecting to BigQuery: {e}")
        return None

def create_line_chart(df):
    """Create line chart for unique users over time"""
    fig = go.Figure()
    
    # Add line for unique users
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['unique_users'],
        mode='lines+markers',
        name='Unique Users',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='Date: %{x}<br>Unique Users: %{y:,}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Unique Users Per Day (Last 30 Days)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title="Date",
        yaxis_title="Unique Users",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        )
    )
    
    return fig

def create_dual_axis_chart(df):
    """Create chart with dual y-axis for users and pageviews"""
    fig = go.Figure()
    
    # Add unique users (left y-axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['unique_users'],
        mode='lines+markers',
        name='Unique Users',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        yaxis='y',
        hovertemplate='Date: %{x}<br>Unique Users: %{y:,}<extra></extra>'
    ))
    
    # Add total pageviews (right y-axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_pageviews'],
        mode='lines+markers',
        name='Total Pageviews',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8),
        yaxis='y2',
        hovertemplate='Date: %{x}<br>Total Pageviews: %{y:,}<extra></extra>'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title={
            'text': 'Unique Users vs Total Pageviews (Last 30 Days)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title="Date",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        # Primary y-axis (left) for Unique Users
        yaxis=dict(
            title="Unique Users",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        ),
        # Secondary y-axis (right) for Pageviews
        yaxis2=dict(
            title="Total Pageviews",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dot'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main app
def main():
    st.title("📊 User Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data from BigQuery..."):
        df = load_data_from_bigquery()
    
    if df is not None and not df.empty:
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Days", 
                len(df)
            )
        
        with col2:
            st.metric(
                "Avg Daily Users", 
                f"{df['unique_users'].mean():.0f}"
            )
        
        with col3:
            st.metric(
                "Max Daily Users", 
                f"{df['unique_users'].max():,}"
            )
        
        with col4:
            st.metric(
                "Total Unique Users", 
                f"{df['unique_users'].sum():,}"
            )
        
        st.markdown("---")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["📈 Unique Users", "📊 Users vs Pageviews"])
        
        with tab1:
            # Simple line chart for unique users
            fig1 = create_line_chart(df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            # Dual axis chart
            fig2 = create_dual_axis_chart(df)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Data table
        st.markdown("### 📋 Raw Data")
        st.dataframe(
            df.style.format({
                'unique_users': '{:,}',
                'total_pageviews': '{:,}'
            }),
            use_container_width=True
        )
        
    elif df is not None and df.empty:
        st.warning("No data found in the specified date range.")
    else:
        st.error("Failed to load data. Please check your BigQuery connection and credentials.")

# Test connection function
def test_connection():
    st.sidebar.markdown("### 🔧 Connection Test")
    if st.sidebar.button("Test BigQuery Connection"):
        try:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            client = bigquery.Client(credentials=credentials)
            
            # Simple test query
            test_query = "SELECT 1 as test"
            result = client.query(test_query).to_dataframe()
            
            st.sidebar.success("✅ BigQuery connection successful!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_connection()
    main()
