import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

st.title("BigQuery Connection Test")

if st.button("Test BigQuery Connection"):
    try:
        # Load credentials
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        st.success("✅ Credentials loaded")
        
        # Create client
        client = bigquery.Client(credentials=credentials)
        st.success("✅ BigQuery client created")
        
        # Test simple query first
        query = "SELECT 1 as test_value"
        result = client.query(query).to_dataframe()
        st.success("✅ Basic query successful")
        
        # Test your actual table
        query2 = """
        SELECT COUNT(*) as row_count 
        FROM `pm-data-217109.dewi_audience_poc.detik_lookup` 
        LIMIT 1
        """
        result2 = client.query(query2).to_dataframe()
        st.success("✅ Table access successful!")
        st.write("Row count in your table:", result2.iloc[0]['row_count'])
        
    except Exception as e:
        st.error(f"❌ BigQuery Error: {e}")
        if "Access Denied" in str(e):
            st.warning("You need to add your service account to the table permissions.")
