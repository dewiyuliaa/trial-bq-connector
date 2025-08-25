try:
    secrets = st.secrets["gcp_service_account"]
    st.success("Secrets loaded successfully!")
except Exception as e:
    st.error(f"Error loading secrets: {e}")
