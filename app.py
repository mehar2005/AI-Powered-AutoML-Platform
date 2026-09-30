import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title = "AI Powered AutoML Platform",
    page_icon = '🤖',
    initial_sidebar_state = 'auto',
    layout = 'wide'
)

st.title("🤖 AI Powered AutoML Platform", text_alignment = 'center')
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.subheader("Bridging the Gap Between Data and Machine Learning", text_alignment = 'center')
with col2:
    st.caption(f"Last Updated: {datetime.today().strftime("%I:%M:%S %p")}")