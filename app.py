import streamlit as st
from datetime import datetime
import file_uploaderOverviewCleaning as file_uploaderOverviewCleaning
from eda import eda_show

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

st.divider()

file = file_uploaderOverviewCleaning.file_uploader_show()

st.divider()

eda_show()

st.title('Isto Agge Rabb Bharose 🙏', text_alignment = 'center')