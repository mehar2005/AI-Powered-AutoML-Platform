import streamlit as st
import pandas as pd
from datetime import datetime

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = pd.DataFrame({})
if 'time_updated' not in st.session_state:
    st.session_state.time_updated = datetime.today().strftime("%I:%M:%S %p")

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

from file_uploaderOverviewCleaning import file_uploader_show
from eda import eda_show
from feature_engineering import fe_show
from ml import ml_show
from predictions import predictions_show

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
    st.caption(f"Last Updated: {st.session_state.get('time_updated')}")

st.divider()

file, eda, fe, ml, pr = st.tabs(['File Upload & Data Cleaning', "Exploratory Data Analysis", 'Feature Engineering', "Machine Learning", "Predictions"])

with file:
    file_uploader_show()

with eda:
    eda_show()

with fe:
    fe_show()

st.divider()

with ml:
    ml_show()

with pr:
    predictions_show()

st.divider()

with st.container(border = True):
    col1, col2 = st.columns([13, 0.7])
    with col1:
        st.subheader("Want to Reset Everything?")
    with col2:
        if st.button(
            label = 'Reset', 
            type = 'primary',
            key = 'reset_session',
            disabled = st.session_state.get('reset_session') is None
            ):
            reset_session()
            st.rerun()