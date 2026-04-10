from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent / "streamlit"

APP_STYLES = """
<style>
    .block-container,
    section.main > div {
        max-width: 100%;
        padding-top: 1.25rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #dbe4ee;
        border-radius: 0.9rem;
        padding: 0.8rem 1rem;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    div[data-testid="stTabs"] [data-baseweb="tab"] {
        height: 2.6rem;
        white-space: nowrap;
        border-radius: 999px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
</style>
"""

st.set_page_config(
    page_title="Análisis Exploratorio (EDA)",
    layout="wide",
)

st.markdown(APP_STYLES, unsafe_allow_html=True)

navigation = st.navigation(
    [
        st.Page(str(APP_DIR / "Home.py"), title="Inicio", default=True),
        st.Page(
            str(APP_DIR / "pages" / "1_Portfolio_Dashboard.py"),
            title="Análisis Exploratorio (EDA)",
        ),
    ]
)

navigation.run()
