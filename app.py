import streamlit as st

st.set_page_config(page_title="AI Internship Projects", page_icon="🤖", layout="wide")

pages = {
    "AI Projects": [
        st.Page("projects/Project-1_Fake_News/app.py", title="Fake News Detector", icon="📰"),
        st.Page("projects/Project-2_Phishing_Email/app.py", title="Phishing Email Detector", icon="📧"),
    ]
}

pg = st.navigation(pages)
pg.run()
