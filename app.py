import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Internship Projects", page_icon="🤖", layout="wide")

# --- CACHE DATA & MODELS ---
@st.cache_resource
def load_fake_news_model():
    path = "projects/Project-1_Fake_News/fake_news_train.csv"
    if not os.path.exists(path):
        return None, None
    data = pd.read_csv(path)
    
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
        
    data['clean_text'] = data['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(data['clean_text'])
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, data['label'])
    return vectorizer, model

@st.cache_resource
def load_phishing_model():
    path = "projects/Project-2_Phishing_Email/phishing_emails.csv"
    if not os.path.exists(path):
        return None, None
    data = pd.read_csv(path)
    
    def clean_email(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\W', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
        
    data['clean_text'] = data['email_text'].apply(clean_email)
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_vec = vectorizer.fit_transform(data['clean_text'])
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, data['label'])
    return vectorizer, model

# Load models
fn_vec, fn_model = load_fake_news_model()
ph_vec, ph_model = load_phishing_model()

# --- SIDEBAR ---
st.sidebar.title("AI Projects")
st.sidebar.markdown("Navigate between the two AI internship projects.")
page = st.sidebar.radio("Select a Project:", ["Fake News Detection", "Phishing Email Detection"])

# --- PAGE 1: FAKE NEWS ---
if page == "Fake News Detection":
    st.title("📰 Fake News Detector")
    st.markdown("This AI model uses Natural Language Processing and Logistic Regression to classify news articles as **Real** or **Fake**.")
    
    if fn_model is None:
        st.error("Model dataset not found. Please ensure 'projects/Project-1_Fake_News/fake_news_train.csv' exists.")
    else:
        user_input = st.text_area("Paste a news article here:", height=200, placeholder="e.g. The quick brown fox jumps over the lazy dog.")
        
        if st.button("Analyze News", type="primary"):
            if user_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                # Clean and predict
                clean_t = re.sub(r'\s+', ' ', re.sub(r'\W', ' ', user_input.lower())).strip()
                vec_input = fn_vec.transform([clean_t])
                pred = fn_model.predict(vec_input)[0]
                prob = fn_model.predict_proba(vec_input)[0]
                
                if pred == 1:
                    st.success(f"✅ **REAL NEWS** (Confidence: {prob[1]:.2%})")
                else:
                    st.error(f"🚨 **FAKE NEWS** (Confidence: {prob[0]:.2%})")

# --- PAGE 2: PHISHING EMAIL ---
elif page == "Phishing Email Detection":
    st.title("📧 Phishing Email Detector")
    st.markdown("This AI model uses NLP and Logistic Regression to analyze email text and detect **Phishing** attempts.")
    
    if ph_model is None:
        st.error("Model dataset not found. Please ensure 'projects/Project-2_Phishing_Email/phishing_emails.csv' exists.")
    else:
        user_input = st.text_area("Paste the email content here:", height=200, placeholder="e.g. URGENT: Your account has been compromised! Click here to verify...")
        
        if st.button("Analyze Email", type="primary"):
            if user_input.strip() == "":
                st.warning("Please enter an email.")
            else:
                # Clean and predict
                clean_t = re.sub(r'\s+', ' ', re.sub(r'\W', ' ', re.sub(r'<[^>]+>', ' ', user_input.lower()))).strip()
                vec_input = ph_vec.transform([clean_t])
                pred = ph_model.predict(vec_input)[0]
                prob = ph_model.predict_proba(vec_input)[0]
                
                if pred == 1:
                    st.error(f"🚨 **PHISHING DETECTED** (Confidence: {prob[1]:.2%})")
                else:
                    st.success(f"✅ **LEGITIMATE EMAIL** (Confidence: {prob[0]:.2%})")
