import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "phishing_emails.csv")
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
    vectorizer = TfidfVectorizer(max_features=3000, analyzer='char_wb', ngram_range=(3, 5))
    X_vec = vectorizer.fit_transform(data['clean_text'])
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, data['label'])
    return vectorizer, model

vec, model = load_model()

st.title("📧 Phishing Email Detector")
st.markdown("This AI model uses NLP and Logistic Regression to analyze email text and detect **Phishing** attempts.")

if model is None:
    st.error("Model dataset not found. Please ensure 'phishing_emails.csv' exists.")
else:
    user_input = st.text_area("Paste the email content here:", height=200)
    if st.button("Analyze Email", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter an email.")
        else:
            clean_t = re.sub(r'\s+', ' ', re.sub(r'\W', ' ', re.sub(r'<[^>]+>', ' ', user_input.lower()))).strip()
            vec_input = vec.transform([clean_t])
            pred = model.predict(vec_input)[0]
            prob = model.predict_proba(vec_input)[0]
            
            if pred == 1:
                st.error(f"🚨 **PHISHING DETECTED** (Confidence: {prob[1]:.2%})")
            else:
                st.success(f"✅ **LEGITIMATE EMAIL** (Confidence: {prob[0]:.2%})")
