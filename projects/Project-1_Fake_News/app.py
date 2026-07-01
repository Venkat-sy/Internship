import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "fake_news_train.csv")
    if not os.path.exists(path):
        return None, None
    data = pd.read_csv(path)
    
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
        
    data['clean_text'] = data['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(3, 5))
    X_vec = vectorizer.fit_transform(data['clean_text'])
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, data['label'])
    return vectorizer, model

vec, model = load_model()

st.title("📰 Fake News Detector")
st.markdown("This AI model uses Natural Language Processing and Logistic Regression to classify news articles as **Real** or **Fake**.")

if model is None:
    st.error("Model dataset not found. Please ensure 'fake_news_train.csv' exists.")
else:
    user_input = st.text_area("Paste a news article here:", height=200)
    if st.button("Analyze News", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            clean_t = re.sub(r'\s+', ' ', re.sub(r'\W', ' ', user_input.lower())).strip()
            vec_input = vec.transform([clean_t])
            pred = model.predict(vec_input)[0]
            prob = model.predict_proba(vec_input)[0]
            
            if pred == 1:
                st.success(f"✅ **REAL NEWS** (Confidence: {prob[1]:.2%})")
            else:
                st.error(f"🚨 **FAKE NEWS** (Confidence: {prob[0]:.2%})")
