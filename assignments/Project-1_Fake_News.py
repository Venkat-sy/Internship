import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_dummy_data(filepath="fake_news_train.csv"):
    print(f"Dataset not found. Creating a synthetic dataset at {filepath} for demonstration...")
    data = {
        'text': [
            "The quick brown fox jumps over the lazy dog.",
            "Fake news is spreading fast on social media.",
            "Scientists discover a new species of frog in the Amazon.",
            "Aliens have landed in New York City and are drinking coffee.",
            "The stock market reached an all-time high today.",
            "You won't believe this one weird trick to lose weight!",
            "Government passes new legislation to improve infrastructure.",
            "Secret society controls the world's weather using satellites.",
            "Local team wins the championship after a stunning comeback.",
            "Drink bleach to cure all diseases, doctor claims."
        ] * 50, # 500 samples
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 50  # 1 for Real, 0 for Fake
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return df

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # remove extra whitespace
    return text

def main():
    dataset_path = "fake_news_train.csv"
    if not os.path.exists(dataset_path):
        create_dummy_data(dataset_path)
        
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)
    
    # Ensure columns exist
    if 'text' not in data.columns or 'label' not in data.columns:
        print(f"Error: Dataset {dataset_path} must contain 'text' and 'label' columns.")
        return

    print("Cleaning text...")
    data['clean_text'] = data['text'].apply(clean_text)
    
    X = data['clean_text']
    y = data['label']
    
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    models = {
        "KNN (Non-Parametric)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression (Parametric)": LogisticRegression(max_iter=1000),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Simple Neural Network (Deep Learning)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    
    print("\n--- Model Training & Evaluation ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
