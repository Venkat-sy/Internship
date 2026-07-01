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

import random

def create_dummy_data(filepath="fake_news_train.csv"):
    print(f"Dataset not found. Creating an enhanced synthetic dataset at {filepath}...")
    
    real_subjects = ["The government", "Scientists", "A local committee", "The president", "Researchers", "The UN", "The central bank", "A tech company"]
    real_verbs = ["announced", "discovered", "reported", "approved", "published", "released", "discussed", "launched"]
    real_objects = ["a new policy", "a breakthrough", "the budget", "a study", "the findings", "an initiative", "economic data", "a software update"]
    
    fake_subjects = ["Aliens", "A secret society", "The Illuminati", "Shadow government", "A weird trick", "Time travelers", "Lizard people"]
    fake_verbs = ["hide", "control", "ban", "cure", "reveal", "manipulate", "destroyed", "invented"]
    fake_objects = ["the truth", "your mind", "all diseases", "the flat earth", "a miracle", "the weather", "the moon landing"]
    
    data_text = []
    data_label = []
    
    # Generate 500 Real News
    for _ in range(500):
        sentence = f"{random.choice(real_subjects)} {random.choice(real_verbs)} {random.choice(real_objects)}."
        data_text.append(sentence)
        data_label.append(1)
        
    # Generate 500 Fake News
    for _ in range(500):
        sentence = f"SHOCKING! {random.choice(fake_subjects)} {random.choice(fake_verbs)} {random.choice(fake_objects)} and they don't want you to know!"
        data_text.append(sentence)
        data_label.append(0)
        
    df = pd.DataFrame({'text': data_text, 'label': data_label})
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
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
    # Force recreation of the enhanced dataset
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
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
    
    print("Vectorizing text (TF-IDF with Char N-Grams)...")
    # Using character n-grams makes the model incredibly robust to unseen words!
    vectorizer = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(3, 5))
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
