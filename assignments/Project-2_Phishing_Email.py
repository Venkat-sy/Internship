import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_dummy_data(filepath="phishing_emails.csv"):
    print(f"Dataset not found. Creating a synthetic dataset at {filepath} for demonstration...")
    data = {
        'email_text': [
            "Dear Customer, your account has been suspended. Please click here to verify.",
            "Hey John, are we still on for lunch tomorrow?",
            "URGENT: You have won a $1000 Walmart gift card! Claim now by replying.",
            "Please find attached the meeting notes from yesterday's sync.",
            "Your password will expire in 24 hours. Click this link to reset it.",
            "Don't forget to submit your timesheet by Friday.",
            "Exclusive offer: Get 80% off on all luxury watches!",
            "Hi team, just a reminder about the all-hands meeting at 2 PM.",
            "Verify your bank details immediately to avoid account closure.",
            "Happy birthday! Hope you have a wonderful day."
        ] * 50, # 500 samples
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 50  # 1 for Phishing, 0 for Legitimate
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return df

def clean_email(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text) # remove HTML tags
    text = re.sub(r'\W', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    dataset_path = "phishing_emails.csv"
    if not os.path.exists(dataset_path):
        create_dummy_data(dataset_path)
        
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)
    
    if 'email_text' not in data.columns or 'label' not in data.columns:
        print(f"Error: Dataset {dataset_path} must contain 'email_text' and 'label' columns.")
        return

    print("Cleaning emails...")
    data['clean_text'] = data['email_text'].apply(clean_email)
    
    X = data['clean_text']
    y = data['label']
    
    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Simple Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    
    print("\n--- Model Training & Evaluation ---")
    
    best_model_name = ""
    best_model = None
    best_acc = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))
        
        if acc >= best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            
        # Plot confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Phishing'], 
                    yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()
        
    print(f"\nSaved confusion matrix plots to disk.")
    
    # Feature Importance for Random Forest
    rf_model = models["Random Forest"]
    if hasattr(rf_model, 'feature_importances_'):
        print("\nPlotting Feature Importance for Random Forest...")
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] # Top 20 features
        
        # Get feature names safely
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            feature_names = vectorizer.get_feature_names()
            
        plt.figure(figsize=(10,6))
        plt.title("Top 20 Feature Importances (Random Forest)")
        plt.bar(range(20), importances[indices], align="center")
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png')
        plt.close()
        print("Saved Random Forest feature importance plot to disk.")

if __name__ == "__main__":
    main()
