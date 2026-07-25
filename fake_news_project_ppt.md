# PPT Structure: AI-Powered Fake News Detection

---

## Slide 1: Title
**Title:** AI-Powered Fake News Detection Using Text Classification
**Subtitle:** A Machine Learning Approach to Combating Misinformation
**Presenter:** Venkatesh Hebbar
**Roll No:** 595766
**College:** NMAM Institute of Technology

---

## Slide 2: Introduction & Problem Statement
**Introduction:**
- Fake news spreads rapidly on social media, eroding public trust and causing societal harm.
- Manual fact-checking is slow and cannot scale to the volume of online content.

**Problem Statement:**
- To design an automated, AI-driven system capable of classifying news articles as "Fake" or "Real" based solely on their textual content.
- Need for a lightweight, accurate, and scalable solution using Natural Language Processing (NLP).

---

## Slide 3: Methodology
**System Workflow:**
1. **Dataset Collection:** Curated dataset of 1,000 balanced Fake and Real news records.
2. **Data Preprocessing:** Lowercasing, removing HTML, punctuation, URLs, emojis, and stopwords.
3. **Feature Extraction:** TF-IDF Vectorization combined with Character N-Grams (3-5 chars) to handle out-of-vocabulary words.
4. **Train/Test Split:** 80% Training Data, 20% Testing Data.
5. **Model Training:** Fitting classifiers to the feature vectors.
6. **Prediction:** Deploying the best model in an interactive web interface.

*(Include the Workflow Diagram from the report here)*

---

## Slide 4: Algorithms Used
**1. Logistic Regression (Parametric)**
- Fast, highly interpretable, performs well on sparse text data.
**2. Random Forest (Ensemble)**
- Robust to noise, captures non-linear patterns, provides feature importance.
**3. K-Nearest Neighbors (Instance-based)**
- Lazy learning, classifies based on proximity in feature space.
**4. Neural Network / Multi-Layer Perceptron (Deep Learning)**
- Models complex interactions between words using hidden layers and non-linear activations.

---

## Slide 5: Results & Performance Comparison
**Evaluation Metrics (Test Set):**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN | 100% | 100% | 100% | 100% |
| Logistic Regression | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% |
| Neural Network | 100% | 100% | 100% | 100% |

**Key Finding:** 
- The use of Character N-Grams made all models extremely robust. 
- Logistic Regression was selected for the final deployment due to its perfect accuracy combined with blazing-fast inference time and high interpretability.

*(Include Confusion Matrix or Bar Chart here)*

---

## Slide 6: Demo
**Interactive Web Application (Streamlit)**
- **Input:** User pastes a news headline or article.
- **Process:** The text undergoes real-time cleaning and TF-IDF transformation.
- **Output:** The model predicts "FAKE NEWS" or "REAL NEWS" with a confidence score.

*(Insert Screenshots of the Streamlit App: One showing a FAKE prediction, one showing a REAL prediction)*

---

## Slide 7: Conclusion & Future Scope
**Conclusion:**
- Successfully built an end-to-end NLP pipeline that accurately classifies text.
- Character N-Grams proved highly effective at handling novel text structures and typos.
- The deployed web module proves the feasibility of real-time AI fact-checking.

**Future Scope:**
- **Contextual Embeddings:** Upgrade to Transformer models (BERT, RoBERTa) for deeper semantic understanding.
- **Live Fact-Checking:** Integrate with Knowledge Graphs or search APIs to verify actual facts, not just textual style.
- **Browser Extension:** Deploy as a Chrome extension for real-time alerts on social media.
