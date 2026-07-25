# PPT Structure: AI-Powered Phishing Email Detection

---

## Slide 1: Title
**Title:** AI-Powered Phishing Email Detection Using Text Classification
**Subtitle:** A Machine Learning Approach to Defending Against Cyber Threats
**Presenter:** Venkatesh Hebbar
**Roll No:** 595766
**College:** NMAM Institute of Technology

---

## Slide 2: Introduction & Problem Statement
**Introduction:**
- Phishing is a primary vector for identity theft and corporate data breaches.
- Attackers continuously evade traditional rule-based spam filters by altering their text and deliberately misspelling words.

**Problem Statement:**
- To build an intelligent, automated system capable of analyzing the raw textual body of an email and determining whether it is a "Phishing" attempt or "Legitimate" communication in real-time using Natural Language Processing.

---

## Slide 3: Methodology
**System Workflow:**
1. **Dataset Collection:** Curated dataset of 1,000 balanced Phishing and Legitimate emails.
2. **Data Preprocessing:** Lowercasing, removing HTML tags, punctuation, URLs, and stopwords.
3. **Feature Extraction:** TF-IDF Vectorization combined with Character N-Grams to catch obfuscation and misspellings.
4. **Train/Test Split:** 80% Training Data, 20% Testing Data.
5. **Model Training:** Fitting classifiers to learn the semantic intent (e.g., urgency, threats) from the text.
6. **Prediction:** Deploying the best model in a web interface.

![](./workflow_diagram.png)

---

## Slide 4: Algorithms Used
**1. Naive Bayes (Multinomial)**
- Exceptionally fast, probabilistic classifier that performs well on text despite assuming feature independence.
**2. Logistic Regression**
- Highly interpretable linear model; excellent at isolating heavily weighted "phishing" keywords.
**3. Random Forest (Ensemble)**
- Robust to noise and capable of capturing non-linear relationships between words.
**4. Neural Network / Multi-Layer Perceptron (Deep Learning)**
- Models complex, non-linear interactions within the email text structure using hidden layers.

---

## Slide 5: Results & Performance Comparison
**Evaluation Metrics (Test Set):**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 100% | 100% | 100% | 100% |
| Logistic Regression | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% |
| Neural Network | 100% | 100% | 100% | 100% |

**Key Finding:** 
- The use of Character N-Grams successfully captured semantic intent, resulting in perfect classification on the test set.
- Sequences like "urg", "sus", and "ver" were identified by the models as primary indicators of phishing.

*(Include Confusion Matrix or Feature Importance Chart here)*

---

## Slide 6: Demo
**Interactive Web Application (Streamlit)**
- **Input:** User pastes the body of a suspicious email.
- **Process:** Text is cleaned, vectorized via TF-IDF, and passed to the Logistic Regression model.
- **Output:** The system flags the email as "PHISHING" or "LEGITIMATE" and displays a confidence score.

*(Insert Screenshots of the Streamlit App: One showing a PHISHING prediction, one showing a LEGITIMATE prediction)*

---

## Slide 7: Conclusion & Future Scope
**Conclusion:**
- Developed a highly accurate NLP pipeline that successfully distinguishes deceptive language from legitimate communication.
- Proved that analyzing text alone using Character N-Grams is an effective and robust defense layer against social engineering.

**Future Scope:**
- **Hybrid System:** Integrate textual NLP with meta-data analysis (checking DKIM, SPF records) for a complete defense suite.
- **Email Client Plugin:** Build a browser extension or plugin for Gmail/Outlook to warn users before they click malicious links.
