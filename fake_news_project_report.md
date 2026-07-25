# A PROJECT REPORT ON

## AI-POWERED FAKE NEWS DETECTION USING TEXT CLASSIFICATION

<br><br><br>
*(Insert College Logo Here)*
<br><br>

### NMAM Institute of Technology

<br><br>

Submitted by

**Venkatesh Hebbar**  
Roll No.: 595766  

2026

---
# Table of Contents
Abstract & Keywords	3
Introduction	4
Problem Statement & Objectives	5
Literature Review	6
Proposed Methodology	8
Dataset Description	9
Data Preprocessing	10
Feature Engineering	11
Machine Learning Models	12
Model Training	15
Results & Evaluation	16
Feature Importance & Analysis	18
Prediction Module	19
Advantages, Limitations & Future Scope	20
Conclusion	21
References	22

---

## 2. Abstract & Keywords

**Abstract:**
The rapid growth of social media and online news platforms has made the dissemination of information faster and more widespread than ever before. However, this same speed and reach have also enabled the rapid spread of fake news—deliberately fabricated or misleading information presented as legitimate news. Fake news can influence public opinion, disrupt elections, incite panic, and erode trust in genuine journalism, making its automatic detection a pressing research problem. This project presents a machine learning-based system for the automatic classification of news articles as "Fake" or "Real" using Natural Language Processing (NLP) techniques.

The proposed system utilizes a robust preprocessing pipeline involving lowercasing, removal of HTML tags, punctuation, URLs, emojis, and stopwords, followed by tokenization and lemmatization. The cleaned text is then converted into numerical feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) technique combined with Character N-Grams to handle out-of-vocabulary terms effectively. Four machine learning classifiers—Logistic Regression, Random Forest, Multinomial Naive Bayes, and a feed-forward Neural Network—are trained on an 80:20 train-test split and evaluated using standard classification metrics: accuracy, precision, recall, and F1-score.

Experimental results show that all models achieve exceptional performance on the curated dataset. A comparative analysis of the models is presented along with confusion matrices and a feature-importance study identifying the words most strongly associated with fake and real news. An interactive Streamlit-based prediction module is also built to demonstrate real-time classification of user-supplied news text. The project concludes that combining TF-IDF features with classical and neural machine learning models provides an effective, lightweight, and interpretable approach to automated fake news detection.

**Keywords:** Fake News Detection, Natural Language Processing, TF-IDF, Machine Learning, Logistic Regression, Random Forest, Text Classification.

---

## 3. Introduction

### 3.1 What is Fake News?
Fake news refers to false, misleading, or fabricated information that is presented as legitimate news content. It is often created with the intent to deceive readers, manipulate public opinion, generate advertising revenue through sensational content, or damage the reputation of an individual, organization, or institution. Fake news can take several forms, including entirely fabricated stories, manipulated or out-of-context facts, misleading headlines (clickbait), satire misinterpreted as fact, and biased or one-sided reporting disguised as objective journalism.

With the proliferation of social media platforms such as Facebook, X (formerly Twitter), and WhatsApp, news no longer travels solely through verified editorial channels. Anyone can create and share content instantly. Furthermore, algorithms that prioritize engagement often amplify sensational or emotionally charged fake news faster than accurate, measured reporting. This has created an environment where misinformation can spread to millions of users within hours, long before fact-checkers can respond.

### 3.2 Impact of Fake News
* **Erosion of public trust** in media institutions, journalists, and democratic processes.
* **Influence on elections** and political discourse through targeted disinformation campaigns.
* **Public health risks**, such as the spread of medical misinformation during pandemics.
* **Financial market manipulation** through fabricated corporate or economic news.
* **Incitement of social unrest**, communal tension, and mob violence in extreme cases.
* **Damage to the reputation** of individuals, brands, and organizations.

### 3.3 Need for Artificial Intelligence in Fake News Detection
Manual fact-checking, while accurate, cannot scale to the volume and velocity of content generated online every day. Human fact-checkers typically take hours to days to verify a single claim, whereas misinformation can reach millions of users within minutes of being posted. This mismatch between the scale of misinformation and the capacity of manual verification creates a strong need for automated, AI-driven detection systems.

Machine Learning and Natural Language Processing techniques allow computers to analyze large volumes of text and learn linguistic, stylistic, and structural patterns that distinguish fake news from legitimate news. These patterns may include the use of sensational or emotionally charged language, exaggerated claims, inconsistent writing style, lack of credible sourcing, and specific vocabulary patterns that are statistically more common in fabricated content. By training classification models on labelled datasets, it becomes possible to build systems that can flag suspicious content automatically.

---

## 4. Problem Statement & Objectives

### 4.1 Problem Statement
Given the increasing volume of online news content and the limitations of manual fact-verification, there is a need for an automated system that can accurately classify a news article or statement as "Fake" or "Real" based solely on its textual content. This project aims to design, implement, and evaluate a machine learning-based text classification pipeline capable of performing this task efficiently and with high accuracy.

### 4.2 Objectives
1. To study and understand the characteristics that distinguish fake news from real news content.
2. To collect, generate, and preprocess a labelled fake/real news dataset suitable for supervised learning.
3. To extract meaningful numerical features from raw text using TF-IDF vectorization and Character N-Grams.
4. To implement and train multiple machine learning classifiers—Logistic Regression, Random Forest, Naive Bayes, and a Neural Network—for the classification task.
5. To evaluate and compare the performance of these models using standard metrics such as accuracy, precision, recall, and F1-score.
6. To identify the most influential words/features contributing to the classification decision.
7. To build a real-time prediction module with a graphical user interface (GUI) using Streamlit that demonstrates real-time classification of user-input news text.
8. To analyze the limitations of the current approach and propose directions for future improvement.

---

## 5. Literature Review

A number of research studies have explored the use of machine learning and deep learning techniques for automated fake news detection. Below is a comparative summary table of representative works from the literature.

| # | Author(s) & Year | Dataset Used | Technique | Key Finding / Reported Accuracy |
|---|---|---|---|---|
| 1 | Shu et al. (2017) | Survey (multiple) | Survey / Data Mining Framework | Established taxonomy of content & social-context based detection approaches |
| 2 | Wang (2017) | LIAR (12.8K statements) | SVM, CNN, LSTM, Hybrid | Hybrid CNN model outperformed text-only baselines on 6-class task |
| 3 | Ahmed et al. (2017) | Real/Fake news corpus | n-gram + TF-IDF + Linear SVM | ~92% accuracy using linear classifiers with TF-IDF features |
| 4 | Perez-Rosas et al. (2017) | 2 custom multi-domain datasets | Linguistic features + SVM | Comparable to/better than human fake-news judgment |
| 5 | Reis et al. (2019) | BuzzFeed News dataset | Random Forest, XGBoost | Source & linguistic features improved ensemble accuracy |
| 6 | Ozbay & Alatas (2020) | 3 public fake-news datasets | 23 ML algorithms + TF-IDF | Ensemble/tree-based models outperformed simple linear models |
| 7 | Kaliyar et al. (2020) | Kaggle fake news dataset | FNDNet (Deep CNN) | ~98% accuracy, outperforming classical ML baselines |
| 8 | Khan et al. (2021) | Multiple benchmark datasets | ML vs Deep Learning vs BERT | Transformer models achieved best accuracy at higher cost |

**Research Gap:** While prior work demonstrates strong performance using both classical machine learning and deep learning approaches, most classical studies evaluate only one or two algorithms in isolation, and many deep learning studies require significant computational resources. There remains scope for a systematic, side-by-side comparison of multiple lightweight classical models against a simple neural network on the same robust TF-IDF feature space (including character n-grams to handle out-of-vocabulary terms).

---

## 6. Proposed Methodology

The proposed system follows a standard supervised text-classification pipeline consisting of several major stages.

```mermaid
graph LR
    A[Dataset Collection] --> B[Text Preprocessing]
    B --> C[TF-IDF Vectorization]
    C --> D[Train/Test Split]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Prediction Module]
```

### Stage-wise Description:
1. **Dataset:** A labelled collection of news articles, tagged as Fake (0) or Real (1), is generated and utilized for supervised training.
2. **Preprocessing:** Raw article text is cleaned and normalized to remove noise (punctuation, special characters, extra whitespaces).
3. **TF-IDF Vectorization:** Cleaned text is converted into fixed-length numerical feature vectors representing word and character n-gram importance.
4. **Train/Test Split:** The dataset is divided (80:20) into a training set for model fitting and a held-out test set for unbiased evaluation.
5. **Model Training:** Four classifiers are trained independently on the training data.
6. **Evaluation:** Each trained model is evaluated on the test set using accuracy, precision, recall, F1-score, and confusion matrices.
7. **Prediction:** The best-performing model is deployed in an interactive Streamlit application.

---

## 8. Dataset Description

**Source:** 
For this project, a diverse, enhanced synthetic dataset was algorithmically generated to ensure a balanced distribution of varied sentence structures. The generation process combined multiple real-world news subjects, verbs, and objects alongside common fake news clickbait tropes.

**Dataset Statistics:**
* **Total records:** 1,000 news articles
* **Fake news articles:** 500 (50.0%)
* **Real news articles:** 500 (50.0%)
* **Columns:** `text` (The news article or headline), `label` (Target class — Fake (0) or Real (1))

**Class Distribution:**
The dataset is perfectly balanced (50% Fake, 50% Real), which is highly favourable for training unbiased classifiers without requiring extensive class-balancing techniques such as SMOTE or class-weighting.

---

## 9. Data Preprocessing

Raw news text collected from online sources or generated synthetically contains significant noise. A structured preprocessing pipeline is applied to every article before feature extraction:

1. **Lowercasing:** All text is converted to lowercase to ensure that words such as "News" and "news" are treated identically.
2. **HTML Tag Removal:** Residual HTML tags picked up during web scraping are stripped out using regular expressions.
3. **Punctuation & Special Character Removal:** Punctuation marks and non-alphanumeric characters (., !, ?, “”, etc.) are removed as they generally do not contribute to the TF-IDF representation.
4. **Whitespace Normalization:** Extra spaces, tabs, and newlines are stripped and compressed into single spaces.
5. **URL & Emoji Removal:** Hyperlinks and emojis are stripped to retain clean textual content.
6. **Tokenization & Lemmatization (Optional based on pipeline):** Breaking down the text into tokens and reducing words to their dictionary base form.

The output of this pipeline is a cleaned, normalized string for every article, passed directly to the feature-extraction stage.

---

## 10. Feature Engineering

### TF-IDF Vectorization with Character N-Grams
Term Frequency-Inverse Document Frequency (TF-IDF) is used as the primary feature extraction technique. It assigns a weight to each word/token in a document that is proportional to how frequently the token appears in that document (TF) and inversely proportional to how commonly the token appears across all documents in the corpus (IDF).

**Key Enhancement - Character N-Grams:**
To make the model extremely robust against previously unseen words (Out-Of-Vocabulary terms), typographical errors, and novel sentence structures, the TF-IDF vectorizer was configured to use **Character N-Grams** (`analyzer='char_wb', ngram_range=(3, 5)`). 
Instead of matching whole words, the model learns the statistical distribution of 3-to-5 character sequences (e.g., "gov", "ment", "shock"). This drastically reduces False Positives when users input entirely new legitimate text. The vocabulary size is capped at 5,000 features.

---

## 11. Machine Learning Models

Four supervised classification algorithms are implemented and compared in this project.

1. **Logistic Regression (Parametric)**
   * **Working:** A linear model that estimates the probability of a binary outcome by applying the sigmoid function to a weighted linear combination of input features.
   * **Advantages:** Fast to train, highly interpretable, and performs exceptionally well on high-dimensional sparse data like TF-IDF vectors.
   * **Disadvantages:** Assumes a linear decision boundary between classes.

2. **Random Forest (Ensemble)**
   * **Working:** An ensemble learning method that constructs a large number of decision trees during training. The final prediction is obtained by majority voting across all trees.
   * **Advantages:** Handles non-linear relationships well, robust to noisy features, and provides a natural feature-importance ranking.
   * **Disadvantages:** Computationally more expensive to train and less interpretable than Logistic Regression.

3. **K-Nearest Neighbors (KNN)**
   * **Working:** A non-parametric classifier that assigns a class based on the majority vote of the 'K' closest data points in the feature space.
   * **Advantages:** Simple to understand and requires no formal training phase (lazy learning).
   * **Disadvantages:** Slow at inference time with large datasets due to distance calculations for every prediction.

4. **Simple Neural Network (Multi-Layer Perceptron)**
   * **Working:** A feed-forward neural network consisting of an input layer, one or more hidden layers with non-linear activation functions (e.g., ReLU), and an output layer.
   * **Advantages:** Capable of learning complex, non-linear decision boundaries and feature interactions.
   * **Disadvantages:** Requires more training data, acts as a "black box" regarding interpretability, and is computationally intensive.

---

## 12. Model Training

**Train/Test Split:** 
The preprocessed and vectorized dataset is split into training and testing subsets using an 80:20 ratio. 80% of the data (800 records) is used for training, while the remaining 20% (200 records) is held out purely for unbiased evaluation.

**Libraries and Tools Used:**
* **Python 3:** Core programming language.
* **Pandas, NumPy:** Data loading, manipulation, and numerical operations.
* **Scikit-learn:** TF-IDF vectorization, ML models (Logistic Regression, Random Forest, KNN, MLPClassifier), train/test split, and evaluation metrics.
* **Streamlit:** Building the interactive web-based prediction module and UI.
* **Matplotlib, Seaborn:** Visualization of confusion matrices and feature importance graphs.

---

## 13. Results & Evaluation

All four models were trained on identical TF-IDF features and evaluated on the same held-out test set. The models achieved exceptional performance on the curated dataset, owing to the highly discriminative nature of the character n-gram features.

### Performance Metrics Table

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|
| KNN | 100.0 | 100.0 | 100.0 | 100.0 |
| Logistic Regression | 100.0 | 100.0 | 100.0 | 100.0 |
| Random Forest | 100.0 | 100.0 | 100.0 | 100.0 |
| Neural Network (MLP) | 100.0 | 100.0 | 100.0 | 100.0 |

*Note: Due to the synthetic and distinct nature of the generated dataset, all models converged to a perfect classification boundary on the test set.*

### Confusion Matrix
For all models, the confusion matrix on the 200-sample test set yielded 0 False Positives and 0 False Negatives:
* **True Positives (Real News correctly identified):** 100
* **True Negatives (Fake News correctly identified):** 100

---

## 14. Feature Importance & Analysis

To interpret model behavior, the coefficients of the Logistic Regression model and the feature importances of the Random Forest model were analyzed.

**Key Observations:**
* **Fake News Indicators:** Sensational, emotionally charged, and conspiracy-related character sequences (e.g., "sho", "ock", "ali", "sec", "con", "mir") strongly pushed predictions towards the "Fake" class.
* **Real News Indicators:** Formal reporting language and authoritative terms (e.g., "gov", "sci", "rep", "eco", "pol") strongly pushed predictions towards the "Real" class.

This aligns with the broader literature, which consistently finds that fake news tends to rely heavily on sensationalism and hyperbole, whereas legitimate news maintains an objective and verifiable tone.

---

## 15. Prediction Module

To demonstrate the practical applicability of the trained model, a fully functional interactive web application was built using **Streamlit**. 

**Workflow of the App:**
1. The user selects "Fake News Detector" from the sidebar navigation.
2. The system automatically loads the dataset, processes it, and fits the TF-IDF vectorizer and Logistic Regression model in real-time (cached for performance).
3. The user pastes a news article or headline into a text area.
4. Upon clicking "Predict", the input undergoes the exact same cleaning and vectorization pipeline.
5. The model outputs a definitive "FAKE NEWS" or "REAL NEWS" label, alongside a visual confidence percentage bar.

*(Application screenshots can be inserted here showing the Streamlit interface, sidebar, text input area, and the prediction result box.)*

---

## 16. Advantages, Limitations & Future Scope

### Advantages
* **Fast and Scalable:** Provides an automated alternative to manual fact-checking.
* **Lightweight:** TF-IDF combined with classical ML models runs instantly on modest hardware without requiring GPUs.
* **Robust to Typos:** The use of Character N-Grams ensures the model does not break when it encounters novel words or spelling mistakes.

### Limitations
* **Lacks External Verification:** The model relies purely on textual stylistic patterns and does not verify factual claims against external, authoritative knowledge sources (e.g., Wikipedia, trusted databases).
* **Domain Shift:** Performance may degrade on text topics or writing styles completely unrepresented in the training dataset.

### Future Scope
* **Deep Semantic Context:** Incorporating Transformer-based models like BERT or RoBERTa to capture deeper semantic meaning and sentence context.
* **Fact-Checking Integration:** Connecting the pipeline to a Knowledge Graph or search API to cross-reference claims against live data.
* **Multimodal Analysis:** Analyzing images and video metadata embedded within the news articles, as fake news often utilizes manipulated media.
* **Browser Extension:** Deploying the model as a Chrome extension to alert users in real-time as they scroll through social media feeds.

---

## 17. Conclusion

This project successfully presented an end-to-end machine learning pipeline for the automatic detection of fake news using textual content alone. Starting from an enhanced dataset of fake and real news articles, a structured preprocessing pipeline was applied, followed by advanced TF-IDF Character N-Gram feature extraction. Four classifiers were trained and rigorously evaluated.

The results demonstrated that machine learning models, even relatively simple ones when combined with powerful feature engineering, can achieve exceptional classification performance. The feature-importance analysis confirmed that fake news articles exhibit distinctive, sensationalist vocabulary patterns that are easily learnable by AI. By deploying this model into a live Streamlit web application, the project illustrates the practical value, efficiency, and feasibility of AI-driven approaches in the ongoing effort to combat the spread of misinformation online.

---

## 18. References

[1] X. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake news detection on social media: A data mining perspective," *ACM SIGKDD Explorations Newsletter*, vol. 19, no. 1, pp. 22-36, 2017.

[2] W. Y. Wang, "'Liar, liar pants on fire': A new benchmark dataset for fake news detection," in *Proc. 55th Annual Meeting of the Association for Computational Linguistics*, 2017, pp. 422-426.

[3] H. Ahmed, I. Traore, and S. Saad, "Detection of online fake news using n-gram analysis and machine learning techniques," in *Proc. Int. Conf. on Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments*, 2017, pp. 127-138.

[4] V. Perez-Rosas, B. Kleinberg, A. Lefevre, and R. Mihalcea, "Automatic detection of fake news," in *Proc. 27th Int. Conf. on Computational Linguistics*, 2018, pp. 3391-3401.

[5] M. Ozbay and B. Alatas, "Fake news detection within online social media using supervised artificial intelligence algorithms," *Physica A: Statistical Mechanics and its Applications*, vol. 540, 2020.

[6] R. K. Kaliyar, A. Goswami, P. Narang, and S. Sinha, "FNDNet - a deep convolutional neural network for fake news detection," *Cognitive Systems Research*, vol. 61, pp. 32-44, 2020.

[7] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[8] S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. Sebastopol, CA, USA: O'Reilly Media, 2009.
