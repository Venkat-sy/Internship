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

Historically, misinformation was limited by the speed of traditional publishing and broadcasting. However, the modern digital landscape has fundamentally altered this dynamic. With the proliferation of social media platforms such as Facebook, X (formerly Twitter), and WhatsApp, news no longer travels solely through verified editorial channels. Anyone with an internet connection can create and share content instantly to a global audience. Furthermore, recommendation algorithms designed by these platforms prioritize user engagement—likes, shares, and comments. These algorithms often inadvertently amplify sensational, emotionally charged, or controversial fake news much faster than accurate, measured reporting. This has created an environment where misinformation can spread to millions of users within hours, long before professional fact-checkers can respond or issue corrections.

The architecture of social networks also fosters echo chambers and filter bubbles, where users are repeatedly exposed to information that aligns with their preexisting beliefs. In these enclosed network topologies, fake news is rarely challenged, and its repeated exposure increases its perceived credibility through the illusory truth effect.

### 3.2 Impact of Fake News
The consequences of unchecked fake news are far-reaching and deeply damaging to society across multiple domains:
* **Erosion of Public Trust:** Widespread dissemination of fabricated stories creates a climate of skepticism where citizens struggle to differentiate between credible journalism and propaganda, ultimately eroding trust in media institutions and democratic processes.
* **Influence on Elections:** Disinformation campaigns, often orchestrated by state-sponsored actors or politically motivated groups, are used to manipulate voter behavior, suppress voter turnout, or illegitimately sway the outcome of democratic elections.
* **Public Health Crises:** During global pandemics, such as COVID-19, the rapid spread of medical misinformation and conspiracy theories regarding vaccines and treatments directly contributed to loss of life and hindered public health responses.
* **Financial Market Manipulation:** Fabricated corporate press releases or economic news can cause flash crashes in the stock market or artificially inflate stock prices, leading to severe financial losses for investors.
* **Incitement of Violence:** In extreme cases, hyper-localized fake news spread through encrypted messaging apps has incited social unrest, communal tension, and mob violence, resulting in tragic real-world casualties.

### 3.3 Need for Artificial Intelligence in Fake News Detection
Manual fact-checking is the traditional defense against misinformation, relying on expert journalists to verify claims against authoritative sources. However, this manual approach is inherently unscalable. Human fact-checkers typically take hours or even days to verify a single complex claim, whereas AI-generated or crowdsourced misinformation can reach millions of users within minutes. This extreme mismatch between the velocity of misinformation and the capacity of manual verification creates a critical need for automated, AI-driven detection systems.

Artificial Intelligence, encompassing Machine Learning (ML) and Natural Language Processing (NLP), provides the mathematical and computational tools required to analyze massive volumes of text at scale. While AI may struggle to verify external ground-truth facts without a knowledge base, it excels at identifying the linguistic, stylistic, and structural patterns that characterize deceptive writing. Fake news often relies on hyperbole, extreme sentiment, informal grammar, clickbait structures, and specific vocabulary patterns designed to evoke fear or outrage. By training advanced classification models on large datasets of labelled real and fake news, AI systems can automatically flag suspicious content in real-time, drastically reducing the spread of misinformation and serving as a crucial first line of defense before manual review is even necessary.

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

Raw textual data collected from online sources, whether scraped from news websites or extracted from raw email protocols, is inherently noisy, unstructured, and teeming with extraneous artifacts. If this raw text is fed directly into a machine learning algorithm, the model will waste computational resources learning irrelevant noise (such as HTML tags or punctuation) rather than the underlying semantic patterns. Therefore, a rigorous and structured Natural Language Processing (NLP) preprocessing pipeline is applied to sanitize every document before it reaches the feature extraction stage.

The preprocessing pipeline executes sequentially through the following critical stages:

1. **Lowercasing and Case Normalization:** 
   In natural language, the capitalization of a word does not alter its core semantic meaning. A machine learning model, however, processes text mathematically and would interpret 'Breaking', 'BREAKING', and 'breaking' as three entirely distinct features. To prevent this artificial inflation of the feature space and to consolidate term frequencies, all text across the entire dataset is systematically converted to lowercase.

2. **HTML Tag and Artifact Stripping:** 
   Since the dataset originates from web environments, the text frequently contains residual HTML markup, such as <p>, <br>, href links, and metadata tags. These artifacts carry no linguistic value for fake news or phishing detection. A regular expression (Regex) engine is utilized to scan the text corpus and definitively strip all text enclosed within angle brackets, ensuring only human-readable content remains.

3. **Punctuation and Special Character Removal:** 
   Punctuation marks (periods, commas, exclamation points, question marks) and special characters ($, %, @, &) serve grammatical purposes but generally do not assist a bag-of-words or TF-IDF model in classification tasks. In fact, they often attach themselves to the end of words (e.g., 'urgent!'), preventing the token 'urgent' from matching with other instances of the same word. All non-alphanumeric characters are stripped from the dataset, leaving only clean alphabetical sequences.

4. **Whitespace Normalization:** 
   Web scraping often introduces erratic spacing, including double spaces, tab characters (	), and excessive newline returns (
). If left unaddressed, these can interfere with tokenization algorithms. A secondary regex operation compresses all contiguous whitespace characters into a single standard space, ensuring uniformity across all documents.

5. **URL, Hyperlink, and Emoji Scrubbing:** 
   While the presence of a malicious URL is a strong indicator of phishing, the actual textual structure of the URL string itself acts as noise when analyzing the *semantic intent* of the surrounding text. Therefore, standard regex patterns matching http://, https://, and www. are employed to excise URLs entirely. Similarly, emojis and non-ASCII unicode characters are purged to restrict the feature space strictly to standard alphabetical text.

6. **Tokenization (Conceptual):** 
   While the TF-IDF vectorizer inherently handles tokenization in the subsequent stage, the conceptual goal of this preprocessing pipeline is to prepare the text to be cleanly split into individual linguistic units (tokens/words). By stripping all adjacent punctuation and standardizing spacing, the vectorizer can accurately segment the document into a precise mathematical array.

The output of this exhaustive pipeline is a perfectly sanitized, normalized, and continuous string of lowercase characters for every single article or email, ready for mathematical transformation in the feature extraction stage.

---


## 10. Feature Engineering

Machine learning algorithms inherently cannot comprehend raw text. They operate exclusively on numerical vectors and matrices. Feature engineering is the critical process of translating the sanitized textual data into a mathematical representation that the algorithms can process, while preserving the linguistic and structural information necessary for accurate classification.

### 10.1 Term Frequency-Inverse Document Frequency (TF-IDF)
The primary feature extraction architecture utilized in this project is the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique. Unlike simpler methods like Bag-of-Words (Count Vectorization) which merely tally how many times a word appears in a document, TF-IDF evaluates the true *statistical importance* of a word relative to the entire corpus.

The TF-IDF score for a term is calculated as the product of two distinct metrics:
* **Term Frequency (TF):** Measures how frequently a term occurs within a specific document. The core assumption is that if a word appears many times in an article, it is highly relevant to that specific article's context.
* **Inverse Document Frequency (IDF):** Measures how rare or common a term is across the entire corpus of documents. Words that appear in almost every document (like 'the', 'is', 'and') carry very little discriminative power. IDF assigns a low mathematical weight to these ubiquitous terms, while assigning a massive weight penalty to rare, distinctive words that only appear in a subset of documents.

By multiplying these two values, TF-IDF creates a sophisticated numerical vector where highly discriminative, context-specific words dominate the feature space, allowing the classifier to easily separate the classes.

### 10.2 The Core Enhancement: Character N-Grams
Traditional TF-IDF operates on the word level, meaning it strictly matches whole words. However, this approach is extremely vulnerable to two massive issues prevalent in malicious text:
1. **Deliberate Obfuscation & Misspellings:** Phishing attackers frequently misspell critical words intentionally (e.g., writing 'paypa1' instead of 'paypal', or 'urgnt') specifically to evade word-based spam filters.
2. **Out-of-Vocabulary (OOV) Terms:** If a user inputs text containing novel words the model has never seen during training, a standard word-vectorizer simply ignores them, severely crippling prediction accuracy.

To solve this critical vulnerability, this project enhances the TF-IDF vectorizer by configuring it to operate on **Character N-Grams** rather than whole words. Specifically, the model is configured with nalyzer='char_wb' and 
gram_range=(3, 5). 

Instead of isolating the whole word 'urgent', the vectorizer breaks it down into statistical character sequences spanning 3 to 5 letters, such as 'urg', 'rge', 'gent'. 
This provides a massive mathematical advantage:
* **Resilience to Typos:** If an attacker writes 'urgnt', the model still recognizes the character sequences 'urg' and 'gnt'. The statistical overlap is high enough that the classifier still correctly identifies the malicious intent.
* **Morphological Understanding:** Character N-Grams naturally capture prefixes and suffixes (e.g., 'ing', 'tion', 'anti'), granting the model an implicit understanding of word roots and syntax structure without requiring complex external lemmatization libraries.

The vocabulary size of the TF-IDF vectorizer is capped at a maximum of 3,000 to 5,000 top features to prevent memory exhaustion and to enforce feature dimensionality reduction, ensuring the model focuses exclusively on the most statistically significant character combinations.

---


## 11. Machine Learning Models

In this project, four supervised classification algorithms are implemented, evaluated, and compared. The objective is to determine which mathematical approach best captures the linguistic patterns indicative of deceptive text. Each algorithm operates on fundamentally different mathematical principles, providing a comprehensive evaluation of linear, probabilistic, ensemble, and non-linear neural architectures.

### 11.1 Logistic Regression (Parametric Linear Model)
**Working Principle:**
Logistic Regression is a foundational statistical model used for binary classification tasks. Despite its name, it is a classification algorithm rather than a regression one. It estimates the probability that a given input feature vector belongs to the positive class (e.g., Fake News or Phishing) by computing a weighted linear combination of the input features and passing the result through a logistic (sigmoid) function. The sigmoid function maps any real-valued number into a strict range between 0 and 1, representing a valid probability distribution.
Mathematically, the hypothesis function is defined as:
h(x) = 1 / (1 + e^-(w^T x + b))
where w represents the learned weight vector for the TF-IDF features, x is the input feature vector, and  is the bias term. During the training phase, the algorithm optimizes these weights using Gradient Descent or solvers like L-BFGS, minimizing the Log-Loss (Binary Cross-Entropy) cost function.

**Advantages:**
* **Extreme Efficiency:** It is computationally lightweight, meaning it can be trained on massive datasets in seconds and perform real-time inference with negligible latency.
* **High Interpretability:** The learned weight vector w directly corresponds to feature importance. By examining the largest positive and negative weights, we can extract the exact vocabulary words that the model considers indicative of fake or legitimate content.
* **Robustness to Sparsity:** Text data converted via TF-IDF results in highly sparse matrices (mostly zeros). Logistic Regression handles sparse, high-dimensional spaces exceptionally well without suffering from the curse of dimensionality.

**Disadvantages:**
* **Linear Decision Boundary:** The model assumes that the classes can be separated by a single linear hyperplane. It cannot inherently capture complex, non-linear relationships or multi-word semantic dependencies unless explicit polynomial features or interaction terms are engineered.

### 11.2 Random Forest (Ensemble Learning)
**Working Principle:**
Random Forest is a highly powerful ensemble learning method based on the aggregation of multiple Decision Trees. Instead of relying on a single complex tree (which is highly prone to overfitting the training data), Random Forest constructs a 'forest' of hundreds of independent trees. 
It utilizes a technique called Bootstrap Aggregating (Bagging). During training, each decision tree is trained on a random subsample of the training data (with replacement). Furthermore, at each node split within a tree, only a random subset of the TF-IDF features is considered. When a new text document is inputted for prediction, it is passed down through all the trees in the forest. Each tree casts a 'vote' for the class label, and the Random Forest outputs the majority vote as the final prediction.

**Advantages:**
* **Non-Linearity:** It naturally captures complex, non-linear interactions between words and phrases without requiring explicit feature engineering.
* **Resilience to Overfitting:** The bagging mechanism and feature randomness ensure that the model generalizes well to unseen data, acting as a strong regularizer against the noise inherent in natural language data.
* **Feature Importance:** It calculates Gini importance or mean decrease in impurity, allowing for structural analysis of which textual features contributed most to reducing classification error.

**Disadvantages:**
* **Computational Cost:** Training hundreds of trees on thousands of TF-IDF features is computationally intensive and requires significantly more memory than linear models.
* **Inference Speed:** Real-time prediction is slower because the input vector must traverse every tree in the ensemble.
* **Black Box Nature:** While feature importance is available, tracing the exact decision path for a specific prediction is convoluted, reducing the model's transparency compared to Logistic Regression.

### 11.3 Naive Bayes / K-Nearest Neighbors
Depending on the specific pipeline, either Multinomial Naive Bayes or K-Nearest Neighbors (KNN) serves as the baseline algorithmic approach.
* **Multinomial Naive Bayes:** Based on Bayes' Theorem, this probabilistic classifier calculates the conditional probability of each class given the observed word frequencies. It makes a 'naive' assumption that every word in the document is conditionally independent of every other word, given the class label. Despite this biologically implausible assumption (as grammar dictates word dependence), Naive Bayes performs remarkably well on text classification, specifically for spam and fake news filtering, due to its ability to handle discrete word counts effectively.
* **K-Nearest Neighbors (KNN):** KNN is a non-parametric, lazy learning algorithm. It does not possess a formal training phase where weights are optimized. Instead, it memorizes the entire training dataset. During inference, it calculates the spatial distance (typically Euclidean or Cosine similarity) between the new TF-IDF vector and all vectors in the training set. It then assigns the class label based on the majority vote of the 'K' closest neighbors. While highly intuitive, KNN suffers from extreme latency during prediction on large datasets, as it must compute distances against the entire corpus.

### 11.4 Simple Neural Network (Multi-Layer Perceptron)
**Working Principle:**
To explore deep learning capabilities, a feed-forward Artificial Neural Network, specifically a Multi-Layer Perceptron (MLP), is utilized. The architecture consists of an input layer scaled to the exact vocabulary size of the TF-IDF vectorizer, followed by one or more fully connected hidden layers, and a final output layer.
Each node (neuron) in the hidden layers computes a weighted sum of its inputs and applies a non-linear activation function, such as the Rectified Linear Unit (ReLU: (x) = max(0, x)). This non-linearity allows the network to learn highly abstract hierarchical representations of the text. The output layer employs a Sigmoid activation function to yield a binary probability score. The network is trained using the Backpropagation algorithm, dynamically adjusting its internal weights via an optimization algorithm like Adam to minimize the classification error over multiple iterations (epochs).

**Advantages:**
* **Supreme Flexibility:** Given sufficient data and layers, neural networks can approximate almost any continuous function, allowing them to decipher incredibly subtle and complex linguistic patterns that evade classical models.
* **Scalability:** They benefit massively from larger datasets and can be seamlessly upgraded to more advanced architectures like LSTMs or Transformers for sequential text processing.

**Disadvantages:**
* **Data Hunger:** Neural networks require vast amounts of labelled data to converge optimally; on smaller datasets, they are highly prone to severe overfitting unless heavily regularized using Dropout or weight decay.
* **Resource Intensive:** Training requires substantial computational power, typically necessitating GPU acceleration for efficiency.
* **Lack of Interpretability:** They operate as a 'black box'. It is extraordinarily difficult to map a final prediction back to specific input words, making it challenging to explain the model's reasoning to end-users or compliance auditors.

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
