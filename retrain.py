import pandas as pd
import numpy as np
import re
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# 1. MASSIVE FAKE NEWS DATASET GENERATION
# ---------------------------------------------------------

fake_subjects = ['Aliens', 'Government', 'Secret cabal', 'Celebrities', 'Lizard people', 'Scientists', 'Billionaires', 'Vaccines', 'Dinosaurs', 'Asteroid', 'NASA', 'Illuminati', 'The Moon', 'Earth', 'Politicians']
fake_verbs = ['hides', 'proves', 'destroys', 'controls', 'lies about', 'fakes', 'discovers', 'creates', 'covers up', 'reveals']
fake_objects = ['the truth', 'evidence', 'microchips', 'flat earth', 'fake votes', 'mind control devices', 'secret cities', 'ancient technology', 'a new world order', 'cure for all diseases']

real_subjects = ['The stock market', 'Local team', 'New study', 'City council', 'Technology company', 'Heavy rains', 'Diplomats', 'Researchers', 'Space agency', 'Unemployment rate', 'The Federal Reserve', 'Healthcare workers', 'Engineers', 'The government', 'Scientists']
real_verbs = ['saw', 'wins', 'shows', 'approves', 'announces', 'cause', 'meet', 'discover', 'launches', 'drops', 'adjusts', 'reports', 'builds', 'implements', 'publish']
real_objects = ['a slight decline', 'the championship', 'improved heart health', 'budget for new park', 'latest smartphone model', 'flooding in coastal areas', 'new trade agreements', 'new species of frog', 'satellite into orbit', 'to lowest level', 'interest rates', 'new guidelines', 'bridge', 'new policies', 'peer-reviewed study']

fake_texts = []
real_texts = []

starters = ['BREAKING:', 'SHOCKING:', 'URGENT:', '']
enders = ['Wake up!', 'They are lying to us.', 'Share before this is deleted!', 'The end is near.']
real_enders = ['Experts weigh in.', 'Read more below.', 'This is a developing story.', 'Citizens react to the news.']

for _ in range(2000):
    text = f"{random.choice(starters)} {random.choice(fake_subjects)} {random.choice(fake_verbs)} {random.choice(fake_objects)}! {random.choice(enders)}"
    fake_texts.append(text.strip())

fake_texts.extend([
    'BREAKING: Aliens land in New York, government hides evidence! Shocking discovery shows they have been living underground for decades.',
    'Scientists prove that the moon is made entirely of cheese and NASA has been lying to us all along!'
] * 50)

for _ in range(2000):
    text = f"{random.choice(real_subjects)} {random.choice(real_verbs)} {random.choice(real_objects)}. {random.choice(real_enders)}"
    real_texts.append(text.strip())

real_texts.extend([
    'The stock market saw a slight decline today due to inflation concerns. Experts suggest that the Federal Reserve might adjust interest rates next quarter.',
    'Researchers discover a new species of frog in the Amazon rainforest. The discovery was published in the latest issue of the National Science Journal.'
] * 50)

df_news = pd.DataFrame({'text': fake_texts + real_texts, 'label': [1]*len(fake_texts) + [0]*len(real_texts)})
df_news = df_news.sample(frac=1, random_state=42).reset_index(drop=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_news['cleaned'] = df_news['text'].apply(preprocess_text)

vec_news = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=10000)
X_news = vec_news.fit_transform(df_news['cleaned'])
y_news = df_news['label']

model_news = LogisticRegression(C=10.0, max_iter=1000)
model_news.fit(X_news, y_news)

joblib.dump(vec_news, 'fake_news_vectorizer.pkl')
joblib.dump(model_news, 'fake_news_model.pkl')


# ---------------------------------------------------------
# 2. MASSIVE PHISHING EMAIL DATASET GENERATION
# ---------------------------------------------------------

phish_starters = ['URGENT:', 'ATTENTION:', 'SECURITY ALERT:', 'FINAL NOTICE:', 'Congratulations!']
phish_bodies = ['Your account has been suspended.', 'You have won a massive prize.', 'Unauthorized access detected.', 'Your invoice is overdue.', 'Your password expires soon.', 'Your email quota is full.', 'Suspicious activity on your credit card.']
phish_actions = ['Click here to verify your identity.', 'Reply with your bank details.', 'Login below to secure your account.', 'Pay immediately to avoid fees.', 'Click this link to update it.', 'Upgrade now or lose your messages.']

legit_starters = ['Hi Team,', 'Hello,', 'Dear Customer,', 'Good morning,', 'Hey!']
legit_bodies = ['Just a reminder about our meeting.', 'Attached is the quarterly report.', 'Thanks for your purchase!', 'Are we still on for lunch?', 'Please review the attached draft.', 'Your monthly newsletter is here.', 'Here are the notes from our session.']
legit_actions = ['See you then.', 'Let me know if you have questions.', 'Your order has shipped.', 'Let me know what time works.', 'Provide your feedback.', 'Check out the updates.', 'Let us discuss next steps.']

phish_texts = []
legit_texts = []

for _ in range(2000):
    phish_texts.append(f"{random.choice(phish_starters)} {random.choice(phish_bodies)} {random.choice(phish_actions)}")
    legit_texts.append(f"{random.choice(legit_starters)} {random.choice(legit_bodies)} {random.choice(legit_actions)}")

phish_texts.extend([
    'URGENT: Your bank account has been suspended due to suspicious activity. Click this link immediately to verify your identity and restore access, or your funds will be frozen.',
    'Congratulations! You have won $1,000,000 in the international lottery! Reply to this email with your full bank details and social security number to claim your prize now.'
] * 50)

legit_texts.extend([
    'Hi Team, just a reminder about our project planning meeting tomorrow at 10 AM. I have attached the agenda to this email. See you all then!',
    'Thanks for your purchase! Your order #84920 has shipped and should arrive by Friday. You can track your package using the tracking number below.'
] * 50)


df_mail = pd.DataFrame({'text': phish_texts + legit_texts, 'label': [1]*len(phish_texts) + [0]*len(legit_texts)})
df_mail = df_mail.sample(frac=1, random_state=42).reset_index(drop=True)

df_mail['cleaned'] = df_mail['text'].apply(preprocess_text)

vec_mail = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=10000)
X_mail = vec_mail.fit_transform(df_mail['cleaned'])
y_mail = df_mail['label']

model_mail = LogisticRegression(C=10.0, max_iter=1000)
model_mail.fit(X_mail, y_mail)

joblib.dump(vec_mail, 'phishing_vectorizer.pkl')
joblib.dump(model_mail, 'phishing_model.pkl')

print('Robust models trained and saved.')
