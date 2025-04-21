import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Check basic structure
print("Columns:", df.columns.tolist())
print("First few rows:\n", df.head())

# Rename target column to "label" if needed
# Assuming 'label' or 'label_num' are indicators for spam (1) and ham (0)
if 'label_num' in df.columns:
    df['label'] = df['label_num']
elif 'label' in df.columns and df['label'].dtype == object:
    # Convert string labels (e.g. 'ham', 'spam') to numeric
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
else:
    raise ValueError("Target label column not found or unknown format.")

# Use the message column
if 'text' in df.columns:
    df['message'] = df['text']
elif 'email' in df.columns:
    df['message'] = df['email']
elif 'body' in df.columns:
    df['message'] = df['body']
else:
    raise ValueError("No column found containing message text.")

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', '', text)  # remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # remove words containing numbers
    return text

df['clean_message'] = df['message'].astype(str).apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X = vectorizer.fit_transform(df['clean_message'])

# Target
y = df['label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
