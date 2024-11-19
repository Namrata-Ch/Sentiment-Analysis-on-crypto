import pandas as pd
data = pd.read_csv('IndiaWantsCrypto.csv', encoding = 'ISO-8859-1')

data.shape
print(data.head())
#loading data from cvs file to panda dataframe
twitter_data = pd.read_csv('IndiaWantsCrypto.csv', encoding = 'ISO-8859-1')

#Preprocess
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text for sentiment analysis
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and special characters
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Assuming 'data' is your DataFrame with 'date' and 'text' columns
# Apply preprocessing only to the 'text' column
data['cleaned_text'] = data['text'].apply(preprocess_text)

import re
def remove_special_characters(text):
    # Define the pattern for special characters
    pattern = r'[^a-zA-Z0-9\s]'
    # Replace special characters with empty string
    return re.sub(pattern, '', text)

# Apply the function to the text column
data['text'] = data['text'].apply(remove_special_characters)

# Display the preprocessed text along with the 'date' column
print(data[['date','hashtags', 'cleaned_text']])
from textblob import TextBlob

# Function to calculate sentiment polarity using TextBlob
def calculate_sentiment(cleaned_text):
    blob = TextBlob(cleaned_text)
    return blob.sentiment.polarity

# Apply sentiment analysis to the 'cleaned_text' column
data['sentiment'] = data['cleaned_text'].apply(calculate_sentiment)

# Display the DataFrame with sentiment polarity
# Display the preprocessed text along with the 'date' column
print(data[['date','hashtags', 'cleaned_text','sentiment']])

# Concatenate the required columns into a single DataFrame
output_data = pd.concat([data['cleaned_text'], data['date'], data['sentiment'], data['hashtags']], axis=1)

# Set options to display all columns side by side
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Print the DataFrame
print(output_data.head())

output_data['hashtags'].value_counts()

# Define a function to classify sentiment scores
def classify_sentiment(score):
    if score < 0:
        return -1
    elif score == 0:
        return 0
    else:
        return 1

# Apply sentiment classification to the 'Sentiment' column
data['Sentiment_Class'] = data['sentiment'].apply(classify_sentiment)
# Set options to display all columns side by side
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Print the DataFrame with preprocessed headlines, sentiment scores, and their corresponding classes
print(data[['cleaned_text', 'date', 'sentiment', 'Sentiment_Class']].head())

# Concatenate the required columns into a single DataFrame
output_data = pd.concat([data['cleaned_text'], data['date'], data['sentiment'], data['Sentiment_Class'], data['hashtags']], axis=1)

# Set options to display all columns side by side
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Print the DataFrame
print(output_data.head())

# Define the keywords dictionary
keywords = ['Cryptocurrency', 'Bitcoin', 'Ethereum', 'Dogecoin', 'NFT']

# Create a function to detect keywords in preprocessed headlines
def detect_keywords_preprocessed(cleaned_text):
    for keyword in keywords:
        if keyword.lower() in cleaned_text.lower():
            return keyword
    return 'Unknown'

# Apply the function to create the 'Keyword' column
output_data['Keyword'] = output_data['cleaned_text'].apply(detect_keywords_preprocessed)
# Set options to display all columns side by side
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Print the DataFrame
print(output_data.head())

output_data['Keyword'].value_counts()

# Replace 'Unknown' with 'Crypto' in the 'Keyword' column
output_data['Keyword'] = output_data['Keyword'].replace('Unknown', 'Crypto')
output_data.head(10)

#Adding train test and model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = data['cleaned_text']
y = data['Sentiment_Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Random Forest classifier
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=100, random_state=30)
clf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

train_accuracy = accuracy_score(y_train, clf.predict(X_train_tfidf))
print("Training Accuracy:", train_accuracy)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming 'df' is your DataFrame containing preprocessed headlines and sentiment class
X = data['cleaned_text']
y = data['Sentiment_Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=9)
clf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.svm import SVC

# Train Support Vector Machine classifier
svm_clf = SVC(kernel='linear', random_state=70)
svm_clf.fit(X_train_tfidf, y_train)

# Predict on the test set using SVM
svm_y_pred = svm_clf.predict(X_test_tfidf)

# Calculate accuracy for SVM
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print("SVM Accuracy:", svm_accuracy)

print("Random Forest Accuracy:", accuracy)
print("SVM Accuracy:", svm_accuracy)

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for RandomForestClassifier
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

