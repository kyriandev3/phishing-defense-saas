import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import joblib

# Load dataset
data = pd.read_csv('datasets/CEAS_08.csv')

# Convert columns to strings to avoid concatenation errors
data['subject'] = data['subject'].astype(str)
data['body'] = data['body'].astype(str)
data['urls'] = data['urls'].astype(str)

# Combine 'subject', 'body', and 'urls' into one column for training
data['combined_text'] = data['subject'] + " " + data['body'] + " " + data['urls']

# Preprocess data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['combined_text']) #Using the combined column
y = data['label']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save X_test and y_test for evaluation
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_test, 'y_test.pkl')

# Save the trained model as well
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')  # Save the vectorizer

# Test accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and vectorizer for future use
import pickle
with open('models/phishing_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)