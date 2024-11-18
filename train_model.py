import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load your dataset
data = pd.read_csv('dish.csv')  # Assuming columns: 'description' and 'ingredients'

# Preprocess text
X = data['description']
y = data['ingredients']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train a basic model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")
