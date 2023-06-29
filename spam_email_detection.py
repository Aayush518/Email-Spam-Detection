import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Step 1: Load the dataset
data = pd.read_csv('spam_ham_dataset.csv')

# Step 2: Split the dataset into training and testing sets
X = data['text']       # Input features (email text)
y = data['label']      # Target variable (spam or non-spam)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature extraction using Bag-of-Words model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)  # Convert email text into a matrix of token counts
X_test = vectorizer.transform(X_test)        # Apply the same transformation to the test set

# Step 4: Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 5: Create the Streamlit app
st.title("Spam Email Detection")

# Step 6: Prompt the user to enter an email and make a prediction
email = st.text_area("Enter an email")
if st.button("Predict"):
    new_email_transformed = vectorizer.transform([email])
    prediction = classifier.predict(new_email_transformed)
    st.write("Prediction:", prediction[0])
