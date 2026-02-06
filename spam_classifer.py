# Spam Email Classifier using Machine Learning

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
emails = [
    "Win money now",
    "Limited offer click here",
    "Hello how are you",
    "Meeting tomorrow",
    "Free gift card available",
    "Let's have lunch today"
]

labels = [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test email
test_email = ["hi ,are we meeting tomorrow?"]
test_vector = vectorizer.transform(test_email)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")