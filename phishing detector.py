import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Dataset
data = {
    "email": [
        "Congratulations! You won a free iPhone. Click here now!",
        "Your bank account has been suspended. Verify immediately.",
        "Meeting scheduled for tomorrow at 10 AM.",
        "Please find the attached project report.",
        "Claim your reward now by clicking this link.",
        "Let's have lunch together tomorrow.",
        "Update your password immediately to avoid suspension.",
        "Thank you for your purchase. Your order is confirmed."
    ],
    "label": [
        "Phishing",
        "Phishing",
        "Safe",
        "Safe",
        "Phishing",
        "Safe",
        "Phishing",
        "Safe"
    ]
}

df = pd.DataFrame(data)

# Features and Labels
X = df["email"]
y = df["label"]

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Phishing", "Safe"],
            yticklabels=["Phishing", "Safe"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Test Custom Email
while True:
    email = input("\nEnter email text (or type 'exit'): ")

    if email.lower() == "exit":
        break

    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)

    print("Prediction:", prediction[0])