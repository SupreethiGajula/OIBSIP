import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Read the data with specified encoding
df = pd.read_csv("/Users/gajulasupreethi/Desktop/Datasets/oibsiptask4.csv", encoding='latin1')

# Data cleaning
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Features and labels
X = df['v2']
y = df['v1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the email texts into numeric values using CountVectorizer
cv = CountVectorizer()
X_train_vec = cv.fit_transform(X_train)
X_test_vec = cv.transform(X_test)

# Build the logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict values
y_pred = model.predict(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam', zero_division=1)
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Example prediction
test = ["Congratulations! You have won a free vacation. Click here to claim your prize."]#spam
#prints spam
pred = model.predict(cv.transform(test))

print("Prediction:", pred)
