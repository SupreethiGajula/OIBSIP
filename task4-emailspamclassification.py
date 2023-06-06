import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

# Read the data with specified encoding
df = pd.read_csv("/Users/gajulasupreethi/Desktop/Datasets/oibsiptask4.csv", encoding='latin1')
#print(df)
#print(df.columns)

#Data cleaning

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis = 1)#some unnecessary extra columns 
#print(df.columns)

#features and labels
X = df['v2']
y = df['v1']
#print(X.head())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



#we use count vectorizer to convert the email texts into numeric values
cv = CountVectorizer()
X_train_vec = cv.fit_transform(X_train)
X_test_vec = cv.transform(X_test)


#build the model
model = SVC(C = 1, gamma = 'auto')
model.fit(X_train_vec,y_train)

#predicted values
y_pred = model.predict(X_test_vec)

#metrics


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam',zero_division=1)


print("Accuracy:", accuracy)
print("Precision:", precision)

