import pandas as pd
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("/Users/gajulasupreethi/Desktop/Datasets/oibsiptask5.csv")
#print(df)
#print(df.dtypes)
#print(df.isna().sum())
#there are 0 null values in the dataset so we can skip the dropping null values and filling 
#in the data preprocessing section

#print(df.columns)
df = df.drop(['Unnamed: 0'],axis = 1)
#print(df.columns)


X = df.drop(['Sales'],axis = 1)
y = df['Sales']
#print(X.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)


model = xgb.XGBRegressor()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print(mse)

#test example

# Load the new data for testing
new_data = pd.read_csv('/Users/gajulasupreethi/Desktop/Datasets/task5test.csv')
print(new_data)
print(new_data.columns)
#new_data = new_data.drop(['NaN'])
#print(new_data)

# Preprocess the new data if required (e.g., handle missing values, encode categorical variables)

# Make predictions on the new data
predictions = model.predict(new_data)

# Display the predictions
print(predictions)
