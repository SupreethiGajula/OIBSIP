import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


df = pd.read_csv("/Users/gajulasupreethi/Desktop/Datasets/oibsiptask3.csv")
#print(df)
#we shall clean the data

# Drop rows with null values in specific columns
#print(df.isna().sum())
df.dropna(subset=['Price','kms_driven', 'fuel_type'], inplace=True)


#there are some modifications in the price column
df['Price'] = df['Price'].str.replace(',','')#we remove commas from the price column
df['Price'].fillna('0', inplace=True)  # Fill missing values with 0
df['Price'] = df['Price'].str.replace('Ask For Price','0')#we remove some strings from the price column
df['Price'] = df['Price'].str.extract('(\d+)').astype(int)  # Extract numeric values from price column
#print(df['Price'])


#feature engineering


#print(df.dtypes)

#we need to convert the categorical values into numerical using onehot encoding/getdummies


encoder = OneHotEncoder()
encoded_data1 = encoder.fit_transform(df[['name']]).toarray()
encoded_data2 = encoder.fit_transform(df[['company']]).toarray()
encoded_data3 = encoder.fit_transform(df[['year']]).toarray()
encoded_data4 = encoder.fit_transform(df[['kms_driven']]).toarray()
encoded_data5 = encoder.fit_transform(df[['fuel_type']]).toarray()

df_combined = pd.concat([pd.DataFrame(encoded_data1),
                         pd.DataFrame(encoded_data2),
                         pd.DataFrame(encoded_data3),
                         pd.DataFrame(encoded_data4),
                         pd.DataFrame(encoded_data5)], axis=1)
                        

df = df.drop(['name','company','year','kms_driven','fuel_type'],axis = 1)

df_final = pd.concat([df,df_combined],axis = 1)

#print(df_final.dtypes)

X = df_final.drop(['Price'],axis=1)
y = df_final['Price']
#print(X.head())
imputer = SimpleImputer()
X = imputer.fit_transform(X)
y = y.fillna(0)


#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


model = DecisionTreeRegressor(max_depth=5,min_samples_split=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)

#for more specefic details we can uncomment some specefic code lines 

