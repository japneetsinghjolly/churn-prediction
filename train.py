import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# load the data
df = pd.read_csv('Telco_Churn.csv')

print('Data loaded:', df.shape)
print('Churn:', df['Churn'].value_counts())

# basic cleaning
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# pick columns to use
cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService', 'Contract', 'OnlineSecurity']

# encode categorical (turn text to numbers)
df_final = pd.get_dummies(df[cols + ['Churn']], drop_first=True)

# separate features and target
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']

print('Features:', X.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
print('Training model...')
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# check accuracy
score = model.score(X_test, y_test)
print('Test Score:', score)

# save model
joblib.dump(model, 'model.pkl')
print('Model saved!')

# save feature names for later
with open('features.txt', 'w') as f:
    for col in X.columns:
        f.write(col + '\n')

print('Done!')
