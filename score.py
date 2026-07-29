import pandas as pd
import joblib

# load model
model = joblib.load('model.pkl')

# load feature names
with open('features.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

# load test data (or all data)
df = pd.read_csv('Telco_Churn.csv')

print('Scoring', len(df), 'customers...')

# prepare data same way as training
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService', 'Contract', 'OnlineSecurity']
df_final = pd.get_dummies(df[cols], drop_first=True)

# ensure all features present
for f in features:
    if f not in df_final.columns:
        df_final[f] = 0

# reorder columns to match training
df_final = df_final[features]

# predict
probs = model.predict_proba(df_final)[:, 1]

# add to dataframe
df['churn_prob'] = probs
df['risk'] = df['churn_prob'].apply(lambda x: 'HIGH' if x > 0.6 else ('MEDIUM' if x > 0.4 else 'LOW'))

# stats
print('\nResults:')
print('HIGH risk:', (df['risk'] == 'HIGH').sum())
print('MEDIUM risk:', (df['risk'] == 'MEDIUM').sum())
print('LOW risk:', (df['risk'] == 'LOW').sum())

# save
df[['churn_prob', 'risk']].to_csv('scores.csv')
print('\nScores saved to scores.csv')

# show high risk
print('\nTop 10 high risk customers:')
high = df[df['risk'] == 'HIGH'].nlargest(10, 'churn_prob')
for i, row in high.iterrows():
    print(f"  Risk: {row['churn_prob']:.0%}")
