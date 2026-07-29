import pandas as pd

# load scores
df = pd.read_csv('scores.csv')

# get high risk
high_risk = df[df['risk'] == 'HIGH'].copy()

print('Sending emails to', len(high_risk), 'customers...')

# mock: generate offers
offers = []
for idx, row in high_risk.iterrows():
    prob = row['churn_prob']
    
    if prob > 0.8:
        offer = '20% discount for 12 months'
    elif prob > 0.7:
        offer = '15% discount for 6 months'
    else:
        offer = '10% discount for 3 months'
    
    offers.append(offer)
    print(f'Email sent: Risk {prob:.0%} - {offer}')

# save
high_risk['offer'] = offers
high_risk[['churn_prob', 'risk', 'offer']].to_csv('emails.csv')

print('\nEmails sent!')
print('Customers contacted:', len(high_risk))
print('Saved to emails.csv')
