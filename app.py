import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# load model
model = joblib.load('model.pkl')

# load feature names
with open('features.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

print('Model loaded with', len(features), 'features')

app = Flask(__name__)

# simple html page
html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Churn Prediction</title>
    <style>
        body { font-family: Arial; margin: 50px; }
        input { padding: 10px; margin: 5px; }
        button { padding: 10px 20px; background: blue; color: white; cursor: pointer; }
        .result { margin-top: 20px; padding: 20px; background: #f0f0f0; }
    </style>
</head>
<body>
    <h1>Churn Prediction</h1>
    <form id="form">
        <label>Tenure (months): <input type="number" id="tenure" value="10"></label><br>
        <label>Monthly Charges: <input type="number" id="monthly" value="50"></label><br>
        <label>Total Charges: <input type="number" id="total" value="500"></label><br>
        <button type="button" onclick="predict()">Predict</button>
    </form>
    <div id="result" class="result" style="display:none;"></div>

    <script>
    function predict() {
        let data = {
            tenure: document.getElementById('tenure').value,
            MonthlyCharges: document.getElementById('monthly').value,
            TotalCharges: document.getElementById('total').value
        };
        
        fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        })
        .then(r => r.json())
        .then(d => {
            let result = document.getElementById('result');
            result.innerHTML = '<b>Churn Probability: ' + (d.probability * 100).toFixed(0) + '%</b><br>Risk: ' + d.risk;
            result.style.display = 'block';
        });
    }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # prepare input
    x = []
    for f in features:
        if f in data:
            x.append(float(data[f]))
        else:
            x.append(0)
    
    # predict
    prob = float(model.predict_proba([x])[0][1])
    
    # determine risk
    if prob > 0.6:
        risk = 'HIGH - Send offer'
    elif prob > 0.4:
        risk = 'MEDIUM - Watch'
    else:
        risk = 'LOW - OK'
    
    return jsonify({'probability': prob, 'risk': risk})

if __name__ == '__main__':
    print('Starting server at http://localhost:5000')
    app.run(debug=True, port=5000)
