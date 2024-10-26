from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('models/phishing_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded", 400
        
        # Read and process the uploaded file
        text = file.read().decode('utf-8')
        prediction = model.predict(vectorizer.transform([text]))

        # Define result and recommendations
        result = 'Phishing' if prediction[0] == 1 else 'Safe'
        recommendation = 'Do not click any links and report it.' if result == 'Phishing' else 'Safe to open.'

        return render_template('index.html', prediction={'result': result, 'recommendation': recommendation, 'class': 'phishing' if result == 'Phishing' else 'safe'})

if __name__ == '__main__':
    app.run(debug=True)
