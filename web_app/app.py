from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model_path = os.path.join('models', 'phishing_model.pkl')
vectorizer_path = os.path.join('models', 'vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input email text
        email_text = request.form['email_text']

        # Preprocess and predict
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]

        # Display result
        result = 'Phishing' if prediction == 1 else 'Not Phishing'
        return render_template('index.html', prediction=result, email_text=email_text)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
