from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
grid_model = pickle.load(open("grid.pkl", "rb"))
standard_scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None

    if request.method == "POST":
        try:
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]

            final_features = standard_scaler.transform([features])
            prediction = grid_model.predict(final_features)[0]

            results = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        except Exception:
            error = "Invalid input values. Please enter correct numerical data."

    return render_template("index.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True)
