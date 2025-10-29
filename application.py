import pickle
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create Flask app
application = Flask(__name__)
app = application

# Load model and scaler
grid_model = pickle.load(open("grid.pkl", "rb"))
standard_scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            age = float(request.form.get("age"))
            sex = float(request.form.get("sex"))
            cp = float(request.form.get("cp"))
            trestbps = float(request.form.get("trestbps"))
            chol = float(request.form.get("chol"))
            fbs = float(request.form.get("fbs"))
            restecg = float(request.form.get("restecg"))
            thalach = float(request.form.get("thalach"))
            exang = float(request.form.get("exang"))
            oldpeak = float(request.form.get("oldpeak"))
            slope = float(request.form.get("slope"))
            ca = float(request.form.get("ca"))
            thal = float(request.form.get("thal"))

        except (TypeError, ValueError):
            return render_template("index.html", error="Invalid input — please enter numbers only.")

        # Scale and Predict
        scaled_data = standard_scaler.transform([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]])

        result = grid_model.predict(scaled_data)[0]  # FIXED

        # Convert numeric result to readable text
        if result == 1:
            result_text = "Heart Disease Risk Detected"
        else:
            result_text = "No Heart Disease Risk (Normal)"

        return render_template("index.html", results=result_text)

    # When GET request
    return render_template("index.html")

# Run server
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
