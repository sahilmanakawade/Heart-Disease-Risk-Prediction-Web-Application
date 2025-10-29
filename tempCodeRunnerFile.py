import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create Flask app
application = Flask(__name__)
app = application

# --- Load your trained model and scaler ---
grid_model = pickle.load(open(
    "grid.pkl", "rb"
))
standard_scaler = pickle.load(open(
    "scaler.pkl", "rb"
))

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Get form data safely
        try:
            age= float(request.form.get("age"))
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

        # Scale and predict
        scaled_data = standard_scaler.transform([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        result = grid_model.predict(scaled_data)

        return render_template("index.html", results=round(result[0], 2))
# --- Start server ---
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
