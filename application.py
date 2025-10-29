from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("grid.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            data = [
                float(request.form.get("cp")),
                float(request.form.get("trestbps")),
                float(request.form.get("chol")),
                float(request.form.get("fbs")),
                float(request.form.get("restecg")),
                float(request.form.get("thalach")),
                float(request.form.get("exang")),
                float(request.form.get("oldpeak")),
                float(request.form.get("slope")),
                float(request.form.get("ca")),
                float(request.form.get("thal"))
            ]

            data = scaler.transform([data])
            prediction = model.predict(data)[0]

            return render_template("index.html", results=prediction)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")
