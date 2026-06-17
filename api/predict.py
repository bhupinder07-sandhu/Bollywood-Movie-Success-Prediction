from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

svm = joblib.load(os.path.join(BASE_DIR, "models", "svm.pkl"))
rf = joblib.load(os.path.join(BASE_DIR, "models", "rf.pkl"))
dt = joblib.load(os.path.join(BASE_DIR, "models", "dt.pkl"))
knn = joblib.load(os.path.join(BASE_DIR, "models", "knn.pkl"))
lr = joblib.load(os.path.join(BASE_DIR, "models", "lr.pkl"))

scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

@app.route("/api/predict", methods=["POST"])
def predict():

    data = request.get_json()

    worldwide = float(data["worldwide"])
    india_net = float(data["india_net"])
    india_gross = float(data["india_gross"])
    overseas = float(data["overseas"])
    budget = float(data["budget"])

    algorithm = data["algorithm"]

    features = np.array([[
        worldwide,
        india_net,
        india_gross,
        overseas,
        budget
    ]])

    features = scaler.transform(features)

    if algorithm == "KNN":
        model = knn

    elif algorithm == "Random Forest":
        model = rf

    elif algorithm == "Decision Tree":
        model = dt

    elif algorithm == "Logistic Regression":
        model = lr

    elif algorithm == "SVM":
        model = svm

    else:
        return jsonify({
            "error": "Invalid Model Selected"
        })

    prediction = model.predict(features)

    result = label_encoder.inverse_transform(prediction)

    return jsonify({
        "prediction": str(result[0])
    })

if __name__ == "__main__":
    app.run(debug=True)