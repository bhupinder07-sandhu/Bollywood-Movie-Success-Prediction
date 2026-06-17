from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

svm = joblib.load("models/svm.pkl")
rf = joblib.load("models/rf.pkl")
dt = joblib.load("models/dt.pkl")
knn = joblib.load("models/knn.pkl")
lr = joblib.load("models/lr.pkl")

scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


@app.route("/api/predict", methods=["POST"])
def predict():

    data = request.get_json()

    budget = float(data["budget"])
    india_net = float(data["india_net"])
    india_gross = float(data["india_gross"])
    overseas = float(data["overseas"])

    algorithm = data["algorithm"]

    features = np.array([[
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
            "error":"Invalid Model"
        })

    prediction = model.predict(features)

    result = label_encoder.inverse_transform(prediction)

    return jsonify({
        "prediction": result[0]
    })


if __name__ == "__main__":
    app.run(debug=True)