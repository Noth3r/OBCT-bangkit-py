from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import auth
from controllers.model import run_prediction
import tensorflow as tf
import logging
import joblib

creds = firebase_admin.credentials.Certificate('./config/firebase-admin.json')
default_app = firebase_admin.initialize_app(creds)

app = Flask(__name__)

model = tf.keras.models.load_model("./config/model.h5")
scaler = joblib.load("./config/std_scaler.bin")

@app.before_request
def hook():
    # verify the user's token
    id_token = request.headers.get('Authorization')

    if not id_token:
        return jsonify({'error': 'No token provided'}), 401

    try:
        id_token = id_token.split('Bearer ')[1]
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']

        user = auth.get_user(uid)
        isFirstTime = user.user_metadata.creation_timestamp == user.user_metadata.last_sign_in_timestamp
        print(user)
        if isFirstTime:
            return jsonify({'error': 'First time login'}), 401
    except:
        # log the error
        logging.exception("message")

        return jsonify({'error': 'Invalid token'}), 401


@app.route('/api/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    prediction = run_prediction(model, scaler, data)
    return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)