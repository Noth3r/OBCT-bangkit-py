from services.model import preprocess, predict

def run_prediction(model, scaler, data):
    preprocessed_data = preprocess(scaler, data)
    prediction = predict(model, preprocessed_data)
    return prediction