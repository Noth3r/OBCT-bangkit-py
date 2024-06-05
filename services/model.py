import pandas as pd
import numpy as np

def preprocess(scaler, data):
    df_input = pd.DataFrame([data])
    print(df_input)
    numeric_cols = ['Age', 'Height', 'Weight']
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    print(df_input)

    return df_input

def predict(model, df_input):
    prediction = model.predict(df_input)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]