from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scalers
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('time_scaler.pkl', 'rb') as f:
    time_scaler = pickle.load(f)
with open('amount_scaler.pkl', 'rb') as f:
    amount_scaler = pickle.load(f)

# Define the feature order expected by the model
# V1-V28, then scaled_amount, then scaled_time
feature_order = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        form_values = request.form.to_dict()
        
        # Extract and scale Time and Amount
        time_val = np.array([float(form_values.pop('Time'))]).reshape(-1, 1)
        amount_val = np.array([float(form_values.pop('Amount'))]).reshape(-1, 1)
        
        scaled_time = time_scaler.transform(time_val)[0, 0]
        scaled_amount = amount_scaler.transform(amount_val)[0, 0]
        
        # Prepare other V features
        v_features = [float(form_values[f'V{i}']) for i in range(1, 29)]
        
        # Combine all features in the correct order
        final_features_list = v_features + [scaled_amount, scaled_time]
        final_features = np.array(final_features_list).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)
        prediction_proba = model.predict_proba(final_features)

        if prediction[0] == 1:
            fraud_prob = prediction_proba[0][1] * 100
            prediction_text = f"Fraudulent Transaction Detected!"
            prediction_details = f"Confidence: {fraud_prob:.2f}%"
            prediction_class = "fraud"
        else:
            safe_prob = prediction_proba[0][0] * 100
            prediction_text = "Transaction Appears to be Legitimate."
            prediction_details = f"Confidence: {safe_prob:.2f}%"
            prediction_class = "safe"

        return render_template('index.html', prediction_text=prediction_text, prediction_details=prediction_details, prediction_class=prediction_class)

    except Exception as e:
        error_message = f"An error occurred: {e}"
        return render_template('index.html', prediction_text=error_message, prediction_class="error")

if __name__ == "__main__":
    app.run(debug=True)