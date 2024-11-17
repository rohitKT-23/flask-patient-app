from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Flask app initialize karein
app = Flask(__name__)

# Load the model and scaler
model = joblib.load('best_model.pkl')  # Change to the correct path if needed
scaler = joblib.load('scaler.pkl')    # Change to the correct path if needed

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Create index.html for the form

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    try:
        # Convert form data to a NumPy array
        input_features = np.array([[
            float(data['age']),
            float(data['ejection_fraction']),
            float(data['time']),
            int(data['anaemia']),
            int(data['diabetes']),
            int(data['high_blood_pressure']),
            int(data['sex']),
            int(data['smoking'])
        ]])
        
        # Scale the inputs
        scaled_features = scaler.transform(input_features)

        # Predict using the model
        prediction = model.predict(scaled_features)

        # Interpret result
        result = "Survival Predicted" if prediction[0] == 0 else "Death Predicted"
        return jsonify({"Prediction": result})
    
    except Exception as e:
        return jsonify({"Error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
