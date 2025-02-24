import os
from flask import Flask, request, render_template
import pickle
import numpy as np
import shap  # SHAP for Explainable AI

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the trained model
Pkl_Filename = "rf_tuned.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)

# Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)

# Define actual feature names 
feature_names = ["Age", "BMI", "Region", "Gender", "Children", "Smoker"]  # Use actual names

@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form inputs to numpy array
        features = [float(x) for x in request.form.values()]  # Allow float values
        final = np.array(features).reshape((1, -1))

        print("✅ Input Features:", final)  # Debugging input values

        # Predict amount
        pred = model.predict(final)[0]

        # Generate SHAP explanation
        shap_values = explainer.shap_values(final)
        print("✅ SHAP Values:", shap_values[0])  # Debugging SHAP values

        feature_names_dict = {
            0: "Age",
            1: "BMI",
            2: "Region",
            3: "Gender",
            4: "Children",
            5: "Smoker"
        }

        # Correct feature order based on input format
        feature_names = ["Age", "Gender", "BMI", "Children", "Smoker", "Region"]  # Ensure correct order

        # Create a dictionary mapping feature names to SHAP values
        shap_values_dict = {feature_names[i]: shap_values[0][i] for i in range(len(feature_names))}

        # Sort by absolute SHAP values
        sorted_factors = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        # Get top 3 contributing factors
        top_features = sorted_factors[:3]

        # Format output
        explanation_text = ", ".join([f"{name}: {value:.2f}" for name, value in top_features])

        explanation_text = ", ".join([f"{name}: {value:.2f}" for name, value in top_features])

        if pred < 0:
            return render_template('op.html', pred="Error calculating Amount!")
        else:
            return render_template('op.html', pred=f"Expected amount is {pred:.3f}", explanation=f"Top factors: {explanation_text}")

    except Exception as e:
        return render_template('op.html', pred="Error occurred!", explanation=str(e))


if __name__ == '__main__':
    app.run(debug=True)
