from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained pipeline model
model = pickle.load(open('pipe.pkl', 'rb'))

# Define the route for the main page
@app.route('/')

# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data from HTML page
    person_age = int(request.form['person_age'])
    person_income = int(request.form['person_income'])
    person_emp_length = float(request.form['person_emp_length'])
    loan_amnt = int(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    loan_percent_income = float(request.form['loan_percent_income'])
    cb_person_cred_hist_length = int(request.form['cb_person_cred_hist_length'])
    person_home_ownership = request.form['person_home_ownership']
    loan_grade = request.form['loan_grade']
    cb_person_default_on_file = request.form['cb_person_default_on_file']
    loan_intent = request.form['loan_intent']
    
    
    # Create feature array in the order expected by the pipeline
    features = np.array([[
        person_age, person_income, person_home_ownership, person_emp_length,
        loan_intent, loan_grade, loan_amnt, loan_int_rate,
        loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length
    ]])

    # Predict using the pipeline model
    prediction = model.predict(features)
    prediction_text = 'Congratulations Your Loan Approved' if prediction[0] == 1 else 'Sorry...Your Loan Not Approved'

    # Render the result in the HTML template
    return jsonify({'Status ' : prediction_text})

if __name__ == '__main__':
    app.run(debug=False)
