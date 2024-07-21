from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('linear_model.pkl')

# Load feature names
col_names = ['Amount_Requested', 'Amount_Funded_By_Investors', 'Loan_Length', 'Debt_To_Income_Ratio',
             'Home_Ownership', 'Monthly_Income', 'FICO_Range', 'Open_CREDIT_Lines',
             'Revolving_CREDIT_Balance', 'Inquiries_in_the_Last_6_Months', 'Employment_Length',
             'Loan_Purpose_credit_card', 'Loan_Purpose_debt_consolidation', 'Loan_Purpose_major_purchase', 'Loan_Purpose_other']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    amount_requested = float(request.form['amount-requested'])
    amount_funded = float(request.form['amount-funded'])
    loan_length = float(request.form['loan-length'])
    debt_to_income = float(request.form['debt-to-income'])
    home_ownership = int(request.form['home-ownership'])
    monthly_income = float(request.form['monthly-income'])
    fico_range = float(request.form['fico-range'])
    open_credit_lines = float(request.form['open-credit-lines'])
    revolving_credit_balance = float(request.form['revolving-credit-balance'])
    inquiries_last_6_months = float(request.form['inquiries-last-6-months'])
    employment_length = float(request.form['employment-length'])
    loan_purpose_credit_card = int(request.form['loan-purpose-credit-card'])
    loan_purpose_debt_consolidation = int(request.form['loan-purpose-debt-consolidation'])
    loan_purpose_major_purchase = int(request.form['loan-purpose-major-purchase'])
    loan_purpose_other = int(request.form['loan-purpose-other'])

    # Prepare the input features for prediction
    features = pd.DataFrame([[amount_requested, amount_funded, loan_length, debt_to_income,
                              home_ownership, monthly_income, fico_range, open_credit_lines,
                              revolving_credit_balance, inquiries_last_6_months, employment_length,
                              loan_purpose_credit_card, loan_purpose_debt_consolidation,
                              loan_purpose_major_purchase, loan_purpose_other]], columns=col_names)

    # Predict interest rate
    # interest_rate = model.predict(features)

    # Return the prediction to the user
    # return render_template('index.html', prediction=f'Predicted Interest Rate: {interest_rate}')

    interest_rate = model.predict(features)

    # Convert prediction result to a scalar without using deprecated method
    interest_rate_scalar = np.squeeze(interest_rate)

    # Convert scalar interest rate to integer
    interest_rate_float = float(interest_rate_scalar)

    # Convert prediction result to a string without decimal places
    prediction_string = f'Predicted Interest Rate: {interest_rate_float:.2f}%'

    # Return the prediction string
    return render_template('index.html', prediction=prediction_string)


if __name__ == '__main__':
    app.run(debug=True)

