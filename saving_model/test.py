import joblib
import pandas as pd
model = joblib.load('../webpage/linear_model.pkl')

col_names = ['Amount_Requested', 'Amount_Funded_By_Investors', 'Loan_Length', 'Debt_To_Income_Ratio',
                 'Home_Ownership', 'Monthly_Income', 'FICO_Range', 'Open_CREDIT_Lines',
                 'Revolving_CREDIT_Balance', 'Inquiries_in_the_Last_6_Months', 'Employment_Length',
                 'Loan_Purpose_credit_card', 'Loan_Purpose_debt_consolidation', 'Loan_Purpose_major_purchase', 'Loan_Purpose_other']

X_test_with_names = pd.DataFrame([[22000.0,	22000.0	,60.0,	18.28,	2	,6083.33,	722.0	,9.0,	20181.0	,0.0,	8.0,	0,	1,	0	,0]], columns=col_names)

re = model.predict(X_test_with_names)
print(re)