import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LinearRegression

col_name = ['Amount_Requested', 'Amount_Funded_By_Investors','Interest_Rate', 'Loan_Length', 'Debt_To_Income_Ratio',
                 'Home_Ownership', 'Monthly_Income', 'FICO_Range', 'Open_CREDIT_Lines',
                 'Revolving_CREDIT_Balance', 'Inquiries_in_the_Last_6_Months', 'Employment_Length',
                 'Loan_Purpose_credit_card', 'Loan_Purpose_debt_consolidation', 'Loan_Purpose_major_purchase', 'Loan_Purpose_other']
df = pd.read_csv('loan_datatrain.csv',names=col_name)
print(df)
features = df.drop('Interest_Rate',axis=1)
print(features)
target =df[['Interest_Rate']]
print(target)
X_train,X_test,y_train,y_test = train_test_split(features,target, test_size=0.25 ,random_state=101)

model =LinearRegression()
model.fit(X_train,y_train)
# result = model.score(X_train,y_train)
# print(result)

joblib.dump(model, '../webpage/linear_model.pkl')