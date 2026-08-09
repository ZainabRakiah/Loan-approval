import pandas as pd
import joblib

model = joblib.load('loan_approval_model.pkl')
model_features = joblib.load('model_features.pkl')

customer = input("Enter customer name: ")
age = int(input("Enter age: "))
city = input("Enter city: ")
country = input("Enter country: ")
professional = input("Enter professional status: ")
salary = int(input("Enter the salary: "))
loan_taken = int(input("Has the customer taken a loan? (1 if yes, 0 if no): "))
if loan_taken == 1:
    previous_loan_amount = int(input("Enter the amount taken before "))
    loan_paid = int(input("Has the customer paid the loan? (Enter 1 if yes, 0 if no) "))
else:
    previous_loan_amount = 0
    loan_paid = 1
loan_required = int(input("Enter loan amount required: "))

prediction_data = pd.DataFrame(
    [[previous_loan_amount, salary, loan_taken, loan_paid, loan_required]],
    columns=model_features
)

model_prediction = model.predict(prediction_data)[0]
print(model_prediction)