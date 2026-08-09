import pandas as pd
import joblib

model = joblib.load("model.pkl")
cols = joblib.load("cols.pkl")

name = input("Enter your name: ")
city = input("Enter your city: ")
income = int(input("Enter your income: "))
credit_score = int(input("enter your credit score: "))
loan_amount = int(input("Enter desired loan amount: "))
years_employed = int(input("Enter years employed: "))

prediction = pd.DataFrame(
    [[income, credit_score, loan_amount, years_employed]],
    columns = cols
)

predicted_points = model.predict(prediction)[0]
print("Points : ", round(predicted_points))

if predicted_points >= 60:
    print("Loan Approved")
    
else:
    print("Loan Denied")
