import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_excel('shuffled_loan_dataset.xlsx')


df['loan_taken'] = df['loan_taken'].map({'yes': 1, 'no': 0})
df['loan_paid'] = df['loan_paid'].map({'yes': 1, 'no': 0, 'na': 1})

X = df.drop(["customer","age","city","country","professional","loan_status"],axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

joblib.dump(model, 'loan_approval_model.pkl')
joblib.dump(X.columns, 'model_features.pkl')