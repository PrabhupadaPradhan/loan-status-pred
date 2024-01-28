import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the loan dataset
df = pd.read_csv("loan_data.csv")

# Preprocess the data
df.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
df = df.replace(to_replace='3+', value=4)
df.replace({'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0},
            'Self_Employed': {'No': 0, 'Yes': 1}, 'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
            'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

# Train-test split
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Define a function to predict loan status
def predict_loan_status(applicant_data):
    # Convert applicant data to a DataFrame
    applicant_df = pd.DataFrame([applicant_data])

    # Preprocess the applicant data similarly to the training data
    applicant_df.replace({'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0},
                          'Self_Employed': {'No': 0, 'Yes': 1}, 'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                          'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

    # Predict the loan status
    prediction = model.predict(applicant_df)[0]
    return prediction

# Create a Streamlit app
st.title("Loan Status Prediction App")

# Get user input for applicant data
applicant_data = {}
for col in X.columns:
    applicant_data[col] = st.text_input(col)

# Predict loan status and display the result
if st.button("Predict Loan Status"):
    prediction = predict_loan_status(applicant_data)
    if prediction == 1:
        st.write("Congratulations! Your loan is likely to be approved.")
    else:
        st.write("Sorry, your loan is likely to be rejected.")
