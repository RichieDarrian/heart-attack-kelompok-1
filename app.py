import streamlit as st
import joblib
import pandas as pd

# Load artifacts
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")


def make_prediction(input_df):
    # Preprocess raw input
    X_processed = preprocessor.transform(input_df)

    # Predict class
    prediction = model.predict(X_processed)[0]

    # Predict probability if available
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_processed)[0][1]

    return prediction, probability


def main():
    st.title("Heart Attack Prediction")

    st.subheader("Patient Information")

    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol (chol)", min_value=50, max_value=700, value=200)
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", options=[0, 1, 2, 3])

    if st.button("Make Prediction"):
        input_df = pd.DataFrame([{
            "age": age,
            "trestbps": trestbps,
            "chol": chol,
            "thalach": thalach,
            "restecg": restecg,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "sex": sex,
            "cp": cp,
            "exang": exang,
            "thal": thal,
        }])

        prediction, probability = make_prediction(input_df)

        if prediction == 1:
            st.error("Prediction: Heart Attack")
        else:
            st.success("Prediction: No Heart Attack")

        if probability is not None:
            st.write(f"Probability of Heart Attack: **{probability:.4f}**")


if __name__ == "__main__":
    main()
