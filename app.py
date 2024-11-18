import streamlit as st
import joblib
import numpy as np  # We use this to work with numbers in lists and arrays!

# Load the saved Random Forest model
# This is where we load our special model that can predict if someone might have heart disease.
model = joblib.load('heart_disease_model.pkl')

# Function to preprocess input data and make a prediction
def preprocess_and_predict(input_data):
    """
    This function takes the health information that you give us,
    cleans it up so the model can understand it, and then makes a prediction.
    Arguments:
        input_data: A table of your health information.
    Returns:
        prediction: The model's guess! (1 means "you might have a risk," 0 means "you're likely okay").
    """
    # Step 1: Fix number data so it looks neat and easy to understand for the model.
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()  # This is like a tool to make numbers more "standard."

    # Clean up the number columns using the scaler.
    input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

    # Step 2: Change words into numbers (because computers don't understand words).
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    label_encoders = {}  # This will help us remember how we changed the words.
    for col in categorical_cols:
        le = LabelEncoder()  # This is like a translator for words to numbers.
        input_data[col] = le.fit_transform(input_data[col])  # Translate!
        label_encoders[col] = le  # Save the translator in case we need it again.

    # Step 3: Let the model make a prediction using the cleaned-up data.
    prediction = model.predict(input_data)
    return prediction[0]

# Create Streamlit UI
# Streamlit helps us build a simple app for users to input their information.
st.title("Heart Disease Prediction")  # This is the title of our app.

# Collect user inputs using a sidebar
st.sidebar.header("Input Features")  # This is the title of the sidebar where users give their information.

# Get input data from the user
age = st.sidebar.slider("Age", 1, 120, 50)  # A slider lets you pick your age.
sex = st.sidebar.selectbox("Sex", options=["M", "F"])  # Pick your gender.
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])  # Pick chest pain type.
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mmHg)", 50, 200, 120)  # Enter your blood pressure.
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 50, 600, 200)  # Enter your cholesterol level.
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1])  # Did you fast? Pick yes or no.
resting_ecg = st.sidebar.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])  # Pick your resting ECG result.
max_hr = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 140)  # Enter your max heart rate.
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", options=["Yes", "No"])  # Did you feel chest pain during exercise? Pick yes or no.
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)  # Pick your oldpeak value.
st_slope = st.sidebar.selectbox("ST Slope", options=["Up", "Flat", "Down"])  # Pick the slope of the ST segment.

# Turn the text inputs into numbers so the model can understand them
# We are translating words into numbers based on what the model expects.
sex = 1 if sex == "M" else 0  # Turn "M" into 1 and "F" into 0.
exercise_angina = 1 if exercise_angina == "Yes" else 0  # Turn "Yes" into 1 and "No" into 0.
resting_ecg_mapping = {"Normal": 0, "ST": 1, "LVH": 2}  # Map ECG results to numbers.
chest_pain_mapping = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}  # Map chest pain types to numbers.
st_slope_mapping = {"Up": 0, "Flat": 1, "Down": 2}  # Map ST slope values to numbers.

# Replace text inputs with their matching numbers.
resting_ecg = resting_ecg_mapping[resting_ecg]
chest_pain_type = chest_pain_mapping[chest_pain_type]
st_slope = st_slope_mapping[st_slope]

# Make a prediction when the user clicks the button
if st.button("Predict"):
    # Combine all the user input into one list for the model.
    input_features = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                                resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
    
    # Ask the model to predict if there's heart disease risk.
    prediction = model.predict(input_features)[0]  # 1 means "risk," 0 means "no risk."
    probability = model.predict_proba(input_features)[0][1]  # How sure is the model about "risk"?

    # Show the result to the user.
    if prediction == 1:
        # If the model predicts risk, show a warning.
        st.error(f"The model predicts you have a heart disease risk with a probability of {probability:.2f}.")
    else:
        # If the model predicts no risk, show a happy message.
        st.success(f"The model predicts you have no heart disease with a probability of {1 - probability:.2f}.")
