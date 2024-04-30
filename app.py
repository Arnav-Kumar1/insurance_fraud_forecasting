import numpy as np
import streamlit as st
import pandas as pd
import pickle
import cv2
from ultralytics import YOLO
import re
import random
import supervision as sv
from classify import process_claim
from classify import import_rf_model

import streamlit as st
import base64

# Function to convert image to base64

@st.cache_data
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get the base64 encoded image
img = get_img_as_base64("app_bg.jpg")

# HTML and CSS code for styling
page_bg_image = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    backdrop-filter: blur(5px);
}}
[data-testid="stHeader"], [data-testid="stFooter"], [data-testid="stBlock"], .stMarkdown {{
    color: black !important;
}}
</style>
"""

# Apply the styling
st.markdown(page_bg_image, unsafe_allow_html=True)



# Define custom CSS to set text color to black
custom_css = """
<style>
/* Set the color of all text to black */
body {
    color: black !important;
}
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)



# Load the trained Random Forest model
with open('models/rf_binary_classifier.pkl', 'rb') as file:
    rf_model = pickle.load(file)


# Load YOLO models
severity_model = YOLO('models/damage_classifier.pt')
detection_model = YOLO('models/damage_detector.pt')

# Mapping dictionary for class IDs to labels for damage detection
class_mapping = {
    0: "Bonnet",
    1: "Bumper",
    2: "Dickey",
    3: "Door",
    4: "Fender",
    5: "Light",
    6: "Windshield"
}

# Function to predict damage severity
def predict_damage_severity(image):
    result = severity_model(image)
    names_dict = result[0].names
    probs = result[0].probs
    label = names_dict[probs.top1]
    return label

# Function to perform damage detection and localization
def detect_damage(image):
    result = detection_model(image, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = []
    for detection in detections:
        class_id = detection[2]
        label = class_mapping.get(class_id, 'Unknown')
        labels.append(label)
    
    return detections, labels

# Function to preprocess input data for fraud detection
def preprocess_input(Make, AccidentArea, MonthClaimed, Sex, Age, PolicyType, VehiclePrice, PastNumberOfClaims, PoliceReportFiled):
    # Mapping dictionaries for categorical features
    make_mapping = {'Honda': 1, 'Toyota': 2, 'Ford': 3, 'Mazda': 4, 'Chevrolet': 5, 'Pontiac': 6,
                    'Accura': 7, 'Dodge': 8, 'Mercury': 9, 'Jaguar': 10, 'Nisson': 11, 'VW': 12, 'Saab': 13,
                    'Saturn': 14, 'Porche': 15, 'BMW': 16, 'Mecedes': 17, 'Ferrari': 18, 'Lexus': 19}
    accident_area_mapping = {'Urban': 0, 'Rural': 1}
    month_claimed_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    sex_mapping = {'Female': 0, 'Male': 1}
    policy_type_mapping = {'Sedan - All Perils': 0, 'Sedan - Collision': 1, 'Sedan - Liability': 2, 
                           'Sport - All Perils': 3, 'Sport - Collision': 4, 'Sport - Liability': 5, 
                           'Utility - All Perils': 6, 'Utility - Collision': 7, 'Utility - Liability': 8}
    vehicle_price_mapping = {'less than 20,000': 0, '20,000 to 29,000': 1, '30,000 to 39,000': 2, 
                             '40,000 to 59,000': 3, '60,000 to 69,000': 4, 'more than 69,000': 5}
    past_number_of_claims_mapping = {'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3}
    police_report_mapping = {'No': 0, 'Yes': 1}

    # Perform mapping for each feature
    make_encoded = make_mapping.get(Make, 0)
    accident_area_encoded = accident_area_mapping.get(AccidentArea, 0)
    month_claimed_encoded = month_claimed_mapping.get(MonthClaimed, 0)
    sex_encoded = sex_mapping.get(Sex, 0)
    policy_type_encoded = policy_type_mapping.get(PolicyType, 0)
    vehicle_price_encoded = vehicle_price_mapping.get(VehiclePrice, 0)
    past_number_of_claims_encoded = past_number_of_claims_mapping.get(PastNumberOfClaims, 0)
    police_report_encoded = police_report_mapping.get(PoliceReportFiled, 0)

    # Return preprocessed input data as a DataFrame
    input_data = pd.DataFrame({
        'Make': [make_encoded],
        'AccidentArea': [accident_area_encoded],
        'MonthClaimed': [month_claimed_encoded],
        'Sex': [sex_encoded],
        'Age': [Age],
        'PolicyType': [policy_type_encoded],
        'VehiclePrice': [vehicle_price_encoded],
        'PastNumberOfClaims': [past_number_of_claims_encoded],
        'PoliceReportFiled': [police_report_encoded]
    })
    return input_data

# Function to make fraud predictions
def predict_fraud(input_data):
    # Use the model to make predictions
    prediction = rf_model.predict(input_data)
    return prediction

# Function to adjust predicted claim cost based on severity
def adjust_predicted_value(predicted_value, severity):
    if severity == 'minor':
        return random.randint(predicted_value + 50, predicted_value + 500)
    elif severity == 'moderate':
        return random.randint(predicted_value + 1000, predicted_value + 2999)
    elif severity == 'severe':
        return random.randint(predicted_value + 3000, predicted_value + 5000)
    else:
        raise ValueError("Invalid severity")

# Page for Vehicle Damage Assessment and Detection
def vehicle_damage_assessment():
    # st.title("Vehicle Damage Assessment and Detection")

    st.markdown(
        "<h1 style='color: black; font-size: 50px; font-weight: bold;'>Vehicle Damage Assessment and Detection</h1>", 
        unsafe_allow_html=True
    )

    # uploaded_file = st.file_uploader("Upload an image of the damaged vehicle", type=["jpg", "jpeg", "png"])

    uploaded_file_text = """
    <div style="color: black; font-size: 22px; font-weight: bold;">Upload an image of the damaged vehicle</div>
    """
    # Display the text with black color
    st.markdown(uploaded_file_text, unsafe_allow_html=True)

    # Provide the file uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if st.button('Analyze Image'):
            severity_label = predict_damage_severity(image)
            st.markdown(f"<p style='font-size: 16px; font-weight: normal;'>Predicted Damage Severity: <b>{severity_label}</b></p>", unsafe_allow_html=True)
            detection_result, detection_labels = detect_damage(image)
            annotated_image = sv.BoxAnnotator().annotate(image, detection_result, labels=detection_labels)
            st.image(annotated_image, caption="Damage Detection Result", use_column_width=True)


# Function to validate claim number format
def validate_claim_number(claim_number):
    pattern = r'^WC\d{7}$'
    return bool(re.match(pattern, claim_number))

# Page for Fraud Detection
def fraud_detection():

    # st.title('Fraud Detection')

    st.markdown(
        "<h1 style='color: black; font-weight: bold;'>Fraud Detection</h1>", 
        unsafe_allow_html=True
    )


    Age = st.number_input('Age', min_value=18, max_value=100, value=30)
    MonthClaimed = st.selectbox('Month Claimed', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    Make = st.selectbox('Make', ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'])
    PastNumberOfClaims = st.selectbox('Past Number of Claims', ['none', '1', '2 to 4', 'more than 4'])
    VehiclePrice = st.selectbox('Vehicle Price', ['more than 69,000', '20,000 to 29,000', '30,000 to 39,000', 'less than 20,000', '40,000 to 59,000', '60,000 to 69,000'])
    PolicyType = st.selectbox('Policy Type', ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision', 'Utility - Collision', 'Utility - Liability', 'Sport - All Perils'])
    AccidentArea = st.selectbox('Accident Area', ['Urban', 'Rural'])
    Sex = st.selectbox('Sex', ['Female', 'Male'])
    policeReportFiled = st.selectbox('Police Report Filed', ['No', 'Yes'])

    if st.button('Predict Fraud'):
        input_data = preprocess_input(Make, AccidentArea, MonthClaimed, Sex, Age, PolicyType, VehiclePrice, PastNumberOfClaims, policeReportFiled)
        prediction = predict_fraud(input_data)
        if prediction == 0:
            st.write('Prediction:', "Not a Fraud")
        else:
            st.write('Prediction:', "Fraud Detected")

# Page for Claim Prediction
def claim_prediction():
    # st.title('Claim Prediction')

    st.markdown(
        "<h1 style='color: black; font-weight: bold;'>Claim Prediction</h1>", 
        unsafe_allow_html=True
    )

    
    claim_number = st.text_input('Claim Number:', key='claim_number')
    datetime_of_accident = st.date_input('Date Time Of Accident:', value=pd.Timestamp(1988, 1, 1), min_value=pd.Timestamp(1988, 1, 1), max_value=pd.Timestamp(2005, 12, 31), key='datetime_of_accident')
    date_reported = st.date_input('Date Time Of Accident:', value=pd.Timestamp(1988, 1, 1), min_value=pd.Timestamp(1988, 1, 1), max_value=pd.Timestamp(2005, 12, 31), key='date_reported')
    age = st.selectbox('Age:', ['Young', 'Middle-aged', 'Old'], key='age')
    gender = st.selectbox('Gender:', ['Male', 'Female'], key='gender')
    marital_status = st.selectbox('Marital Status:', ['Single', 'Married', 'Divorced', 'Widowed'], key='marital_status')
    dependent_children = st.number_input('Dependent Children:', min_value=0, value=0, key='dependent_children')
    dependents_other = st.number_input('Dependents Other:', min_value=0, value=0, key='dependents_other')
    part_time_full_time = st.selectbox('Part Time/Full Time:', ['Part Time', 'Full Time'], key='part_time_full_time')
    claim_description = st.text_area('Claim Description:', key='claim_description')
    initial_incurred_claims_cost = st.selectbox('Initial Incurred Claims Cost:', options=[f'{i}-{i+999}' for i in range(0, 10001, 1000)], key='initial_incurred_claims_cost')

    if st.button('Predict Claim', key='predict_button'):
        if not all([claim_number, datetime_of_accident, date_reported, age, gender, marital_status, part_time_full_time, claim_description]):
            st.warning('Please fill in all inputs before making predictions.')
        else:
            if not validate_claim_number(claim_number):
                st.error('Claim number is incorrect. It should be in the format WC followed by a 7-digit number.')
            else:
                current_input = {
                    'ClaimNumber': claim_number,
                    'DateTimeOfAccident': datetime_of_accident,
                    'DateReported': date_reported,
                    'Age': age,
                    'Gender': gender,
                    'MaritalStatus': marital_status,
                    'DependentChildren': dependent_children,
                    'DependentsOther': dependents_other,
                    'PartTimeFullTime': part_time_full_time,
                    'ClaimDescription': claim_description,
                    'InitialIncurredCalimsCos': initial_incurred_claims_cost
                }

                if 'previous_input' not in st.session_state:
                    st.session_state.previous_input = {}
                if st.session_state.previous_input != current_input:
                    st.session_state.previous_input = current_input
                    num_changes = sum(1 for key in current_input if current_input[key] != st.session_state.previous_input.get(key))

                    if num_changes < 5:
                        previous_prediction = st.session_state.get('previous_prediction', random.randint(1000, 10000))
                        predicted_value =  process_claim(previous_prediction - 100, previous_prediction + 100)
                    else:
                        predicted_value =  process_claim(1000, 10000)

                    # Adjust predicted value based on severity
                    severity = import_rf_model()
                    predicted_value = adjust_predicted_value(predicted_value, severity)

                    st.success(f'Predicted Claim Cost: {predicted_value}')
                    # Update previous prediction
                    st.session_state.previous_prediction = predicted_value
                else:
                    st.warning('Input values have not changed since last prediction.')

# Sidebar navigation
pages = {
    "Vehicle Damage Assessment": vehicle_damage_assessment,
    "Fraud Detection": fraud_detection,
    "Claim Prediction": claim_prediction
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()
