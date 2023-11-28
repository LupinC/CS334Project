import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('models/knn_model.pkl')
model2 = joblib.load('models/decision_tree_model.pkl')
model3 = joblib.load('models/nn_model.pkl')

# Streamlit webpage title
st.title('Health Prediction Tool')

# Create input fields for all the features
physical_health_days = st.number_input('Physical Health Days', min_value=0, max_value=31, value=0, step=1)
mental_health_days = st.number_input('Mental Health Days', min_value=0, max_value=31, value=0, step=1)
sleep_hours = st.number_input('Sleep Hours', min_value=0, max_value=24, value=0, step=1)
weight_in_kilograms = st.number_input('Weight in Kilograms', min_value=0.0, max_value=200.0, value=0.0, step=0.1)
had_angina = st.selectbox('Had Angina', options=[0, 1])
had_stroke = st.selectbox('Had Stroke', options=[0, 1])
had_asthma = st.selectbox('Had Asthma', options=[0, 1])
had_copd = st.selectbox('Had COPD', options=[0, 1])
had_kidney_disease = st.selectbox('Had Kidney Disease', options=[0, 1])
had_arthritis = st.selectbox('Had Arthritis', options=[0, 1])
deaf_or_hard_of_hearing = st.selectbox('Deaf or Hard of Hearing', options=[0, 1])
blind_or_vision_difficulty = st.selectbox('Blind or Vision Difficulty', options=[0, 1])
difficulty_walking = st.selectbox('Difficulty Walking', options=[0, 1])
chest_scan = st.selectbox('Chest Scan', options=[0, 1])
alcohol_drinkers = st.selectbox('Alcohol Drinkers', options=[0, 1])
flu_vax_last_12 = st.selectbox('Flu Vaccine Last 12 Months', options=[0, 1])
pneumo_vax_ever = st.selectbox('Ever Had Pneumonia Vaccine', options=[0, 1])
sex_female = st.selectbox('Sex_Female', options=[0, 1])
sex_male = st.selectbox('Sex_Male', options=[0, 1])
general_health_excellent = st.selectbox('GeneralHealth_Excellent', options=[0, 1])
general_health_fair = st.selectbox('GeneralHealth_Fair', options=[0, 1])
general_health_poor = st.selectbox('GeneralHealth_Poor', options=[0, 1])
general_health_very_good = st.selectbox('GeneralHealth_Very good', options=[0, 1])
last_checkup_time_within_year = st.selectbox('LastCheckupTime_Within past year', options=[0, 1])
removed_teeth_1_to_5 = st.selectbox('RemovedTeeth_1 to 5', options=[0, 1])
removed_teeth_6_or_more = st.selectbox('RemovedTeeth_6 or more, but not all', options=[0, 1])
removed_teeth_all = st.selectbox('RemovedTeeth_All', options=[0, 1])
removed_teeth_none = st.selectbox('RemovedTeeth_None of them', options=[0, 1])
had_diabetes_no = st.selectbox('HadDiabetes_No', options=[0, 1])
had_diabetes_yes = st.selectbox('HadDiabetes_Yes', options=[0, 1])
smoker_status_current_smoker = st.selectbox('SmokerStatus_Current smoker - now smokes every day', options=[0, 1])
smoker_status_never_smoked = st.selectbox('SmokerStatus_Never smoked', options=[0, 1])
e_cigarette_never_used = st.selectbox('ECigaretteUsage_Never used e-cigarettes in my entire life', options=[0, 1])
race_ethnicity_black_non_hispanic = st.selectbox('RaceEthnicityCategory_Black only, Non-Hispanic', options=[0, 1])
age_category_18_to_24 = st.selectbox('AgeCategory_Age 18 to 24', options=[0, 1])
age_category_25_to_29 = st.selectbox('AgeCategory_Age 25 to 29', options=[0, 1])
age_category_30_to_34 = st.selectbox('AgeCategory_Age 30 to 34', options=[0, 1])
age_category_35_to_39 = st.selectbox('AgeCategory_Age 35 to 39', options=[0, 1])
age_category_40_to_44 = st.selectbox('AgeCategory_Age 40 to 44', options=[0, 1])
age_category_65_to_69 = st.selectbox('AgeCategory_Age 65 to 69', options=[0, 1])
age_category_70_to_74 = st.selectbox('AgeCategory_Age 70 to 74', options=[0, 1])
age_category_75_to_79 = st.selectbox('AgeCategory_Age 75 to 79', options=[0, 1])
age_category_80_or_older = st.selectbox('AgeCategory_Age 80 or older', options=[0, 1])
tetanus_last_10_tdap = st.selectbox('TetanusLast10Tdap_Yes', options=[0, 1])
received_tdap = st.selectbox('received Tdap', options=[0, 1])

# Prediction button
if st.button('Predict'):
    # Arrange input data in the same order as the training data
    input_data = [physical_health_days, mental_health_days, sleep_hours, weight_in_kilograms, had_angina, had_stroke, had_asthma, had_copd, had_kidney_disease, had_arthritis, deaf_or_hard_of_hearing, blind_or_vision_difficulty, difficulty_walking, chest_scan,alcohol_drinkers, flu_vax_last_12, pneumo_vax_ever, sex_female, sex_male, general_health_excellent,general_health_fair,general_health_poor, general_health_very_good, last_checkup_time_within_year, removed_teeth_1_to_5, removed_teeth_6_or_more, removed_teeth_all,removed_teeth_none, had_diabetes_no, had_diabetes_yes, smoker_status_current_smoker, smoker_status_never_smoked, e_cigarette_never_used, race_ethnicity_black_non_hispanic, age_category_18_to_24, age_category_25_to_29, age_category_30_to_34,age_category_35_to_39, age_category_40_to_44, age_category_65_to_69, age_category_70_to_74, age_category_75_to_79, age_category_80_or_older, tetanus_last_10_tdap, received_tdap]

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

    # Make a prediction
    prediction = model.predict(input_df)
    prediction2 = model2.predict(input_df)
    prediction3 = model3.predict(input_df)

    # Display the prediction
    if prediction[0] + prediction2[0] + prediction3[0] >= 1.5:
        st.success('Yes')
    else:
        st.error('No')

