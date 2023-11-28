import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the data
data = pd.read_csv('../encoded_heart_data.csv')

# Select the features
features = ['PhysicalHealthDays', 'MentalHealthDays', 'PhysicalActivities', 'SleepHours',
            'HadAngina', 'HadStroke', 'HadAsthma', 'HadCOPD', 'HadKidneyDisease',
            'HadArthritis', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
            'DifficultyWalking', 'ChestScan', 'WeightInKilograms', 'AlcoholDrinkers',
            'FluVaxLast12', 'PneumoVaxEver', 'Sex_Female', 'Sex_Male',
            'GeneralHealth_Excellent', 'GeneralHealth_Fair', 'GeneralHealth_Poor',
            'GeneralHealth_Very good', 'LastCheckupTime_Within past year (anytime less than 12 months ago)',
            'RemovedTeeth_1 to 5', 'RemovedTeeth_6 or more, but not all', 'RemovedTeeth_All',
            'RemovedTeeth_None of them', 'HadDiabetes_No', 'HadDiabetes_Yes',
            'SmokerStatus_Current smoker - now smokes every day', 'SmokerStatus_Never smoked',
            'ECigaretteUsage_Never used e-cigarettes in my entire life',
            'RaceEthnicityCategory_Black only, Non-Hispanic', 'AgeCategory_Age 18 to 24',
            'AgeCategory_Age 25 to 29', 'AgeCategory_Age 30 to 34', 'AgeCategory_Age 35 to 39',
            'AgeCategory_Age 40 to 44', 'AgeCategory_Age 65 to 69', 'AgeCategory_Age 70 to 74',
            'AgeCategory_Age 75 to 79', 'AgeCategory_Age 80 or older', 'TetanusLast10Tdap_Yes, received Tdap']

# Assign x and y
X = data[features]
y = data['HadHeartAttack']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the nn model
mlp = MLPClassifier(activation='relu', alpha=0.05, hidden_layer_sizes=(10,), learning_rate='constant', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# Save the model
joblib.dump(mlp, '../models/nn_model.pkl')

# Prediction function for nn
def predict_mlp(input_data):
    model = joblib.load('../models/nn_model.pkl')
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)
    return prediction[0]

# Example usage for nn
input_tuple = (5, 2, 1, 7, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 70, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1)
print(predict_mlp(input_tuple))
