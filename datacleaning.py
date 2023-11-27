import pandas as pd

# Load the dataset
file_path = 'heart_2022_no_nans.csv'
df = pd.read_csv(file_path)

def encode_categorical_variables(df):
    # Copy the dataframe to avoid modifying the original data
    encoded_df = df.copy()

    # Process each column
    for column in df.columns:
        # Check if the column is non-numeric (categorical)
        if df[column].dtype == 'object':
            # Check if the column is binary with 'Yes'/'No' values
            if sorted(df[column].unique()) == ['No', 'Yes']:
                encoded_df[column] = df[column].map({'Yes': 1, 'No': 0})
            else:
                # Apply one-hot encoding to non-binary categorical variables
                dummies = pd.get_dummies(df[column], prefix=column)
                # Drop the original column and concatenate the new one-hot encoded columns
                encoded_df = pd.concat([encoded_df.drop(column, axis=1), dummies], axis=1)

    return encoded_df

# Apply the encoding function to the dataset
encoded_df = encode_categorical_variables(df)

# Display the first few rows of the transformed dataframe
print(encoded_df.head())

# Define the file path where you want to save the dataset
output_file_path = 'encoded_heart_data.csv'

# Save the dataframe to a CSV file
encoded_df.to_csv(output_file_path, index=False)