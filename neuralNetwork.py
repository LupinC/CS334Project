import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv('train.csv')

# Feature selection with Lasso
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
lasso = LassoCV().fit(X, y)
important_features = lasso.coef_ != 0
X_selected = X.loc[:, important_features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Define the model creation function
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_dim=X_scaled.shape[1], activation=activation))
        else:
            model.add(Dense(nodes, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Set up k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV for hyperparameter tuning
model = KerasClassifier(build_fn=create_model, verbose=0)
parameters = {
    'batch_size': [16, 32],
    'epochs': [50, 100],
    'layers': [(50,), (50, 30), (30, 20, 10)],
    'activation': ['sigmoid', 'tanh', 'relu', 'softplus']
}
clf = GridSearchCV(model, parameters, cv=kfold)
clf.fit(X_scaled, y)

# Best parameters and model evaluation
print("Best Parameters:", clf.best_params_)
