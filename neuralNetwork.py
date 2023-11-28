import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv('encoded_heart_data.csv')

# Feature selection with Lasso
y = data['HadHeartAttack']
X = data.drop('HadHeartAttack', axis=1)
lasso = LassoCV().fit(X, y)
important_features = lasso.coef_ != 0
X_selected = X.loc[:, important_features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Print the names of selected features
selected_feature_names = X.columns[important_features]
print("Selected Features:", selected_feature_names.tolist())

# Define the model creation function
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
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

# Plotting accuracy for different hyperparameters
results = pd.DataFrame(clf.cv_results_)
plt.figure(figsize=(12, 6))
for param in parameters.keys():
    for value in parameters[param]:
        subset = results[results['param_' + param] == value]
        plt.plot(subset['mean_test_score'], label=f'{param}={value}')
    plt.title(f'Accuracy for different {param}')
    plt.xlabel('Hyperparameter combination index')
    plt.ylabel('Mean CV Accuracy')
    plt.legend()
    plt.show()

# Plotting AUC for the best model
best_model = clf.best_estimator_.model
y_pred = best_model.predict(X_scaled).ravel()
fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
