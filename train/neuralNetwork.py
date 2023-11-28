import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load data
data = pd.read_csv('../encoded_heart_data.csv')

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

# Set up k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV for hyperparameter tuning
parameters = {
    'hidden_layer_sizes': [(5,), (10,), (5, 2)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier(max_iter=100, random_state=42)
clf = GridSearchCV(mlp, parameters, cv=kfold)
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
y_pred = clf.predict_proba(X_scaled)[:, 1]
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
