import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('encoded_heart_data.csv')

# Feature selection with Lasso
y = data['HadHeartAttack']
X = data.drop('HadHeartAttack', axis=1)
lasso = LassoCV().fit(X, y)
important_features = lasso.coef_ != 0
X_selected = X.loc[:, important_features]

selected_feature_names = X.columns[important_features]
print("Selected Features:", selected_feature_names.tolist())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Set up k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 11),
    'min_samples_leaf': range(1, 6)
}

# Initialize the Decision Tree Classifier
dtree = DecisionTreeClassifier()

# GridSearchCV for hyperparameter tuning
clf = GridSearchCV(dtree, param_grid, cv=kfold, scoring='accuracy')
clf.fit(X_train, y_train)

# Plotting accuracy for different hyperparameter combinations
scores = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']

# Creating a DataFrame to plot
df = pd.DataFrame(parameters)
df['Accuracy'] = scores

plt.figure(figsize=(10, 6))
for criterion in ['gini', 'entropy']:
    subset = df[df['criterion'] == criterion]
    plt.plot(subset['max_depth'], subset['Accuracy'], label=f'Criterion: {criterion}')

plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy for Different Max Depths and Criteria')
plt.legend()
plt.grid(True)
plt.show()
# Creating a DataFrame to plot
df = pd.DataFrame(parameters)
df['Accuracy'] = scores

# Plotting accuracy for different 'min_samples_leaf'
plt.figure(figsize=(12, 8))

# Plotting for each criterion
for criterion in ['gini', 'entropy']:
    # Filter the DataFrame for each criterion and aggregate by min_samples_leaf
    subset = df[df['criterion'] == criterion].groupby('min_samples_leaf').max()
    plt.plot(subset.index, subset['Accuracy'], label=f'Criterion: {criterion}')

plt.xlabel('Min Samples Leaf')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy for Different Min Samples Leaf, Criteria')
plt.legend()
plt.grid(True)
plt.show()

# Best parameters and model evaluation
best_dtree = clf.best_estimator_
print("Best Parameters:", clf.best_params_)

# Plotting the AUC graph for the best hyperparameters
y_pred_proba = best_dtree.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Best Decision Tree Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
