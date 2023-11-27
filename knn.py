import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier
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

# Print the names of selected features
selected_feature_names = X.columns[important_features]
print("Selected Features:", selected_feature_names.tolist())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Hyperparameter tuning with k-Fold Cross-Validation
parameters = {'n_neighbors': range(1, 31), 'metric': ['euclidean', 'manhattan']}
knn = KNeighborsClassifier()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
clf = GridSearchCV(knn, parameters, cv=kfold)
clf.fit(X_train, y_train)

# Extracting results for plotting
parameters = clf.cv_results_['params']
mean_scores = clf.cv_results_['mean_test_score']
euclidean_scores = [mean_scores[i] for i in range(len(mean_scores)) if parameters[i]['metric'] == 'euclidean']
manhattan_scores = [mean_scores[i] for i in range(len(mean_scores)) if parameters[i]['metric'] == 'manhattan']
k_values = range(1, 31)

# Plotting accuracy for different values of k and metrics
plt.figure(figsize=(10, 6))
plt.plot(k_values, euclidean_scores, label='Euclidean')
plt.plot(k_values, manhattan_scores, label='Manhattan')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy for different values of k in kNN with Euclidean and Manhattan metrics')
plt.legend()
plt.grid(True)
plt.show()

# Best parameters and model evaluation
best_knn = clf.best_estimator_
print("Best Parameters:", clf.best_params_)

# Plotting the AUC graph for the best k
y_pred_proba = best_knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Best kNN Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
