import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier

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

# Hyperparameter tuning with k-Fold Cross-Validation
parameters = {'n_neighbors': range(1, 31), 'metric': ['euclidean', 'manhattan']}
knn = KNeighborsClassifier()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
clf = GridSearchCV(knn, parameters, cv=kfold)
clf.fit(X_scaled, y)

# Best parameters and model evaluation
best_knn = clf.best_estimator_
print("Best Parameters:", clf.best_params_)
