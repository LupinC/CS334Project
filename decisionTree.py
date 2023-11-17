import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

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
clf.fit(X_scaled, y)

# Best parameters and model evaluation
best_dtree = clf.best_estimator_
print("Best Parameters:", clf.best_params_)

#clf.predict(X_test)
