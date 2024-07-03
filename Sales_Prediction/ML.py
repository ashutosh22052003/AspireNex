import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from scipy.stats import randint

file_path = '/Users/ashutoshkumar/Downloads/IMDb Movies India.csv'
movies_df = pd.read_csv(file_path, encoding='ISO-8859-1')

movies_df = movies_df.dropna(subset=['Name', 'Year', 'Genre', 'Rating'])

movies_df['Year'] = movies_df['Year'].str.extract(r'(\d{4})').astype(float)

movies_df['Duration'] = movies_df['Duration'].str.extract(r'(\d+)').astype(float)

movies_df['Votes'] = movies_df['Votes'].str.replace(',', '').astype(float)

movies_df = movies_df.dropna()

simplified_df = movies_df[['Year', 'Duration', 'Votes', 'Rating']]

X = simplified_df.drop('Rating', axis=1)
y = simplified_df['Rating']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

rf_model.fit(X_train_poly, y_train_poly)
y_pred_rf = rf_model.predict(X_test_poly)

gb_model.fit(X_train_poly, y_train_poly)
y_pred_gb = gb_model.predict(X_test_poly)

mse_rf = mean_squared_error(y_test_poly, y_pred_rf)
mae_rf = mean_absolute_error(y_test_poly, y_pred_rf)
r2_rf = r2_score(y_test_poly, y_pred_rf)

mse_gb = mean_squared_error(y_test_poly, y_pred_gb)
mae_gb = mean_absolute_error(y_test_poly, y_pred_gb)
r2_gb = r2_score(y_test_poly, y_pred_gb)

print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest Mean Absolute Error: {mae_rf}')
print(f'Random Forest R-squared: {r2_rf}')

print(f'Gradient Boosting Mean Squared Error: {mse_gb}')
print(f'Gradient Boosting Mean Absolute Error: {mae_gb}')
print(f'Gradient Boosting R-squared: {r2_gb}')

joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')

rf_param_grid = {
    'n_estimators': randint(100, 1000),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

gb_param_grid = {
    'n_estimators': randint(100, 1000),
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

gb_random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=gb_param_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_random_search.fit(X_train_poly, y_train_poly)
gb_random_search.fit(X_train_poly, y_train_poly)

rf_best_params = rf_random_search.best_params_
gb_best_params = gb_random_search.best_params_

print(f'Best parameters for Random Forest: {rf_best_params}')
print(f'Best parameters for Gradient Boosting: {gb_best_params}')

rf_best_model = rf_random_search.best_estimator_
gb_best_model = gb_random_search.best_estimator_

y_pred_rf_best = rf_best_model.predict(X_test_poly)
y_pred_gb_best = gb_best_model.predict(X_test_poly)

mse_rf_best = mean_squared_error(y_test_poly, y_pred_rf_best)
mae_rf_best = mean_absolute_error(y_test_poly, y_pred_rf_best)
r2_rf_best = r2_score(y_test_poly, y_pred_rf_best)

mse_gb_best = mean_squared_error(y_test_poly, y_pred_gb_best)
mae_gb_best = mean_absolute_error(y_test_poly, y_pred_gb_best)
r2_gb_best = r2_score(y_test_poly, y_pred_gb_best)

print(f'Tuned Random Forest Mean Squared Error: {mse_rf_best}')
print(f'Tuned Random Forest Mean Absolute Error: {mae_rf_best}')
print(f'Tuned Random Forest R-squared: {r2_rf_best}')

print(f'Tuned Gradient Boosting Mean Squared Error: {mse_gb_best}')
print(f'Tuned Gradient Boosting Mean Absolute Error: {mae_gb_best}')
print(f'Tuned Gradient Boosting R-squared: {r2_gb_best}')

joblib.dump(rf_best_model, 'tuned_random_forest_model.pkl')
joblib.dump(gb_best_model, 'tuned_gradient_boosting_model.pkl')

residuals_gb = y_test_poly - y_pred_gb
plt.figure(figsize=(10, 6))
sns.histplot(residuals_gb, kde=True)
plt.title('Residuals Distribution (Gradient Boosting)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_gb, residuals_gb)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Gradient Boosting)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
