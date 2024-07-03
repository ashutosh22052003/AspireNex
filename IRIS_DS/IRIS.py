import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

iris = datasets.load_iris()
X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First five rows of data:\n", X[:5])
print("Target values:", y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('iris_classification_results.csv', index=False)

new_sample = np.array([[6.2, 3.4, 5.5, 2.2]])
new_sample = scaler.transform(new_sample)
prediction = model.predict(new_sample)
predicted_species = iris.target_names[prediction][0]
print("Predicted species for the new sample:", predicted_species)

joblib.dump(model, 'iris_knn_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
