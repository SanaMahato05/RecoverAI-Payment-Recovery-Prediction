import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/transactions.csv')
print(data.head())
print(data.shape)
print(data.columns)

encoder = LabelEncoder()
data["payment_method"] = encoder.fit_transform(data["payment_method"])
data["failure_reason"] = encoder.fit_transform(data["failure_reason"])
data["recovery_status"] = encoder.fit_transform(data["recovery_status"])
print(data.head())

X = data[
    [
        "amount",
        "payment_method",
        "failure_reason",
        "previous_failures"
    ]
]
y = data["recovery_status"]
print("\nInput Features:")
print(X.head())
print("\nTarget :")
print(y.head())

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining data:", X_train.shape)
print("\nTesting data:", X_test.shape)

#Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
#Train the model using the traininng data
model.fit(X_train, y_train)
print("\nModel trained successfully!")

#Make prediction using unseen test data
predictions = model.predict(X_test)

#Calculate model accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

results = pd.DataFrame({
    "Actual" : y_test,
    "Predicted" : predictions,
})
print("\nActual Vs Predicted:")
print(results.head())

#Save the trained ML model
joblib.dump(model, "recovery_model.pkl")
print("\nModel saved successfully!")