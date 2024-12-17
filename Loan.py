import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv("loan_data.csv")

X = data.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
y = data["TARGET"]

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

#RandomForestClassifier

print("Random Forest Classifier Results:")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# Perceptron and Feed Forward neural network 
print("Perceptron Results:")
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train, y_train)
perceptron_pred = perceptron.predict(X_test)
print("Accuracy:", accuracy_score(y_test, perceptron_pred))
print("Classification Report:\n", classification_report(y_test, perceptron_pred))

print("\nFeedforward Neural Network Results:")

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer with ReLU activation
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer (sigmoid for binary classification)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Feedforward Neural Network Accuracy: {accuracy:.4f}")

y_pred_nn = (model.predict(X_test) > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred_nn))