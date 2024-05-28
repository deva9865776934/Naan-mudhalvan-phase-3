import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import tensorflow as tf

st.title('Income Prediction using ML and Neural Networks')

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
               "hours-per-week", "native-country", "income"]
    data = pd.read_csv(url, header=None, names=columns, na_values=' ?')
    data.dropna(inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    return data

data = load_data()

st.write("Data Preview:")
st.write(data.head())

X = data.drop('income_ >50K', axis=1)
y = data['income_ >50K']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.write("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

st.write("Random Forest Model Results:")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
st.write(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")
st.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")

# Convert data to numpy arrays for TensorFlow
X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

st.write("Training TensorFlow model...")
tf_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reduce verbosity of the fit function
history = tf_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

st.write("TensorFlow Model Results:")
accuracy = tf_model.evaluate(X_test, y_test, verbose=0)[1]
y_pred_tf = (tf_model.predict(X_test, verbose=0) > 0.5).astype("int32")

st.write(f"Accuracy: {accuracy}")
st.write(f"Classification Report:\n{classification_report(y_test, y_pred_tf)}")
st.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_tf)}")
