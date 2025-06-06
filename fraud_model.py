# Train fraud classification model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("delivery_data.csv")
df['pickup_time'] = pd.to_datetime(df['pickup_time'])
df['delivery_time'] = pd.to_datetime(df['delivery_time'])
df['dropzone_time'] = pd.to_datetime(df['dropzone_time'])

df['delivery_duration_min'] = (df['delivery_time'] - df['pickup_time']).dt.total_seconds() / 60
df['dropzone_lag_min'] = (df['delivery_time'] - df['dropzone_time']).dt.total_seconds() / 60
df['delivery_speed_kmph'] = df['distance_km'] / (df['delivery_duration_min'] / 60)

features = ['delivery_duration_min', 'dropzone_lag_min', 'delivery_speed_kmph', 'distance_km']
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(method='bfill')
y = df['is_fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
