import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

print("Loading test data...")
test_df = pd.read_csv('test.csv')

if 'CreditDefault' in test_df.columns:
    y_test = test_df['CreditDefault']
    test_df = test_df.drop('CreditDefault', axis=1)
    print(f"Dropped target variable. Test shape: {test_df.shape}")

payment_history = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

if not os.path.exists('BinaryFolder'):
    print("Error: BinaryFolder directory not found.")
    exit(1)

print("Loading scaler...")
scaler = joblib.load('BinaryFolder/scaler.pkl')

X_test_scaled = scaler.transform(test_df)

print("Loading models...")
lr_model = joblib.load('BinaryFolder/LogisticRegression_model.pkl')
rf_model = joblib.load('BinaryFolder/RandomForest_model.pkl')
gb_model = joblib.load('BinaryFolder/GradientBoosting_model.pkl')
try:
    nn_model = load_model('BinaryFolder/nn_model.h5')
except:
    print("Warning: Failed to load neural network model.")
    nn_model = None

meta_learner = joblib.load('BinaryFolder/StackedModel.pkl')

print("Making predictions with base models...")
lr_preds = lr_model.predict_proba(X_test_scaled)[:, 1]
rf_preds = rf_model.predict_proba(X_test_scaled)[:, 1]
gb_preds = gb_model.predict_proba(X_test_scaled)[:, 1]

if nn_model:
    nn_preds = (nn_model.predict(X_test_scaled) > 0.5).astype(int).reshape(-1)
else:
    nn_preds = np.zeros(X_test_scaled.shape[0])

X_test_meta = np.column_stack([
    lr_preds,
    rf_preds,
    gb_preds,
    nn_preds
])

print("Making predictions with stacked model...")
stacked_preds = meta_learner.predict(X_test_meta)

pd.DataFrame({'CreditDefault': stacked_preds}).to_csv('CreditDefault_Predictions.csv', index=False)
print(f"Saved predictions to CreditDefault_Predictions.csv")

print("Production script completed successfully.")