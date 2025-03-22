import pandas as pd
import numpy as np
import os
import joblib
import random
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load the data
df = pd.read_csv('CreditDefault.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check target variable distribution
print("\nTarget variable distribution:")
print(df['CreditDefault'].value_counts())
print(df['CreditDefault'].value_counts(normalize=True))

# Explore categorical features
print("\nCategorical feature distributions:")
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
for feature in categorical_features:
    print(f"\n{feature} distribution:")
    print(df[feature].value_counts())

# Check the payment history variables
payment_history = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for feature in payment_history:
    print(f"\n{feature} distribution:")
    print(df[feature].value_counts())

# Define bill and payment columns
bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Data preparation and feature engineering
print("\nPerforming data preparation and feature engineering...")

# Drop ID column as it's not useful for prediction
df = df.drop('ID', axis=1)

# Create feature engineering
# Payment history features have correlation with default
df['TOTAL_BILL'] = df[bill_columns].sum(axis=1)
df['TOTAL_PAY'] = df[pay_columns].sum(axis=1)
df['PAYMENT_RATIO'] = df['TOTAL_PAY'] / df['TOTAL_BILL'].replace(0, 0.01)
df['AVG_BILL_AMT'] = df[bill_columns].mean(axis=1)
df['AVG_PAY_AMT'] = df[pay_columns].mean(axis=1)

# Late payment count (positive values in PAY_* indicate payment delay)
df['LATE_PAYMENT_COUNT'] = df[payment_history].apply(lambda x: sum(x > 0), axis=1)

# Define X and y
X = df.drop('CreditDefault', axis=1)
y = df['CreditDefault']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Save the test data to csv for Production.py
test_df = pd.concat([X_test, y_test], axis=1)
train_df = pd.concat([X_train, y_train], axis=1)

# Create directory for saving models
if not os.path.exists('BinaryFolder'):
    os.makedirs('BinaryFolder')

# Save the datasets
test_df.to_csv('test.csv', index=False)
train_df.to_csv('train.csv', index=False)

# Now split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'BinaryFolder/scaler.pkl')

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Now we'll add code to build models
print("\nPreparing to build models...")

# Function to evaluate models with cross-validation
def evaluate_model(model, X, y, cv=5):
    # Perform k-fold cross-validation
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='f1')

    # Print results
    print(f"F1 Score: {cv_results.mean():.4f} ({cv_results.std():.4f})")
    return cv_results.mean(), cv_results.std()

# Base models
models = {}
model_scores = {}

print("\nTraining and evaluating base models...")

# Logistic Regression
print("\nLogistic Regression:")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
lr_f1_mean, lr_f1_std = evaluate_model(lr, X_val_scaled, y_val)
models['LogisticRegression'] = lr
model_scores['LogisticRegression'] = (lr_f1_mean, lr_f1_std)

# Random Forest
print("\nRandom Forest:")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, y_train)
rf_f1_mean, rf_f1_std = evaluate_model(rf, X_val_scaled, y_val)
models['RandomForest'] = rf
model_scores['RandomForest'] = (rf_f1_mean, rf_f1_std)

# Gradient Boosting
print("\nGradient Boosting:")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_f1_mean, gb_f1_std = evaluate_model(gb, X_val_scaled, y_val)
models['GradientBoosting'] = gb
model_scores['GradientBoosting'] = (gb_f1_mean, gb_f1_std)

# Save the base models
for name, model in models.items():
    joblib.dump(model, f'BinaryFolder/{name}_model.pkl')
    print(f"Saved {name} model")

# Neural Network model
print("\nBuilding Neural Network model...")

# Function to create a neural network model
def create_nn_model(optimizer='adam', nodes1=64, nodes2=32, dropout_rate=0.2, activation='relu', kernel_initializer='he_uniform'):
    inputs = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(nodes1, activation=activation, kernel_initializer=kernel_initializer)(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(nodes2, activation=activation, kernel_initializer=kernel_initializer)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation=activation, kernel_initializer=kernel_initializer)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Tune hyperparameters - IMPROVED VERSION
print("\nPerforming hyperparameter tuning for neural network...")

# Reduced hyperparameter search space
optimizers = ['adam']  # Just use the best optimizer
nodes1_options = [32, 64]
nodes2_options = [16, 32]
dropout_rates = [0.1, 0.3]
activations = ['relu', 'elu']
kernel_initializers = ['he_uniform', 'glorot_uniform']

# Generate all possible combinations
param_combinations = []
for optimizer in optimizers:
    for nodes1 in nodes1_options:
        for nodes2 in nodes2_options:
            for dropout_rate in dropout_rates:
                for activation in activations:
                    for kernel_initializer in kernel_initializers:
                        param_combinations.append({
                            'optimizer': optimizer,
                            'nodes1': nodes1,
                            'nodes2': nodes2,
                            'dropout_rate': dropout_rate,
                            'activation': activation,
                            'kernel_initializer': kernel_initializer
                        })

# Choose a subset of parameter combinations
# Use all combinations if there are 20 or fewer
sampled_combinations = random.sample(param_combinations, min(20, len(param_combinations)))

# Initialize tracking variables
best_f1 = 0
best_params = {}

# Try sampled parameter combinations
for params in sampled_combinations:
    # Create and compile model
    model = create_nn_model(**params)

    # Early stopping with less patience
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Reduced from 10
        verbose=0,
        restore_best_weights=True
    )

    # Train the model with fewer epochs
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=25,  # Reduced from 50
        batch_size=128,  # Increased from 64
        callbacks=[early_stopping],
        verbose=0
    )

    # Evaluate the model
    y_pred = (model.predict(X_val_scaled) > 0.5).astype(int).reshape(-1)
    current_f1 = f1_score(y_val, y_pred)

    # Track best parameters
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params = params

    print(f"Params: {params['optimizer']}, {params['nodes1']}, {params['nodes2']}, {params['dropout_rate']}, {params['activation']}, {params['kernel_initializer']} - F1: {current_f1:.4f}")

print(f"\nBest Neural Network parameters: {best_params}")
print(f"Best F1 Score: {best_f1:.4f}")

# Train the final Neural Network model with best parameters
print("\nTraining final Neural Network model with best parameters...")
final_nn_model = create_nn_model(**best_params)

# Create model checkpoint callback
checkpoint_path = "BinaryFolder/nn_model.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train with early stopping and checkpoint
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    verbose=1,
    restore_best_weights=True
)

history = final_nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=128,  # Increased batch size
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Evaluate the final NN model
y_pred_nn = (final_nn_model.predict(X_val_scaled) > 0.5).astype(int).reshape(-1)
nn_f1 = f1_score(y_val, y_pred_nn)
print("\nNeural Network classification report:")
print(classification_report(y_val, y_pred_nn))
models['NeuralNetwork'] = final_nn_model
model_scores['NeuralNetwork'] = (nn_f1, 0)  # No std deviation since we didn't use cross-validation

# Build a stacked model
print("\nBuilding stacked model...")
# Get predictions from base models
X_train_meta = np.column_stack([
    models['LogisticRegression'].predict_proba(X_train_scaled)[:, 1],
    models['RandomForest'].predict_proba(X_train_scaled)[:, 1],
    models['GradientBoosting'].predict_proba(X_train_scaled)[:, 1],
    (models['NeuralNetwork'].predict(X_train_scaled) > 0.5).astype(int).reshape(-1)
])

X_val_meta = np.column_stack([
    models['LogisticRegression'].predict_proba(X_val_scaled)[:, 1],
    models['RandomForest'].predict_proba(X_val_scaled)[:, 1],
    models['GradientBoosting'].predict_proba(X_val_scaled)[:, 1],
    (models['NeuralNetwork'].predict(X_val_scaled) > 0.5).astype(int).reshape(-1)
])

# Train meta-learner
meta_learner = LogisticRegression(max_iter=1000, class_weight='balanced')
meta_learner.fit(X_train_meta, y_train)

# Evaluate stacked model
stacked_preds = meta_learner.predict(X_val_meta)
stacked_f1 = f1_score(y_val, stacked_preds)
print("\nStacked model classification report:")
print(classification_report(y_val, stacked_preds))
models['StackedModel'] = meta_learner
model_scores['StackedModel'] = (stacked_f1, 0)

# Save the meta-learner model
joblib.dump(meta_learner, 'BinaryFolder/StackedModel.pkl')
print("Saved stacked model")

# Print summary of all model performances
print("\nModel Performance Summary (F1 Score):")
for name, (mean_f1, std_f1) in model_scores.items():
    print(f"{name}: {mean_f1:.4f} ({std_f1:.4f})")