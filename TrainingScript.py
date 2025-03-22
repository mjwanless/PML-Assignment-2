import pandas as pd
import numpy as np
import os
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

if not os.path.exists('Visualizations'):
    os.makedirs('Visualizations')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

df = pd.read_csv('CreditDefault.csv')

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget variable distribution:")
print(df['CreditDefault'].value_counts())
print(df['CreditDefault'].value_counts(normalize=True))

plt.figure(figsize=(10, 6))
ax = sns.countplot(x='CreditDefault', data=df)
plt.title('Distribution of Credit Default', fontsize=16)
plt.xlabel('Default Status (1 = Default, 0 = No Default)', fontsize=12)
plt.ylabel('Count', fontsize=12)

total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 300,
            f'{height}\n({height/total:.1%})',
            ha="center", fontsize=12)

plt.tight_layout()
plt.savefig('Visualizations/default_distribution.png', dpi=300)
plt.close()

print("\nCategorical feature distributions:")
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

fig, axes = plt.subplots(len(categorical_features), 1, figsize=(12, 15))
for i, feature in enumerate(categorical_features):
    print(f"\n{feature} distribution:")
    print(df[feature].value_counts())

    sns.countplot(x=feature, hue='CreditDefault', data=df, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature} by Default Status', fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].legend(title='Default Status', labels=['No Default', 'Default'])

plt.tight_layout()
plt.savefig('Visualizations/categorical_features.png', dpi=300)
plt.close()

fig, axes = plt.subplots(len(categorical_features), 1, figsize=(12, 15))
for i, feature in enumerate(categorical_features):
    default_rate = df.groupby(feature)['CreditDefault'].mean().sort_values(ascending=False)

    sns.barplot(x=default_rate.index, y=default_rate.values, ax=axes[i])
    axes[i].set_title(f'Default Rate by {feature}', fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel('Default Rate', fontsize=12)

    for p in axes[i].patches:
        axes[i].annotate(f'{p.get_height():.1%}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'bottom', fontsize=10)

plt.tight_layout()
plt.savefig('Visualizations/default_rate_by_category.png', dpi=300)
plt.close()

payment_history = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

fig, axes = plt.subplots(3, 2, figsize=(16, 15))
axes = axes.flatten()

for i, feature in enumerate(payment_history):
    print(f"\n{feature} distribution:")
    print(df[feature].value_counts())

    sns.countplot(x=feature, hue='CreditDefault', data=df, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature} by Default Status', fontsize=14)
    axes[i].set_xlabel('Payment Status (Positive = Delay in days)', fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].legend(title='Default Status', labels=['No Default', 'Default'])

plt.tight_layout()
plt.savefig('Visualizations/payment_history.png', dpi=300)
plt.close()

default_rate_by_pay1 = df.groupby('PAY_1')['CreditDefault'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=default_rate_by_pay1.index, y=default_rate_by_pay1.values)
plt.title('Default Rate by Payment Status (PAY_1)', fontsize=16)
plt.xlabel('Payment Status', fontsize=12)
plt.ylabel('Default Rate', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1%}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'bottom', fontsize=10)

plt.tight_layout()
plt.savefig('Visualizations/default_rate_by_payment.png', dpi=300)
plt.close()

bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

plt.figure(figsize=(12, 6))
sns.boxplot(x='CreditDefault', y='LIMIT_BAL', data=df)
plt.title('Credit Limit by Default Status', fontsize=16)
plt.xlabel('Default Status (1 = Default, 0 = No Default)', fontsize=12)
plt.ylabel('Credit Limit', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Visualizations/credit_limit_boxplot.png', dpi=300)
plt.close()

print("\nPerforming data preparation and feature engineering...")

df = df.drop('ID', axis=1)

df['TOTAL_BILL'] = df[bill_columns].sum(axis=1)
df['TOTAL_PAY'] = df[pay_columns].sum(axis=1)
df['PAYMENT_RATIO'] = df['TOTAL_PAY'] / df['TOTAL_BILL'].replace(0, 0.01)
df['AVG_BILL_AMT'] = df[bill_columns].mean(axis=1)
df['AVG_PAY_AMT'] = df[pay_columns].mean(axis=1)
df['LATE_PAYMENT_COUNT'] = df[payment_history].apply(lambda x: sum(x > 0), axis=1)

engineered_features = ['TOTAL_BILL', 'TOTAL_PAY', 'PAYMENT_RATIO', 'LATE_PAYMENT_COUNT']
fig, axes = plt.subplots(len(engineered_features), 1, figsize=(12, 20))

for i, feature in enumerate(engineered_features):
    sns.boxplot(x='CreditDefault', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f'{feature} by Default Status', fontsize=14)
    axes[i].set_xlabel('Default Status (1 = Default, 0 = No Default)', fontsize=12)
    axes[i].set_ylabel(feature, fontsize=12)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Visualizations/engineered_features.png', dpi=300)
plt.close()

plt.figure(figsize=(16, 14))
corr_columns = payment_history + bill_columns[:1] + pay_columns[:1] + engineered_features + ['CreditDefault']
correlation_matrix = df[corr_columns].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", mask=mask)
plt.title('Correlation Matrix of Key Features', fontsize=16)
plt.tight_layout()
plt.savefig('Visualizations/correlation_matrix.png', dpi=300)
plt.close()

X = df.drop('CreditDefault', axis=1)
y = df['CreditDefault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

test_df = pd.concat([X_test, y_test], axis=1)
train_df = pd.concat([X_train, y_train], axis=1)

if not os.path.exists('BinaryFolder'):
    os.makedirs('BinaryFolder')

test_df.to_csv('test.csv', index=False)
train_df.to_csv('train.csv', index=False)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'BinaryFolder/scaler.pkl')

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

print("\nPreparing to build models...")

def evaluate_model(model, X, y, cv=5):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='f1')

    print(f"F1 Score: {cv_results.mean():.4f} ({cv_results.std():.4f})")
    return cv_results.mean(), cv_results.std()

models = {}
model_scores = {}

print("\nTraining and evaluating base models...")

print("\nLogistic Regression:")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
lr_f1_mean, lr_f1_std = evaluate_model(lr, X_val_scaled, y_val)
models['LogisticRegression'] = lr
model_scores['LogisticRegression'] = (lr_f1_mean, lr_f1_std)

print("\nRandom Forest:")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, y_train)
rf_f1_mean, rf_f1_std = evaluate_model(rf, X_val_scaled, y_val)
models['RandomForest'] = rf
model_scores['RandomForest'] = (rf_f1_mean, rf_f1_std)

print("\nGradient Boosting:")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_f1_mean, gb_f1_std = evaluate_model(gb, X_val_scaled, y_val)
models['GradientBoosting'] = gb
model_scores['GradientBoosting'] = (gb_f1_mean, gb_f1_std)

for name, model in models.items():
    joblib.dump(model, f'BinaryFolder/{name}_model.pkl')
    print(f"Saved {name} model")

plt.figure(figsize=(12, 8))
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importance.values[:15], y=feature_importance.index[:15])
plt.title('Top 15 Feature Importance from Random Forest', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig('Visualizations/feature_importance.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))

base_models = ['LogisticRegression', 'RandomForest', 'GradientBoosting']
colors = ['blue', 'green', 'orange']

for i, model_name in enumerate(base_models):
    model = models[model_name]
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for Base Models', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('Visualizations/roc_curves_base.png', dpi=300)
plt.close()

print("\nBuilding Neural Network model...")

def create_nn_model(optimizer='adam', nodes1=64, nodes2=32, nodes3=16, dropout_rate=0.2,
                    activation='relu', kernel_initializer='he_uniform', learning_rate=0.001):
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    inputs = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(nodes1, activation=activation, kernel_initializer=kernel_initializer)(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(nodes2, activation=activation, kernel_initializer=kernel_initializer)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nodes3, activation=activation, kernel_initializer=kernel_initializer)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

print("\nPerforming hyperparameter tuning for neural network...")

optimizers = ['adam', 'rmsprop']
nodes1_options = [32, 64, 128]
nodes2_options = [16, 32, 64]
nodes3_options = [8, 16, 32]
dropout_rates = [0.1, 0.2, 0.3]
activations = ['relu', 'elu', 'selu']
kernel_initializers = ['he_uniform', 'he_normal', 'glorot_uniform']
learning_rates = [0.001, 0.01, 0.0001]
batch_sizes = [64, 128, 256]

param_combinations = []
for optimizer in optimizers:
    for nodes1 in nodes1_options:
        for nodes2 in nodes2_options:
            for nodes3 in nodes3_options:
                for dropout_rate in dropout_rates:
                    for activation in activations:
                        for kernel_initializer in kernel_initializers:
                            for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                    param_combinations.append({
                                        'optimizer': optimizer,
                                        'nodes1': nodes1,
                                        'nodes2': nodes2,
                                        'nodes3': nodes3,
                                        'dropout_rate': dropout_rate,
                                        'activation': activation,
                                        'kernel_initializer': kernel_initializer,
                                        'learning_rate': learning_rate,
                                        'batch_size': batch_size
                                    })

sampled_combinations = random.sample(param_combinations, min(30, len(param_combinations)))

best_f1 = 0
best_params = {}
tuning_results = []

print(f"Testing {len(sampled_combinations)} hyperparameter combinations...")
for i, params in enumerate(sampled_combinations):
    print(f"Testing combination {i+1}/{len(sampled_combinations)}")

    model = create_nn_model(
        optimizer=params['optimizer'],
        nodes1=params['nodes1'],
        nodes2=params['nodes2'],
        nodes3=params['nodes3'],
        dropout_rate=params['dropout_rate'],
        activation=params['activation'],
        kernel_initializer=params['kernel_initializer'],
        learning_rate=params['learning_rate']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,  # More epochs with early stopping
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = (model.predict(X_val_scaled) > 0.5).astype(int).reshape(-1)
    current_f1 = f1_score(y_val, y_pred)

    tuning_results.append({
        'optimizer': params['optimizer'],
        'nodes1': params['nodes1'],
        'nodes2': params['nodes2'],
        'nodes3': params['nodes3'],
        'f1_score': current_f1
    })

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params = params.copy()

    parameter_summary = f"optimizer={params['optimizer']}, nodes1={params['nodes1']}, nodes2={params['nodes2']}, nodes3={params['nodes3']}, lr={params['learning_rate']}"
    print(f"  {parameter_summary} - F1: {current_f1:.4f}")

print(f"\nBest Neural Network parameters: {best_params}")
print(f"Best F1 Score: {best_f1:.4f}")

tuning_df = pd.DataFrame(tuning_results)
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='optimizer', y='f1_score', data=tuning_df)
plt.title('F1 Score by Optimizer', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
sns.boxplot(x='nodes1', y='f1_score', data=tuning_df)
plt.title('F1 Score by First Layer Nodes', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
sns.boxplot(x='nodes2', y='f1_score', data=tuning_df)
plt.title('F1 Score by Second Layer Nodes', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
sns.boxplot(x='nodes3', y='f1_score', data=tuning_df)
plt.title('F1 Score by Third Layer Nodes', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Visualizations/hyperparameter_tuning.png', dpi=300)
plt.close()

print("\nTraining final Neural Network model with best parameters...")
final_nn_model = create_nn_model(
    optimizer=best_params['optimizer'],
    nodes1=best_params['nodes1'],
    nodes2=best_params['nodes2'],
    nodes3=best_params['nodes3'],
    dropout_rate=best_params['dropout_rate'],
    activation=best_params['activation'],
    kernel_initializer=best_params['kernel_initializer'],
    learning_rate=best_params['learning_rate']
)

checkpoint_path = "BinaryFolder/nn_model.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

history = final_nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=best_params['batch_size'],
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Visualizations/nn_training_history.png', dpi=300)
plt.close()

y_pred_nn = (final_nn_model.predict(X_val_scaled) > 0.5).astype(int).reshape(-1)
nn_f1 = f1_score(y_val, y_pred_nn)
print("\nNeural Network classification report:")
print(classification_report(y_val, y_pred_nn))
models['NeuralNetwork'] = final_nn_model
model_scores['NeuralNetwork'] = (nn_f1, 0)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred_nn)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Neural Network Confusion Matrix', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('Visualizations/nn_confusion_matrix.png', dpi=300)
plt.close()

print("\nBuilding stacked model...")
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

meta_learner = LogisticRegression(max_iter=1000, class_weight='balanced')
meta_learner.fit(X_train_meta, y_train)

stacked_preds = meta_learner.predict(X_val_meta)
stacked_f1 = f1_score(y_val, stacked_preds)
print("\nStacked model classification report:")
print(classification_report(y_val, stacked_preds))
models['StackedModel'] = meta_learner
model_scores['StackedModel'] = (stacked_f1, 0)

joblib.dump(meta_learner, 'BinaryFolder/StackedModel.pkl')
print("Saved stacked model")

plt.figure(figsize=(10, 6))
model_names = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'NeuralNetwork']
weights = meta_learner.coef_[0]
sns.barplot(x=model_names, y=weights)
plt.title('Stacked Model Weights', fontsize=16)
plt.xlabel('Base Model', fontsize=12)
plt.ylabel('Weight', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Visualizations/stacked_model_weights.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, stacked_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Stacked Model Confusion Matrix', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('Visualizations/stacked_model_confusion_matrix.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
models_to_compare = list(model_scores.keys())
f1_means = [score[0] for score in model_scores.values()]
f1_stds = [score[1] for score in model_scores.values()]

bars = plt.bar(models_to_compare, f1_means, yerr=f1_stds, capsize=5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.title('Model Performance Comparison (F1 Score)', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.ylim(0, max(f1_means) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Visualizations/model_comparison.png', dpi=300)
plt.close()

print("\nModel Performance Summary (F1 Score):")
for name, (mean_f1, std_f1) in model_scores.items():
    print(f"{name}: {mean_f1:.4f} ({std_f1:.4f})")