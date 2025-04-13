import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# --- 1. Load Data ---
# Replace 'vital_signs_dataset.csv' with the actual path to your file
try:
    df = pd.read_csv('vital_signs.csv')
    print("Dataset loaded successfully.")
    # Display first few rows and info to check column names
    print(df.head())
    print(df.info())
except FileNotFoundError:
    print("Error: Dataset file not found. Make sure 'vital_signs_dataset.csv' is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

# --- 2. Select Features (X) and Target (y) ---
# Define the input features you want to use
feature_cols = [
    'Heart Rate',
    'Systolic Blood Pressure',
    'Diastolic Blood Pressure',
    'Body Temperature',
    'Oxygen Saturation'
]
target_col = 'Risk Category'

# Verify columns exist in the DataFrame
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    print(f"Error: The following feature columns are missing from the CSV: {missing_features}")
    exit()
if target_col not in df.columns:
    print(f"Error: The target column '{target_col}' is missing from the CSV.")
    exit()

X = df[feature_cols]
y = df[target_col]

print(f"\nSelected Features (X shape): {X.shape}")
print(f"Selected Target (y shape): {y.shape}")
print(f"Target value counts:\n{y.value_counts()}")

# --- 3. Encode Target Variable ---
# Convert 'Low Risk'/'High Risk' into 0/1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Check encoding: 0 might be 'High Risk', 1 might be 'Low Risk' or vice-versa
# Print the mapping to be sure
print(f"\nTarget variable encoding: {list(label_encoder.classes_)} -> {list(range(len(label_encoder.classes_)))}")
# Example: Target variable encoding: ['High Risk', 'Low Risk'] -> [0, 1] means High Risk=0, Low Risk=1

# --- 4. Split Data ---
# Split into 80% training and 20% testing data
# stratify=y_encoded ensures the proportion of High/Low Risk is similar in train and test sets
# random_state makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

# --- 5. Feature Scaling ---
# Scale numerical features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)     # Only transform the test data (use scaling learned from train)

print("\nFeature scaling applied.")

# --- 6. Instantiate Model ---
# Create a Random Forest Classifier model
# n_estimators=100 is a common starting point (number of trees)
# random_state ensures reproducibility of the model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# using class_weight='balanced' can help if one risk category is much more frequent than the other

print(f"\nInstantiated Model: {type(rf_model).__name__}")

# --- 7. Train Model ---
print("Training the model...")
rf_model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 8. Evaluate Model ---
print("\nEvaluating the model on the test set...")

# Make predictions on the scaled test data
y_pred = rf_model.predict(X_test_scaled)
# Get probability predictions for AUC calculation (needed for the positive class)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1] # Probability of class '1'

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Generate Confusion Matrix
# Rows: Actual, Columns: Predicted
# [[TN, FP],
#  [FN, TP]]  (assuming 0=Negative, 1=Positive - check your encoding!)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Generate Classification Report (Precision, Recall, F1-score)
# Make sure to map the 0/1 back to High/Low Risk for readability
target_names = label_encoder.classes_
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Calculate AUC-ROC
# AUC measures the model's ability to distinguish between the classes
# Closer to 1 is better, 0.5 is random guessing
try:
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC Score: {auc_score:.4f}")
except ValueError as e:
    print(f"\nCould not calculate AUC-ROC: {e}. This might happen if only one class is present in y_test or predictions.")

# --- Optional: Feature Importances ---
# Random Forest can tell you which features it found most important
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance_df)