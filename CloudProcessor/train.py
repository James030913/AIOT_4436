import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load and prepare your data
# Assuming your data is in a CSV file - update the filename as needed
df = pd.read_csv('vital_signs.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nData Preview:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Drop unnecessary columns
df_model = df.drop(['Patient ID', 'Timestamp'], axis=1)

# Convert categorical variables
df_model = pd.get_dummies(df_model, columns=['Gender'], drop_first=True)
risk_mapping = {'Low Risk': 0, 'High Risk': 1}
df_model['Risk Category'] = df_model['Risk Category'].map(risk_mapping)

# Step 2: Define features and target
# Select the features you want to use
X = df_model[['Heart Rate', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 
              'Body Temperature', 'Oxygen Saturation']]

# Target variable
y = df_model['Risk Category']

# Print class distribution
print("\nClass Distribution:")
print(y.value_counts())
print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}")

# Step 3: Scale the features
# Initialize and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future predictions
scaler_filename = 'feature_scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as '{scaler_filename}'")

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Step 5: Train the Random Forest model
# Create and train the model
rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Maximum depth of trees
    min_samples_split=2,     # Minimum samples required to split a node
    min_samples_leaf=1,      # Minimum samples required at each leaf node
    random_state=42,         # For reproducibility
    n_jobs=-1                # Use all available cores
)

print("\nTraining Random Forest model...")
rf.fit(X_train, y_train)
print("Model training complete!")

# Step 6: Make predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]  # Probability of high risk

# Step 7: Evaluate the model
print("\n--- Model Evaluation ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'")

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='roc_auc')
print(f"\nCross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Feature importance plot saved as 'feature_importance.png'")

# Step 8: Save the model
# Save with the expected filename for your prediction function
model_filename = 'risk_prediction_model.pkl'
joblib.dump(rf, model_filename)
print(f"\nModel saved as '{model_filename}'")

# Step 9: Test your prediction function
def predict_risk(heart_rate, systolic_bp, diastolic_bp, body_temp, oxygen_sat):
    # Load the model and scaler
    try:
        model = joblib.load('risk_prediction_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        # Prepare the input data
        patient_data = [[heart_rate, systolic_bp, diastolic_bp, body_temp, oxygen_sat]]
        patient_data_scaled = scaler.transform(patient_data)
        
        # Make prediction
        prediction = model.predict(patient_data_scaled)
        probability = model.predict_proba(patient_data_scaled)
        
        # Convert numeric prediction back to label
        risk_labels = {v: k for k, v in risk_mapping.items()}
        predicted_label = risk_labels[prediction[0]]
        
        return {
            'risk_category': predicted_label,
            'probability': probability[0][list(model.classes_).index(prediction[0])]
        }
    except Exception as e:
        print(f"Error in risk prediction: {e}")
        return {
            'risk_category': 'Unknown',
            'probability': 0.0
        }

# Example usage with the sample data provided
print("\n--- Testing Prediction Function ---")
sample_heart_rate = 60
sample_systolic_bp = 124
sample_diastolic_bp = 86
sample_body_temp = 36.86
sample_oxygen_sat = 95.70

prediction_result = predict_risk(
    heart_rate=sample_heart_rate,
    systolic_bp=sample_systolic_bp,
    diastolic_bp=sample_diastolic_bp,
    body_temp=sample_body_temp,
    oxygen_sat=sample_oxygen_sat
)

print(f"Input values:")
print(f"  Heart Rate: {sample_heart_rate} bpm")
print(f"  Systolic BP: {sample_systolic_bp} mmHg")
print(f"  Diastolic BP: {sample_diastolic_bp} mmHg")
print(f"  Body Temperature: {sample_body_temp}°C")
print(f"  Oxygen Saturation: {sample_oxygen_sat}%")
print(f"\nPredicted Risk Category: {prediction_result['risk_category']}")
print(f"Probability: {prediction_result['probability']:.4f} ({prediction_result['probability']*100:.2f}%)")