import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Directly load CSV file (keep creditcard.csv in the same folder as script)
data = pd.read_csv("creditcard.csv")

print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
data.info()

# Remove missing values
data = data.dropna()

print("\nSummary statistics before scaling:")
print(data.describe())

# Scale 'Amount' and 'Time' columns
scaler = StandardScaler()
data[['Amount', 'Time']] = scaler.fit_transform(data[['Amount', 'Time']])

# Split features and labels
X = data.drop(columns=['Class'])
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_train)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = np.where(train_predictions == -1, 1, 0)
test_predictions = np.where(test_predictions == -1, 1, 0)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['Time'],
    y=data['Amount'],
    hue=data['Class'],
    palette={0: 'blue', 1: 'red'}
)
plt.title('Transaction Amounts Over Time (Fraud vs Normal)')
plt.xlabel('Time (scaled)')
plt.ylabel('Amount (scaled)')

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['Normal (Blue)', 'Fraud (Red)'], title='Class')
plt.show()

# Save results
results = X_test.copy()
results['Actual'] = y_test
results['Predicted'] = test_predictions
results.to_csv("fraud_detection_results.csv", index=False)

print("\nFraud detection process completed. Check the visualizations and results.")
