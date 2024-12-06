"""
Authors: Filip Labuda, Jędrzej Stańczewski
How to use: Run the script to train a Decision Tree and SVM classifier on the Abalone dataset.
This script visualizes data, evaluates model performance, and tests classifiers on sample input data.
Example output in readme.md:
- The script trains a Decision Tree and SVM classifier to predict the number of rings in abalone data.
- It provides classification reports, accuracy scores, and visualizations of feature importance and confusion matrices.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load and prepare Abalone dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
           'Rings']
data_abalone = pd.read_csv(url, names=columns)
data_abalone = pd.get_dummies(data_abalone, columns=['Sex'])
X_abalone = data_abalone.drop('Rings', axis=1)
y_abalone = data_abalone['Rings']

# Split the data into training and testing sets
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(X_abalone, y_abalone, test_size=0.2,
                                                                                    random_state=42)

# Train Decision Tree
tree_model_abalone = DecisionTreeClassifier(random_state=42)
tree_model_abalone.fit(X_train_abalone, y_train_abalone)

# Estimate Decision Tree
y_pred_tree_abalone = tree_model_abalone.predict(X_test_abalone)
accuracy_abalone = accuracy_score(y_test_abalone, y_pred_tree_abalone)

# Train SVM
svm_model_abalone = SVC(random_state=42)
svm_model_abalone.fit(X_train_abalone, y_train_abalone)

# Estimate SVM
y_pred_svm_abalone = svm_model_abalone.predict(X_test_abalone)
accuracy_svm_abalone = accuracy_score(y_test_abalone, y_pred_svm_abalone)

print("Decission Tree - Results:")
print(classification_report(y_test_abalone, y_pred_tree_abalone))
print("Decission Tree accuracy:", accuracy_abalone)

print("SVM - Results:")
print(classification_report(y_test_abalone, y_pred_svm_abalone))
print("SVM accuracy:", accuracy_svm_abalone)

# Visualize the data and model performance
# Visualization of Rings column distribution
plt.figure(figsize=(8, 6))
sns.histplot(y_abalone, kde=False, bins=15, color='blue')
plt.title("Rozkład liczby pierścieni (Rings)")
plt.xlabel("Liczba pierścieni")
plt.ylabel("Liczba obserwacji")
plt.show()

# Importance in Decision Tree
importances = tree_model_abalone.feature_importances_
features = X_abalone.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, color='green')
plt.title("Istotność cech w drzewie decyzyjnym (Abalone)")
plt.xlabel("Istotność")
plt.ylabel("Cechy")
plt.show()

# Confusion matrix for Decision Tree
conf_matrix_tree = confusion_matrix(y_test_abalone, y_pred_tree_abalone)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion matrix for Decision Tree")
plt.xlabel("Estimated")
plt.ylabel("Actual")
plt.show()

# Confusion matrix for SVM
conf_matrix_svm = confusion_matrix(y_test_abalone, y_pred_svm_abalone)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion matrix for SVM")
plt.xlabel("Estimated")
plt.ylabel("Actual")
plt.show()

# Test classifiers on sample input data
sample_data = [[1, 0, 0, 0.5, 0.4, 0.2, 0.6, 0.3, 0.1, 0.15]]  # Sex column

# Decision Tree prediction
tree_prediction = tree_model_abalone.predict(sample_data)
print("Decision Tree prediction:", tree_prediction)

# SVM prediction
scaler = StandardScaler().fit(X_abalone)
sample_data_scaled = scaler.transform(sample_data)
svm_prediction = svm_model_abalone.predict(sample_data_scaled)
print("SVM prediction:", svm_prediction)
