from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

data = pd.read_excel("diabetic_data.xlsx")

clf = tree.DecisionTreeClassifier(criterion="entropy")
encoder = OneHotEncoder(sparse=False)

data_encoded = encoder.fit_transform(data)

feature_labels = data.columns[:-1]
features = data_encoded[data_encoded.columns]

target = data_encoded[data.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test, y_test)
accuracy = accuracy_score(target, predictions)
print(f"Accuracy: {accuracy:.4f}")

