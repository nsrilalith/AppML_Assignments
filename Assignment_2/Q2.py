from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import pydotplus

def writegraphtofile(clf, featurelabels, filename): 
    dot_data = tree.export_graphviz(clf, feature_names=featurelabels, out_file=None) 
    graph=pydotplus.graph_from_dot_data(dot_data) 
    graph.write_png(filename)


data = pd.read_excel("AlienMushrooms.xlsx")

clf = tree.DecisionTreeClassifier(criterion="entropy")

feature_labels = data.columns[:-1]
features = data[feature_labels]

target = data[data.columns[-1]]

clf.fit(features, target)

writegraphtofile(clf, feature_labels, "treepic.png")

predictions = clf.predict(features)

# Calculate metrics
accuracy = accuracy_score(target, predictions)
precision = precision_score(target, predictions, average='macro')  # Use 'macro' for multi-class classification
recall = recall_score(target, predictions, average='macro')
f1 = f1_score(target, predictions, average='macro')
conf_matrix = confusion_matrix(target, predictions)

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)