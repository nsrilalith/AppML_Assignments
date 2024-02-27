from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
data = pd.read_excel("diabetic_data.xlsx")

mix_col = []
for col in data.columns:
    unique_types = data[col].apply(type).unique()
    if len(unique_types) > 1:
        print(f"Column '{col}' has mixed types: {unique_types}")
        mix_col.append(col)

data[mix_col] = data[mix_col].astype(str)

# Separate features and target
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]  # The last column

# Identify categorical columns (modify this list based on your dataset)
categorical_features = ["race", "gender", "age", "weight", "admission_type_id", "discharge_disposition_id", "admission_source_id", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3", 'max_glu_serum', 'A1Cresult',  'change', 'diabetesMed']  # Example column names

#Identify Ordinal columns (for medications)
ordinal_features_medications = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',]
medication_order = [['No', 'Steady', 'Up', 'Down']]
medication_order_for_all_columns = medication_order * len(ordinal_features_medications)

# Create a ColumnTransformer to encode categorical columns
print("Beginning Pre-processing")
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False), categorical_features),
        ('ord_meds', OrdinalEncoder(categories=medication_order_for_all_columns), ordinal_features_medications)
    ],
    remainder='passthrough'  # Keep the rest of the columns unchanged
)

print("Finished Pre-Processing, Starting Transformation")
# Apply transformations
X_encoded = preprocessor.fit_transform(X)

print("Finished Transformation of dataset, Starting Training Test Set")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

print("Finished Training, now Performing Predictions")

print("Predictions:")
# Make predictions
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")


