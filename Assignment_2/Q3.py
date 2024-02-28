import numpy as np
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pydotplus


def loadData(filename):
    # Load data
    data = pd.read_excel(filename)
    return data


def writegraphtofile(clf, featurelabels, filename):
    dot_data = tree.export_graphviz(clf, feature_names=featurelabels, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(filename)


def preprocessData(data):
    mix_col = []
    for col in data.columns:
        unique_types = data[col].apply(type).unique()
        if len(unique_types) > 1:
            mix_col.append(col)

    data[mix_col] = data[mix_col].astype(str)
    data.replace("?", np.nan, inplace=True)
    # Separate features and target
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]  # The last column

    X = X.drop(['encounter_id', 'patient_nbr'], axis=1)

    # Identify categorical columns (modify this list based on your dataset)
    categorical_features = ["race", "gender", "age", "weight", "admission_type_id", "discharge_disposition_id",
                            "admission_source_id", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3",
                            'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']  # Example column names

    # Identify Ordinal columns (for medications)
    ordinal_features_medications = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                                    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                                    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
                                    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                                    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', ]
    medication_order = [['No', 'Steady', 'Up', 'Down']]
    medication_order_for_all_columns = medication_order * len(ordinal_features_medications)

    # Imputers for categorical and ordinal features
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
    ordinal_imputer = SimpleImputer(strategy='most_frequent')

    # Identify numerical columns (assuming all other columns are numerical if not categorical or ordinal)
    numerical_features = [col for col in X.columns if col not in categorical_features + ordinal_features_medications]

    # Imputers
    numerical_imputer = SimpleImputer(strategy='mean')  # Impute numerical columns with their mean

    # Encoder
    one_hot_encoder = OneHotEncoder()
    ordinal_encoder = OrdinalEncoder(categories=medication_order_for_all_columns)

    # Update ColumnTransformer to include numerical imputation
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([('imputer', categorical_imputer), ('encoder', one_hot_encoder)]), categorical_features),
            ('ord_meds', Pipeline([('imputer', ordinal_imputer),
                                   ('encoder', ordinal_encoder)]),
             ordinal_features_medications),
            ('num', numerical_imputer, numerical_features)
        ],
        remainder='passthrough'  # Ensure no column is left unprocessed
    )

    # Check for NaN values post-imputation

    print("Finished Pre-Processing, Starting Transformation")
    # Apply transformations
    X_encoded = preprocessor.fit_transform(X)

    # Extracting new feature names after preprocessing
    feature_names = []

    # Getting feature names for categorical features
    for cat_feature, one_hot in zip(categorical_features,
                                    preprocessor.named_transformers_['cat']['encoder'].categories_):
        feature_names.extend([f"{cat_feature}_{category}" for category in one_hot])

    # Adding ordinal feature names as is
    feature_names.extend(ordinal_features_medications)

    # Adding numerical feature names as is
    feature_names.extend(numerical_features)

    print("Finished Transformation of dataset")
    return X_encoded, y, feature_names


def MultiClassPrediction(X_encoded, y):
    # Split data for the multiclass classification model
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_encoded, y, test_size=0.2,
                                                                                random_state=42)

    # Initialize and train the multiclass classifier
    clf_multiclass = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf_multiclass.fit(X_train_multi, y_train_multi)

    # Predictions for multiclass classification
    predictions_multi = clf_multiclass.predict(X_test_multi)
    predictions_multi_train = clf_multiclass.predict(X_train_multi)

    # Evaluate the multiclass model
    accuracy_multi = accuracy_score(y_test_multi, predictions_multi)
    print(f"Multiclass Classification Accuracy with Test Set: {accuracy_multi:.4f}")

    accuracy_multi_train = accuracy_score(y_train_multi, predictions_multi_train)
    print(f"Multiclass Classification Accuracy with Train Set: {accuracy_multi_train:.4f}")
    # Plotting the multiclass classification tree
    return clf_multiclass


def BinaryClassification(X_encoded, y):
    # Binary target preprocessing
    y_binary = y.replace({'NO': 0, '<30': 1, '>30': 1})

    # Split data for the binary classification model
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_encoded, y_binary, test_size=0.2,
                                                                        random_state=42)

    # Initialize and train the binary classifier
    clf_binary = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf_binary.fit(X_train_bin, y_train_bin)

    # Predictions for binary classification
    predictions_bin_test = clf_binary.predict(X_test_bin)
    predictions_bin_train = clf_binary.predict(X_train_bin)

    # Evaluate the binary model
    accuracy_bin_test = accuracy_score(y_test_bin, predictions_bin_test)
    print(f"Binary Classification Accuracy with Test Set: {accuracy_bin_test:.4f}")

    accuracy_bin_train = accuracy_score(y_train_bin, predictions_bin_train)
    print(f"Binary Classification Accuracy with Train Set: {accuracy_bin_train:.4f}")

    # Plotting the binary classification tree
    return clf_binary


if __name__ == "__main__":
    file_location = "diabetic_data.xlsx"
    dataframe = loadData(file_location)  # created a dataframe
    feature_labels = dataframe.columns[:-1].tolist()

    X, Y, feature_names = preprocessData(dataframe)  # Preprocessing the dataframe, and obtaining (features, target)
    print("Dataset Is Ready \n")

    print("Training Dataset")

    clf_binary = BinaryClassification(X, Y)
    writegraphtofile(clf_binary, feature_names, "Assignment_2" + "tree_pic_binary.png")
    clf_multiclass = MultiClassPrediction(X, Y)
    writegraphtofile(clf_multiclass, feature_names, "Assignment_2" + "tree_pic_multiclass.png")
