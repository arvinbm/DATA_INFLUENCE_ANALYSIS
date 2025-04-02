import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def perform_hot_one_encoding(X_train, X_test, y_train, y_test):
    # OneHotEncode X
    encoder_X = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_X.fit(X_train)

    X_train_encoded = pd.DataFrame(encoder_X.transform(X_train), columns=encoder_X.get_feature_names_out())
    X_test_encoded = pd.DataFrame(encoder_X.transform(X_test), columns=encoder_X.get_feature_names_out())

    # LabelEncode y
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())
    y_test_encoded = label_encoder.transform(y_test.values.ravel())

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded

def load_and_prepare_data(csv_file_path):
    # Load CSV
    df = pd.read_csv(csv_file_path)

    # Separate features and labels
    X = df.drop(columns=['target'])
    y = df['target']

    # split the dataset into the train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform one-hot encoding to be compatible with the RandomForestClassifier
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = perform_hot_one_encoding(X_train, X_test, y_train, y_test)

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded


def train_random_forest_classifier(X, y):
    model = RandomForestClassifier()
    model.fit(X, y);
    return model


def compute_loo_values(base_model, X_train, y_train, X_test, y_test):
    loo_scores = []
    loo_result = {
        "positive": [],
        "negative": []
    }

    # Compute the model score when all instances are present in the training dataset
    base_predictions = base_model.predict(X_test)
    base_score = accuracy_score(y_test, base_predictions)

    # For every data point remove it from the dataset and retrain the model
    # Calculate how much the accuracy score changes
    for index in range(len(X_train)):
        removed_X_train = X_train.drop(X_train.index[index])
        removed_y_train = np.delete(y_train, index)

        retrained_model = train_random_forest_classifier(removed_X_train, removed_y_train)
        new_loo_predictions = retrained_model.predict(X_test)
        new_loo_score = accuracy_score(y_test, new_loo_predictions)

        change_in_accuracy = new_loo_score - base_score
        loo_scores.append((index, change_in_accuracy))

    # Order the results based on the index
    loo_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Add the top 5 positively and negatively influential data points
    for index, value in loo_scores[:5]:
        loo_result["positive"].append({"index": int(index), "value": round(value, 5)})
    
    for index, value in loo_scores[-5:]:
        loo_result["negative"].append({"index": int(index), "value": round(value, 5)})

    return loo_result


def compute_shapely_values(base_model, X_train, y_train, X_test, y_test):
    # Using the shap library to compute Shapley values because computing these
    # values manually without approximations is NP-complete
    shapely_result = {
        "positive": [],
        "negative": []
    }

    # Compute the shapley values
    explainer = shap.TreeExplainer(base_model)
    shapley_values = explainer.shap_values(X_train)

    # Extract the shapley values for class 1
    if isinstance(shapley_values, list):
        shapley_values = shapley_values[1]

    # Sum all the shapley values corresponding to each feature for each data point
    summed_shapley_values = np.sum(shapley_values, axis=1)

    # Construct a list of tuples which contains the index & the total shap value for each data point
    indexed_shapley_values = []
    for index, value in enumerate(summed_shapley_values):
        indexed_shapley_values.append((index, value))

    # Sort the list of tuples based on the value
    indexed_shapley_values.sort(key=lambda x: x[1], reverse=True)

    # Collect the top 5 most positive and negative influential data points
    for index, value in indexed_shapley_values[:5]:
        shapely_result["positive"].append({"index" : index, "value" : round(value, 5)})
    
    for index, value in indexed_shapley_values[-5:]:
        shapely_result["negative"].append({"index": index, "value": round(value, 5)})

    return shapely_result

def get_loo_shapely_results(csv_file_path):
    # Load the data and prepare for training the model
    X_train, y_train, X_test, y_test = load_and_prepare_data(csv_file_path)

    # Train the model (RandomForestClassifier)
    base_model = train_random_forest_classifier(X_train, y_train)

    # Get top 5 most positively and negatively influential data points
    loo_result = compute_loo_values(base_model, X_train, y_train, X_test, y_test)
    shapely_result = compute_shapely_values(base_model, X_train, y_train, X_test, y_test)

    # Combine both into one response
    loo_shapely_results = {
        "loo": loo_result,
        "shapely": shapely_result
    }
    
    return loo_shapely_results