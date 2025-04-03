import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def perform_hot_one_encoding(X_train, X_test, y_train, y_test):
    # Drop the date column (Unique for each row; thus, will not contribute to the overall score)
    X_train_encoded = X_train.drop(columns=['date'], errors='ignore')
    X_test_encoded = X_test.drop(columns=['date'], errors='ignore')

    # LabelEncode y
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())
    y_test_encoded = label_encoder.transform(y_test.values.ravel())

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded

def load_and_prepare_data(csv_file_path):
    # Load CSV
    df = pd.read_csv(csv_file_path)

    # Separate features and labels
    X = df.drop(columns=['weather'])
    y = df['weather']

    # split the dataset into the train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform one-hot encoding to be compatible with the RandomForestClassifier
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = perform_hot_one_encoding(X_train, X_test, y_train, y_test)

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded


def train_random_forest_classifier(X, y):
    # Use lower number of trees to make shapley value computation more efficient
    model = RandomForestClassifier(n_estimators=10)
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
        "top_5_influential": []
    }

    # Compute the Shapley values
    explainer = shap.Explainer(base_model.predict_proba, X_train)
    shapley_values = explainer(X_train)

    # Extract the raw SHAP value array
    shapley_values_array = shapley_values.values

    # Sum over all features and classes to get total absolute impact per sample
    total_influence = np.sum(np.abs(shapley_values_array), axis=(1, 2))

    # Get top 5 most influential samples
    top_5 = sorted(enumerate(total_influence), key=lambda x: x[1], reverse=True)[:5]

    # Store the result
    for index, value in top_5:
        shapely_result["top_5_influential"].append({
            "index": index,
            "value": round(value, 5)
        })

    return shapely_result

def get_loo_shapely_results(csv_file_path):
    # Load the data and prepare for training the model
    X_train, X_test, y_train, y_test = load_and_prepare_data(csv_file_path)

    print(y_train)

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