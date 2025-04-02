import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_random_forest_classifier(X, y):
    model = RandomForestClassifier()
    model.fit(X, y);
    return model


def compute_loo_values(csv_file_path):
    loo_result = {
        "positive": [],
        "negative": []
    }
    
    return loo_result


def compute_shapely_values(csv_file_path):
    shapely_result = {
        "positive": [],
        "negative": []
    }
    
    return shapely_result

def get_loo_shapely_results(csv_file_path):
    # Load and prepare data
    df = pd.read_csv(csv_file_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train model
    model = train_random_forest_classifier(X, y)

    # Get values
    loo_result = compute_loo_values(model, X, y)
    shapely_result = compute_shapely_values(model, X, y)

    # Combine both into one response
    loo_shapely_results = {
        "loo": loo_result,
        "shapely": shapely_result
    }
    
    return loo_shapely_results