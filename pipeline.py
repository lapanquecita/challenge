"""
This script creates the model, optimizes and cross-validates it.

It then is used to predict the values from the test dataset.
"""

import json
import pickle

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier


# The file name of the saved model.
MODEL_FILENAME = './model.pickle'


def train_model():

    # We load the training dataset.
    df = pd.read_csv("./assets/train_products.csv", encoding="utf-8")
 
    # Select the features we will use.
    X = df.iloc[:, 4:]

    # Select the target.
    y = df["ecoscore_grade"]

    # We split the data for training and testing the model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)

    # We create and fit the model.
    my_model = XGBClassifier(
        random_state=0,
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
    )
    my_model.fit(X_train, y_train)

    # We test our model and get the F1-score.
    y_pred = my_model.predict(X_test)

    score = f1_score(y_test, y_pred, average="weighted")
    print("F1-Score:", score)

    pickle.dump(my_model, open(MODEL_FILENAME, "wb"))
    print("Model generated.")


def predict_new():

    # We load the previously generated model.
    loaded_model = pickle.load(open(MODEL_FILENAME, 'rb'))

    # We load the test dataset.
    df = pd.read_csv("./assets/test_products.csv")

    # Select the features we will use.
    features = df.iloc[:, 4:]

    new_predictions = loaded_model.predict(features)

    data = dict()

    for index, item in enumerate(new_predictions):
        data[index] = int(item)

    data = {"target": data}

    # We save the results to a JSON file.
    json.dump(data, open("target.json", "w"))


def optimize_parameters():
    """
    This function is used for two things:

    1. It helps us determine the most accurate parameters.
    2. It performs cross-validatoion with all the combinations of parameters.

    The results of this function were used to create the final model.
    """

    # We load the training dataset.
    df = pd.read_csv("./assets/train_products.csv", encoding="utf-8")

    # Select the features we will use.
    X = df.iloc[:, 4:]

    # Select the target.
    y = df["ecoscore_grade"]

    # Our grid search will iterate over all combinations of the following parameters.
    grid_params = {
        "learning_rate": [0.1, 0.2, 0.3],
        "max_depth": [6, 7, 8],
        "n_estimators": [100, 150, 200],
    }

    # We create and fit the model.
    my_model = XGBClassifier(random_state=0)

    # We feed the parameters to the GridSearchCV function.
    grid_search = GridSearchCV(
        estimator=my_model,
        param_grid=grid_params,
    )

    grid_search.fit(X, y)

    # We print the results.
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)


if __name__ == "__main__":

    # optimize_parameters()
    train_model()
    predict_new()
