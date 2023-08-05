"""
This script reads the JSON files and extracts the features needed for our model.

It contains code for feature engineering, simple imputation and one-hot encoding.
"""

import pandas as pd
import json


def extract_materials():
    """
    This function extracts the product materials found in the training dataset
    and saves them into a .txt file.
    """

    # We load the JSON flie.
    products = json.load(open("./assets/train_products.json"))

    # This list will hold all the materials found.
    data = list()

    # We iterate over all products.
    for _, v in products.items():

        # we extract the materials and add them to our list.
        packaging_materials = v["packaging_materials"]
        data.extend(packaging_materials)

    # We check how many items we have in our list.
    print(len(data))

    # We turn the list into a set to remove duplicates.
    # Then we turn the set back into a list so we can save it to a txt file.
    data = list(set(data))
    print(len(data))

    # We save the list to a .txt file.
    open("./assets/materials.txt", "w").write("\n".join(data))


def extract_top_categories():
    """
    This function extracts the products top category found in the training dataset
    and saves them into a .txt file.
    """

    # We load the JSON flie.
    products = json.load(open("./assets/train_products.json"))

    # This list will hold all the materials found.
    data = list()

    # We iterate over all products.
    for _, v in products.items():

        # we extract the top category and add it to our list.
        categories = v["categories_hierarchy"]
        data.append(categories[0])

    # We check how many categories we have.
    print(len(data))

    # We turn the list into a set to remove duplicates.
    # Then we turn the set back into a list so we can save it to a txt file.    data = list(set(data))
    print(len(data))

    # We save the list to a .txt file.
    open("./assets/top_categories.txt", "w",
         encoding="utf-8").write("\n".join(data))


def process_file(file):
    """
    This function reads a products JSON file and converts into a
    CSV file that can later be used to train a model.

    Parameters
    ----------
    file : str
        Either 'train' or 'test' for training or test datasets.

    """

    # Load the materials and top categories lists.
    materials = open("./assets/materials.txt", "r",
                     encoding="utf-8").read().splitlines()

    top_categories = open("./assets/top_categories.txt", "r",
                          encoding="utf-8").read().splitlines()

    nutrition_scores = ["a", "b", "c", "d", "e", "unknown"]

    # Load the specified JSON file.
    products = json.load(
        open(f"./assets/{file}_products.json", "r", encoding="utf-8"))

    # This list will hold the observations.
    data = list()

    # We iterate over all products.
    for k, v in products.items():

        # We extract the materials and categories for one-hot encoding.
        packaging_materials = v["packaging_materials"]
        categories_hierarchy = v["categories_hierarchy"]

        # Additives count sometimes is a string with the value 'unknown'
        # other times it has an int value. We will impute the value '1' when it is unkonwn
        # as it gives more accuracy.
        additives_count = v["additives_count"]

        if additives_count == "unknown":
            additives_count = 1

        # We select the features that improve accuracy.
        temp_dict = {
            "id": k,
            "name": v["name"],
            "brand": v["brand"],

            # The test dataset doesn't have this value, we will set it to 'NA'.
            "ecoscore_grade": v.get("ecoscore_grade", "NA"),

            "non_recyclable_and_non_biodegradable_materials_count": v["non_recyclable_and_non_biodegradable_materials_count"],
            "additives_count": additives_count,
            "is_beverage": v["is_beverage"],

            "est_co2_agriculture": v["est_co2_agriculture"],
            "est_co2_consumption": v["est_co2_consumption"],
            "est_co2_distribution": v["est_co2_distribution"],
            "est_co2_packaging": v["est_co2_packaging"],
            "est_co2_processing": v["est_co2_processing"],
            "est_co2_transportation": v["est_co2_transportation"],
        }

        # We will perform one-hot encoding in materials, categories and nutrition grade.
        for material in materials:
            temp_dict[f"mat_{material}"] = 1 if material in packaging_materials else 0

        for category in top_categories:
            temp_dict[f"top_cat_{category}"] = 1 if category in categories_hierarchy else 0

        for nut_score in nutrition_scores:
            temp_dict[f"nut_{nut_score}"] = 1 if v["nutrition_grade"] == nut_score else 0

        # We add the dict to the main list.
        data.append(temp_dict)

    # We create a pandas DataFrame with our list of dicts and save it to CSV.
    df = pd.DataFrame.from_records(data)
    df.to_csv(f"./assets/{file}_products.csv", index=False, encoding="utf-8")
    print("File processed:", file)


if __name__ == "__main__":

    # extract_materials()
    # extract_top_categories()

    process_file("train")
    process_file("test")
