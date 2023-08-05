# Danone Hackathon - Data Science Challenge

For this challenge we needed to classify the ecoscore for each product in the test dataset.

For this task I tried various classification models: RandomForestClassifier, GradientBoostingClassifier and XGBClassifier.

After several tries, I had the most accurate results with XGBClassifier.

The datasets were in JSON format, one of the tasks was to create a preprocessing pipeline and feature extraction. I used the Python standard library to achieve this. I implemented one-hot encoding for the categorical features and only required one feature to be imputed.

For optimizing the model and cross-validation I used the GridSearchCV function from the scikit-learn library. Once I got the optimal parameters I used them for training the final model.
