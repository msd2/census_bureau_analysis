# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from model import train_model
from data import process_data

data_path = '../data/cleaned_census_data.csv'
model_filename = 'trained_model.sav'
encoder_filename = 'encoder.sav'

if __name__ == '__main__':
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True)

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label='salary',
        encoder=encoder,
        lb=lb,
        training=False)

    model = train_model(X_train, y_train)

    pickle.dump(model, open(model_filename, 'wb'))
    pickle.dump(encoder, open(encoder_filename, 'wb'))