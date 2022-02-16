from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
from model.model import train_model
from model.process_data import process_data
from model.model import compute_model_metrics, inference



if __name__ == '__main__':

    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, 'data/cleaned_census_data.csv')
    data = pd.read_csv(data_path)
    data = data.drop('Unnamed: 0', axis=1)
    train, test = train_test_split(data, test_size=0.20)
    
    model_filename = 'trained_model.sav'
    encoder_filename = 'encoder.sav'
    lb_filename = 'lb.sav'
    slice_performance_output = 'slice_output.txt'

    cat_features = ["workclass","education","marital-status","occupation",
                    "relationship","race","sex","native-country"]

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
    
    preds = inference(model, X_test)
    fbeta, precision, recall = compute_model_metrics(y_test, preds)
    print(f'fbeta: {fbeta}, precision: {precision}, recall: {recall}')


    pickle.dump(model, open(model_filename, 'wb'))
    pickle.dump(encoder, open(encoder_filename, 'wb'))
    pickle.dump(lb, open(lb_filename, 'wb'))