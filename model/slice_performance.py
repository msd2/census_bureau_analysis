import os
import pandas as pd
import pickle
from sklearn import preprocessing
from model import inference, compute_model_metrics
from data import process_data


def slice_performance(data, categorical_features, encoder, lb, model, output):

    for feature in categorical_features:
        for cls in data[feature].unique():

            slice = data[data[feature]==cls]
            X, y, encoder, lb = process_data(X=slice,
                                           categorical_features=categorical_features,
                                           label='salary',
                                           training=False,
                                           encoder=encoder,
                                           lb=lb)
            
            preds = inference(model, X)
            fbeta, precision, recall = compute_model_metrics(y, preds)
            slice_output = ','.join(str(x) for x in [feature, cls, fbeta, precision, recall])

            with open(output, 'a') as f:
                f.write('\n')
                f.write(slice_output)

            

if __name__ == '__main__':

    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, 'data/cleaned_census_data.csv')
    data = pd.read_csv(data_path)

    cat_features = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]

    lb = preprocessing.LabelBinarizer()
    lb.fit(data['salary'])
    output = 'slice_output.txt'

    with open('encoder.sav', 'rb') as f:
        encoder = pickle.load(f)

    with open('trained_model.sav', 'rb') as f:
        model = pickle.load(f)
    
    with open(output, 'w') as f:
        f.write('feature,class,fbeta,precision,recall')

    slice_performance(data, cat_features, encoder, lb, model, output)
    
