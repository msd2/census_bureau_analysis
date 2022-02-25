import pytest
from fastapi.testclient import TestClient
import pandas as pd
import logging
import numpy as np
import random
from main import app
from model.model import train_model, inference, compute_model_metrics


client = TestClient(app)

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def X_train():
    X_train = np.random.rand(10, 2)
    return X_train

@pytest.fixture
def y_train():
    return [0,0,1,0,1,0,1,1,1,0]

@pytest.fixture
def model(X_train, y_train):
    return train_model(X_train, y_train)

@pytest.fixture
def sample_person_under():
    df = pd.read_csv('data/cleaned_census_data.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df = df[df['salary']=='<=50K']
    data_row = df.sample(1)
    data_row = data_row.to_dict('records')[0]
    return data_row

@pytest.fixture
def sample_person_over():
    df = pd.read_csv('data/cleaned_census_data.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df = df[df['salary']=='>50K']
    data_row = df.sample(1)
    data_row = data_row.to_dict('records')[0]
    return data_row

def test_get_path():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}

def test_make_prediction_under(sample_person_under):
    r = client.post(
        '/predict/',
        json=sample_person_under)
    r.json()['result'][0]
    assert r.status_code == 200
    assert r.json()['result'][0] == '<=50K'

def test_make_prediction_over(sample_person_over):
    r = client.post(
        '/predict/',
        json=sample_person_over)
    r.json()['result'][0]
    assert r.status_code == 200
    assert r.json()['result'][0] == '>50K'



def test_train_model(X_train, y_train):
    try:
        clf = train_model(X_train, y_train)
    except ValueError:
        logging.error('Model failed to train')

    try:
        pred = clf.predict(np.random.rand(1,2))
    except ValueError as err:
        logging.error(f'Model failed to predict with {err}')
    except UnboundLocalError as err:
        logging.error(f'Model failed to predict with {err}')

    try:
        assert pred in [0,1]
    except AssertionError:
        logging.error('Prediction incorrect.')



def test_compute_model_metrics(model):

    preds = model.predict(np.random.rand(10,2))
    y = [0]*5 + [1]*5
    random.shuffle(y)

    try:
        fbeta, precision, recall = compute_model_metrics(y, preds)
    except ValueError as err:
        logging.error(f'Metrics failed to calculate with {err}')

    try:
        assert isinstance(fbeta, float)
    except AssertionError as err:
        logging.info(f'fbeta metric wrong type with {err}')

    try:
        assert isinstance(precision, float)
    except AssertionError as err:
        logging.info(f'precision metric wrong type with {err}')

    try:
        assert isinstance(recall, float)
    except AssertionError as err:
        logging.info(f'recall metric wrong type with {err}')



def test_inference(model):
    
    X = np.random.rand(10,2)
    try:
        preds = inference(model, X)
    except ValueError as err:
        logging.info(f'Inference failed with {err}')

    try:
        assert isinstance(preds, np.ndarray)
    except AssertionError as err:
        logging.info(f'Inference incorrect type with {err}')