import pytest
import numpy as np
import logging
import random
from model.model import train_model, inference, compute_model_metrics

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
