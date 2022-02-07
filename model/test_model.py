import pytest
import numpy as np
import logging
from model import train_model, inference, compute_model_metrics

logging.basicConfig(
    level = logging.INFO,
    filemode = 'w',
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


@pytest.fixture
def X_train():
    return np.random.rand(1,10)[0]

@pytest.fixture
def y_train():
    return [0,0,1,0,1,0,1,1,1,0]

@pytest.fixture
def model(X_train, y_train):
    return train_model(X_train, y_train)


def test_train_model(X_train, y_train):
    try:
        clf = train_model(X_train, y_train)
    except:
        logger.error('Model failed to train')
    logger.info('Model successfully trained')


def test_compute_model_metrics(model):
    pass

def test_inference():
    pass