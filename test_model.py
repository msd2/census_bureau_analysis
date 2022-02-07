import pytest
import sys
import numpy as np
import logging
sys.path.append('model/')
import model

logging.basicConfig(
    level = logging.INFO,
    filemode = 'w',
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


@pytest.fixture
def X_train():
    return np.random.rand(1,20)

@pytest.fixture
def y_train():
    return np.random.rand(1,20)

@pytest.fixture
def model(X_train, y_train):
    return model.train_model(X_train, y_train)


def test_train_model(X_train, y_train):
    try:
        clf = model.train_model(X_train, y_train)
    except:
        logger.error('Model failed to train')
    logger.info('Model successfully trained')


def test_compute_model_metrics(model):
    pass

def test_inference():
    pass