import pytest
from fastapi.testclient import TestClient
import pandas as pd
import json
from main import app, Person

client = TestClient(app)

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