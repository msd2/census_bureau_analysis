import requests
import pandas as pd

def sample_person_under():
    df = pd.read_csv('data/cleaned_census_data.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df = df[df['salary']=='<=50K']
    data_row = df.sample(1)
    data_row = data_row.to_dict('records')[0]
    return data_row

example = sample_person_under()

response = requests.post(
    'https://census-bureau-analysis-udacity.herokuapp.com/predict/',
    json=example)

print(response.status_code)
print(response.json())