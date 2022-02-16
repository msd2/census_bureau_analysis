from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

from model.model import inference
from model.process_data import process_data


app = FastAPI()


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(alias='capital-gain')
    capital_loss: float = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        allow_population_by_field_nane=True


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post('/predict/')
async def predict_salary(person: Person):

    # Convert Person object to dataframe
    person = pd.DataFrame(person.dict(), index=[0])

    with open('model/encoder.sav', 'rb') as f:
        encoder = pickle.load(f)

    with open('model/trained_model.sav', 'rb') as f:
        model = pickle.load(f)

    with open('model/lb.sav', 'rb') as f:
        lb = pickle.load(f)

    cat_features = ["workclass","education","marital_status","occupation",
                    "relationship","race","sex","native_country"]

    X, y, encoder, lb = process_data(
        X=person,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb)

    pred = inference(model, X)
    pred = lb.inverse_transform(pred)
    print(pred)
    result = {"result": pred.tolist()}
    print(result)

    return result