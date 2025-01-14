
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chords_prog_proj.interface.main import pred
from chords_prog_proj.ml_logic.registry import load_model

import random
import numpy as np
import json

app = FastAPI()
# To avoid loading the model at each GET request:
# Load the model into memory on startup and store it in a global variable in app.state,
# which is kept in memory and accessible across all routes
app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
    song: str,
    n_chords: int,
    randomness: int
):
    """
    Make a prediction using the latest trained model
    """

    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """

    song = song.split(',')
    n_chords = int(n_chords)
    randomness = int(randomness)

    n_chords_pred = pred(
        song = song,
        n_chords = n_chords,
        randomness = randomness,
        model = app.state.model
        )

    # ⚠️ fastapi only accepts simple python data types as a return value
    # among which dict, list, str, int, float, bool
    # in order to be able to convert the api response to json

    return {'predicted chords': n_chords_pred}

@app.get("/test")
def root():
    return dict(greeting="Hello Test")
