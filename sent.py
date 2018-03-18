from textblob import TextBlob
import pandas as pd
import datetime as dt


def analyse(x):
    text=x
    blob = TextBlob(text)

    for sentence in blob.sentences:
        pol=(sentence.sentiment.polarity)
    return pol