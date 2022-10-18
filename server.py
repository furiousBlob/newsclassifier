
from unittest import result
import flask
import pickle
from flask import request
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Use pickle to load in the pre-trained model.
model='rfcengmodel.pkl'
vocab='tfidf.pkl'


def predict_category(text):
    classifier =pickle.load(open(model, "rb"))
    tfidf=CountVectorizer()
    text=tfidf.fit_transform(text)
    y_pred= classifier.predict(text)
    return y_pred

    
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':  
            to_predict_list = request.form.to_dict() 
            news = list(to_predict_list.values()) 
            result=predict_category(news)
            if result == [0]:
                pred = "Business News"
            elif result == [1]:
                pred = "Tech News"
            elif result == [2]:
                pred = "Politics News"
            elif result == [3]:
                pred = "Sports News"
            elif result == [4]:
                pred = "Entertainment News"
        
    return flask.render_template('main.html',
                                    original_input={'News Text':news},
                                    result=pred)

if __name__ == "__main__":
    app.run(debug=True)