
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

engmodel='rfcengmodel.pkl'
nepmodel='NBnepmodel.pkl'
vocab='tfidf.pkl'

def predict_engcategory(text):
    classifier =pickle.load(open(engmodel, "rb"))
    tfidf=CountVectorizer(max_features=5000)
    text=tfidf.fit_transform(text)
    y_pred= classifier.predict(text)
    return y_pred

def predict_nepcategory(news):
    classifier =pickle.load(open(nepmodel, "rb"))
    tfidf=CountVectorizer(max_features=5000)
    news=tfidf.fit_transform(news)
    y_pred1= classifier.predict(news)
    return y_pred1
    
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
@app.route('/english', methods=['GET', 'POST'])
def english():
    if flask.request.method == 'GET':
        return(flask.render_template('english.html'))
    if flask.request.method == 'POST':  
            to_predict_list = flask.request.form.to_dict() 
            news = list(to_predict_list.values()) 
            result=predict_engcategory(news)
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
        
    return flask.render_template('english.html', prediction=pred) 

@app.route('/nepali', methods=['GET', 'POST'])
def nepali():
    if flask.request.method == 'GET':
        return(flask.render_template('nepali.html'))
    if flask.request.method == 'POST':  
            to_predict_list = flask.request.form.to_dict() 
            news = list(to_predict_list.values()) 
            result=predict_nepcategory(news)
            if result == [0]:
                pred = "Entertainment News"
            elif result == [1]:
                pred = "Business News"
            elif result == [2]:
                pred = "Sports News"
        
    return flask.render_template('nepali.html', predict=pred) 

if __name__ == "__main__":
    app.run(debug=True)