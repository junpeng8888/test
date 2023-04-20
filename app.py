"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
from gevent.pywsgi import WSGIServer


#from sklearn.model_selection import train_test_split
#from scipy.sparse.linalg import svds

from surprise import Dataset,Reader
from surprise.model_selection import train_test_split
from surprise import SVD, KNNWithMeans
from surprise import accuracy

app = Flask(__name__)
model=pickle.load(open('model2.pkl','rb'))

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/')
@app.route('/predict', methods=['GET'])
def home():
    #input1 = request.args.get('input1', type=float)
 
    ele_ratings_df = pd.read_csv('Testing1.csv')

    ele_ratings_df.drop_duplicates(keep='first',inplace=True)

    users_count = ele_ratings_df.ApplicationUserId.value_counts()

    ele_ratings_df_sample = ele_ratings_df[ele_ratings_df.ApplicationUserId.isin(users_count[users_count >= 1].index)]
   
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(ele_ratings_df_sample[['ApplicationUserId', 'ProductId', 'Rating']], reader)

    trainset, testset = train_test_split(data, test_size=.30, random_state=123)
   

    uid = "A1ORUSHRRG0VWN"

    knn_test_pred = model.test(testset)
    pred = pd.DataFrame(knn_test_pred)
   
    pred2=pred[pred['uid'] == uid][['iid', 'r_ui','est']].sort_values(by = 'r_ui', ascending = False).head(10)
    pred2=pred2['iid']
   
    #pred = model.predict(uid, iid, r_ui=0.0, verbose=True)
 
    #return render_template('index.html',prediction_text="{}".format( pred2.to_string(index=False)))
    return {'result':pred2.to_string(index=False)}

@app.route('/pre', methods=['GET'])
def predict():
    ele_ratings_df = pd.read_csv('Testing1.csv')

    ele_ratings_df.drop_duplicates(keep='first',inplace=True)

    users_count = ele_ratings_df.ApplicationUserId.value_counts()

    ele_ratings_df_sample = ele_ratings_df[ele_ratings_df.ApplicationUserId.isin(users_count[users_count >= 1].index)]
   
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(ele_ratings_df_sample[['ApplicationUserId', 'ProductId', 'Rating']], reader)

    trainset, testset = train_test_split(data, test_size=.30, random_state=123)
   

    uid = "A1ORUSHRRG0VWN"

    knn_test_pred = model.test(testset)
    pred = pd.DataFrame(knn_test_pred)
   
    pred2=pred[pred['uid'] == uid][['iid', 'r_ui','est']].sort_values(by = 'r_ui', ascending = False).head(10)
    pred2=pred2['iid']
   
    #pred = model.predict(uid, iid, r_ui=0.0, verbose=True)
 
    #return render_template('index.html',prediction_text="{}".format( pred2.to_string(index=False)))
    return {'result':pred2.to_string(index=False)}

if __name__ == '__main__':
    #import os
    #HOST = os.environ.get('SERVER_HOST', 'localhost')
    #try:
    #    PORT = int(os.environ.get('SERVER_PORT', '5555'))
    #except ValueError:
    #    PORT = 5555
    #app.run(HOST, PORT)
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
