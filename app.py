import pickle
import os
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()

app=Flask(__name__)
#loading the model
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict_api",methods = ['post'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.fit_transform(np.array(list(data.values())).reshape(1,-1))
    output =regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)





