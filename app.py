import pickle
from flask import Flask,request,app,url_for,jsonify,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.get_json()
    input_data=np.array(list(data.values())).reshape(1,-1)
    prediction=regmodel.predict(input_data)
    return jsonify({'prediction': float(prediction[0])})