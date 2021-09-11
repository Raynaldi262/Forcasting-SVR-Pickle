#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv) 
from sklearn.svm import SVR 
import math 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV  
from sklearn.metrics import make_scorer
from flask import Flask,render_template,url_for,request,jsonify
from random import sample
import pickle
import requests
import json 

modelRBF = pickle.load(open('modelRBF.pkl','rb'))
modelLinear = pickle.load(open('modelLinear.pkl','rb'))
modelPoly = pickle.load(open('modelPoly.pkl','rb'))  
# In[ ]:


app = Flask(__name__)
 
@app.route('/')
def index():
    df = pd.read_csv("BrentOilPrices.csv")
    df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y") #format date data to appropriate format
    df.columns=['date', 'price']
    df = df[df['date'] >= '2016-1-1']

    date = []
    for row in df['date'].dt.strftime('%Y/%m/%d'): # each row is a list
        date.append(row)
        
    price = []
    for i in df['price']: # each row is a list
        price.append(i)

    #data untuk chart
    chart = {"renderTo": 'chart_ID', "type": 'line', "height": 500,}
    series = [{"name": 'Per Barrel ($)', "data": price}]
    title = {"text": 'Harga Minyak Bumi'}
    xAxis = {"categories": date}
    yAxis = {"title": {"text": 'Harga'}}

    length = len(price) #panjang data yg dipakai

    return render_template('index.html', chartID='chart_ID', chart=chart, series=series, title=title, xAxis=xAxis, 
        yAxis=yAxis, date=date ,price=price,len=length)

@app.route('/predicted',methods=['POST'])
def predicted():
    if request.method == 'POST':
        startD = request.form['start']
        endD = request.form['end']

    #input dataset
        df = pd.read_csv("BrentOilPrices.csv")
        df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y") #format date data to appropriate format
        df.columns=['date', 'price']
        df = df[df['date'] >= '2016-1-1']

        x =np.array(range(955))
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=None, shuffle= False)
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1,1) 

        df_t = pd.read_csv('brent-daily_csv.csv')
        df_t['date'] = pd.to_datetime(df_t['date'], format="%m/%d/%Y") #format date data to appropriate format
        df_t['price'] = np.array(df_t['price'], dtype=float)

        inputD = df_t[(df_t['date' ] >= startD) & (df_t['date' ] <= endD)]

    #menyiapkan data untuk label di chart
        dateI = []

        for row in inputD['date'].dt.strftime('%Y/%m/%d'): # each row is a list
            dateI.append(row)
        
        priceI = []
        
        for i in inputD['price']: # each row is a list
                priceI.append(i)    
    #menyiapkan data untk prediksi
        x_iD = np.array(range(955, 955+len(inputD)))
        y_iD = inputD['price']
        x_iD = x_iD.reshape(-1, 1) 

        #memlakukan prediksi
        rbfr = rbf(X_train,y_train,x_iD)
        liner = linear(X_train,y_train,x_iD)
        polyr = poly(X_train,y_train,x_iD)

        #menghtung mape
        mapeRbf = mape(y_iD,rbfr)
        mapeLine = mape(y_iD,liner)
        mapePoly = mape(y_iD,polyr)

        #hasil prediksi dibuat array
        rbffff = []
        for i in rbfr: # each row is a list
                rbffff.append(round(i,2))

        linerrr = []
        for i in liner: # each row is a list
                linerrr.append(round(i,2))

        polyrrr = []
        for i in polyr: # each row is a list
                polyrrr.append(round(i,2))

        length = len(y_iD) #panjang data yg dipakai

        #membuat chart
        chart = {"renderTo": 'predict1', "type": 'line', "height": 500,} 
        series = [{"name": 'Harga aktual', "data": priceI}, {"name": 'Prediksi RBF', "data": rbffff}
                    ,{"name": 'Prediksi Linear', "data": linerrr},{"name": 'Prediksi Polynomial', "data": polyrrr}]
        title = {"text": 'Hasil Prediksi Harga Minyak Bumi'}
        xAxis = {"categories": dateI}
        yAxis = {"title": {"text": 'Harga'}}

    return render_template('result.html', chartID='predict1', chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis, mapeR=mapeRbf
        ,mapeL=mapeLine,mapeP=mapePoly,len=length,date=dateI,Aprice=priceI,Rprice=rbffff,Lprice=linerrr,Pprice=polyrrr)


def rbf(x,y,z):
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.0002, epsilon=0.005)
    svr_rbf.fit(x,y)
    pred = svr_rbf.predict(z)
    return pred

def linear(x,y,z):
    svr_linear = SVR(kernel='linear', C=0.001, epsilon=1) 
    svr_linear.fit(x,y)
    pred = svr_linear.predict(z)
    return pred

def poly(x,y,z):
    svr_poly = SVR(kernel= 'poly', degree=2, gamma=1, coef0=2, C=0.001, epsilon=0.1)
    svr_poly.fit(x,y)
    pred = svr_poly.predict(z)
    return pred

def mape(a, b): 
    mask = a != 0
    x = round(((np.fabs(a - b)/a)[mask].mean())*100,2)
    return x

if __name__ == '__main__':
    app.run(debug=True)

