from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    features=[np.array(features)]
    prediction=model.predict(features)
    if prediction==0:
        prediction="iris-Setosa"
    elif prediction == 1:
        prediction= "Iris-versicolor"
    elif prediction == 2:
        prediction = "Iris-virginica"
    return render_template('index.html',prediction=(prediction))

if __name__ =="__main__":
    app.run(port=5000,debug=True)

