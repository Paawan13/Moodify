import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for,redirect
import pickle



app = Flask(__name__)
model = pickle.load(open('model\lr_model.pkl', 'rb'))  


@app.route('/', methods= ['GET','POST'])
def home():
    if request.method == "POST":
            danceability = float(request.form.get('danceability'))
            energy = float(request.form.get('energy'))
            acousticness = float(request.form.get('acousticness'))
            liveness = float(request.form.get('liveness'))
            tempo = float(request.form.get('tempo'))
            feature = np.array([danceability, energy, acousticness, liveness, tempo])
            feature.reshape(1,5)
            features = pd.DataFrame([feature],columns = ['danceability','energy','acousticness','liveness','tempo'])
            prediction = model.predict(features)
            prediction = prediction[0]
            return redirect(url_for('predict',prediction = prediction))
    return render_template('index.html')



@app.route('/predict/<prediction>')
def predict(prediction):
    mydict = {'0':'Sad','1':'Happy','2':'Energetic','3':'Calm'}
    result = mydict[prediction]         
    print(f" Your song type is {result}")
    return render_template('output.html', prediction = result)
    

if __name__ == "__main__":
    app.run(debug = True)
