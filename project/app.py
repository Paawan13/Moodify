import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import jsonify

app = Flask(__name__)
model = pickle.load(open('model\lr_model.pkl', 'rb'))  
print('Model Loaded')


@app.route('/')
def home():
    return render_template('index.html')
print('Step 1 done')
@app.route('/', methods=['POST']) 
def predict():
    try:
    #     data = request.get_json()  # Get JSON data from the POST request
    #     # Extract the input features from the JSON data
    #     danceability = float(data['danceability'])
    #     energy = float(data['energy'])
    #     acousticness = float(data['acousticness'])
    #     liveness = float(data['liveness'])
    #     tempo = float(data['tempo'])
    #     print(tempo)

        # Make the prediction using the model

        if request.method == "POST":
            danceability = request.form.get("fdanceability")
            energy = request.form.get("energy")
            acousticness = request.form.get("acousticness")
            liveness = request.form.get("liveness")
            tempo = request.form.get("tempo")

            feature = np.array([[danceability, energy, acousticness, liveness, tempo]])
            feature.reshape(1,5)
            features = pd.DataFrame([feature],columns = ['danceability','energy','acousticness','liveness','tempo'])
            prediction = model.predict(features)

        # Return the prediction result as JSON
            return jsonify({'prediction_text': 'Song type is: {}'.format(prediction)})
        

    except Exception as e:
        # Print the error for debugging purposes
        print("Error occurred:", e)
        # Return an error message as JSON
        return jsonify({'error': 'An error occurred while processing the request'}), 500

    #int_features = [float(x) for x in request.form.values()]
    #features = [np.array(int_features)]
    #prediction = model.predict(features)
    
    #return render_template('index.html', prediction_text='Song type is: {}'.format(prediction))
    #'Song type is: {}'.format

if __name__ == "__main__":
    app.run()
