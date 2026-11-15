from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
bike = pd.read_csv('bike_clean.csv')

app = Flask(__name__)

@app.route('/')
def index():
    bike_name=sorted(bike['bike_name'].unique())
    city=sorted(bike['city'].unique())
    owner=sorted(bike['owner'].unique())
    brand=sorted(bike['brand'].unique())
    bike_options = {}
    for b in brand:
        bike_options[b] = sorted(bike[bike['brand']==b]['bike_name'].unique().tolist())
    return render_template('index.html', bike_names=bike_name, citys=city, owners=owner, brands=brand, bike_options=bike_options)

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form.get('brand')
    bike_name = request.form.get('bike_name')
    city = request.form.get('city')
    kms_driven = float(request.form.get('kms_driven'))
    age = float(request.form.get('age'))
    power = float(request.form.get('power'))
    owner = request.form.get('owner')
    input_data = pd.DataFrame([[bike_name, city, kms_driven, owner, age, power, brand]], columns=['bike_name', 'city', 'kms_driven', 'owner', 'age', 'power', 'brand'])
    prediction = model.predict(input_data)
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)