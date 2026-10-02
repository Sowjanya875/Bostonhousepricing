from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.get_json()

    input_data = np.array([[
        data['CRIM'],
        data['ZN'],
        data['INDUS'],
        data['CHAS'],
        data['NOX'],
        data['RM'],
        data['AGE'],
        data['DIS'],
        data['RAD'],
        data['TAX'],
        data['PTRATIO'],
        data['B'],
        data['LSTAT']
    ]])


    scaled_data = scaler.transform(input_data)
    prediction = regmodel.predict(scaled_data)


    return jsonify({
        'prediction': prediction[0]
    })


if __name__ == '__main__':
    app.run(debug=True)