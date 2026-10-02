from flask import Flask, request, jsonify,render_template,app,url_for
import numpy as np
import pickle

app = Flask(__name__)

regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


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
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__ == '__main__':
    app.run(debug=True)