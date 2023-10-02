from flask import Flask, render_template, request , jsonify
import joblib
import numpy as np


model = joblib.load('../models/model.pkl')
scaler = joblib.load('../models/scaler.pkl')

app = Flask(__name__)


def calculate_DPF(family_history):
    if family_history <= 0:
        return 0.0
    elif family_history == 1:
        return 0.25
    elif family_history == 2:
        return 0.5
    elif family_history == 3:
        return 0.75
    else:
        return 1.0


@app.route('/API', methods=["POST"])
def api():
    try:
        Pregnancies = request.args.get('Pregnancies')
        Glucose = request.args.get('Glucose')
        BloodPressure = request.args.get('BloodPressure')
        SkinThickness = request.args.get('SkinThickness')
        Insulin = request.args.get('Insulin')
        BMI = request.args.get('BMI')
        DiabetesPedigreeFunction = request.args.get('DiabetesPedigreeFunction')
        Age = request.args.get('Age')

        input_data = (
            int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(BMI),
            calculate_DPF(int(DiabetesPedigreeFunction)),
            int(Age))

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = model.predict(std_data)

        return jsonify({"prediction": int(prediction[0])})


    except Exception as e:

        print("Exception:", e)

        return jsonify({"error": str(e)})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']

        def calculate_DPF(family_history):
            if family_history <= 0:
                return 0.0
            elif family_history == 1:
                return 0.25
            elif family_history == 2:
                return 0.5
            elif family_history == 3:
                return 0.75
            else:
                return 1.0

        input_data = (
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, calculate_DPF(int(DiabetesPedigreeFunction)),
        Age)

        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # standardize the input data
        std_data = scaler.transform(input_data_reshaped)

        prediction = model.predict(std_data)

        if (prediction[0] == 0):
            pred = 'هذا الشخص ليس مريض سكرى'
        else:
            pred = 'هذا الشخص فد يكون مريض بالسكرى'

        return render_template('results.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
