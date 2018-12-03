import re
from datetime import datetime
from os import listdir
from os.path import basename, dirname, isfile, join

from flask import (Flask, json, jsonify, render_template, request, send_file,
                   send_from_directory)
from werkzeug.utils import secure_filename

import pandas as pd
from ClassifierManager import ClassifierManager
from Exceptions import InputError
from flask_cors import CORS
from sklearn.externals import joblib
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score)

STORE_LOCATION = 'store'

app = Flask(__name__)
app.config['STORE_LOCATION'] = STORE_LOCATION
CORS(app)

classifierManager = ClassifierManager()
classifierManager.loadAll()

@app.route('/models')
def get_models():
    model_store = app.config['STORE_LOCATION']
    models = [f for f in listdir(model_store) if (isfile(join(model_store, f)) and f != "empty.md")]
    print(models)
    results = []
    for model in models:
        result = {}
        result['name'] = model
        result['url'] = "http://" + request.host + "/models/" + model
        results.append(result)

    response = app.response_class(
        response=json.dumps(results),
        mimetype='application/json'
    )
    return response

@app.route('/models/<model_name>')
def get_stored_model(model_name):
    model_file_path = join(app.config['STORE_LOCATION'], model_name)
    if not isfile(model_file_path):
        return 404

    return send_from_directory(app.config['STORE_LOCATION'], model_name)

@app.route('/models/train', methods = ['POST'])
def train_model():
    if not request.files:
        return 400

    data = request.files['dataFile']
    model_name = request.form.get('modelName')

    if 'modelFile' in request.files.keys():
        model = load_model(request.files['modelFile'])
    else:
        existing_model_name = request.form.get('existingModelName')
        if not existing_model_name:
            return 400

        model_file_path = join(app.config['STORE_LOCATION'], existing_model_name)
        if not isfile(model_file_path):
            return 404

        model = load_model(model_file_path)

    train(data, model)
    results = save_model(model, model_name)
    
    return jsonify(results)

@app.route('/models/<model_name>/predict', methods =['POST'])
def predict_model(model_name):
    print("Predict called")
    model_file_path = join(app.config['STORE_LOCATION'], model_name)
    if not isfile(model_file_path):
        return 404
    
    model = load_model(model_file_path)
    dataframe = read_files(request.files['dataFile'])
    text = dataframe.iloc[:, 1]
    x = model.pre_process(dataframe.iloc[:, 1])
    y = dataframe.iloc[:, 0]
    y_predictions = model.predict(x)

    classifications=[]

    for data, label, prediction in zip(text.tolist(), y.tolist(), y_predictions.tolist()):
        result = {}
        result['text'] = data
        result['label'] = label
        result['prediction'] = prediction
        result['result'] =  "Positive" if label == prediction else "Negative"
        classifications.append(result)

    return jsonify(
        modelName = model_name,
        accuracy = accuracy_score(y, y_predictions),
        classificationMatrix = classification_report_data(classification_report(y, y_predictions)),
        classifications = classifications,
        modelUri = "http://" + request.host + "/models/" + model_name)


@app.route('/models/<model_name>/predictOne', methods =['POST'])
def predict_model_one(model_name):
    model_file_path = join(app.config['STORE_LOCATION'], model_name)
    if not isfile(model_file_path):
        return 404
    
    model = load_model(model_file_path)
    content = request.get_json(force=True)
    text = content['text']
    print(text)
    dataframe = pd.DataFrame([text])
    x = model.pre_process([text])
    y = model.predict(x)

    return jsonify(text = text, prediction = y[0])

@app.route('/classifiers')
def get_classifiers():
    return jsonify([
        {'id': classifier_name, 'name': classifier_name}
        for classifier_name in classifierManager.classifiers.keys()])

@app.route('/classifiers/<classifier_name>')
def get_classifier(classifier_name):
    if not classifier_name in classifierManager.classifiers.keys():
        return 404
    
    return jsonify(name = classifier_name)

@app.route('/classifiers/<classifier_name>/train', methods =['POST'])
def train_classifier(classifier_name):
    if not classifier_name in classifierManager.classifiers.keys():
        return 404
    if not request.files or not request.files['dataFile']:
        return 400

    model_name = request.form.get('modelName')
    data = request.files['dataFile']

    classifier = classifierManager.classifiers[classifier_name]
    train(data, classifier)
    results = save_model(classifier, model_name)

    return jsonify(results)

def read_files(file_obj):
    file_type = file_obj.filename[file_obj.filename.rfind('.'):]
    dataset = None

    if (file_type == '.json'):
        dataset = pd.read_json(file_obj)
    elif (file_type == '.csv'):
        dataset = pd.read_csv(file_obj, encoding="ISO-8859-1")

    return dataset

def parse_input_data(input_data):
    list_of_lines = str(input_data).strip().splitlines()
    dataset = pd.DataFrame(list_of_lines)
    return dataset

def classification_report_data(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:]:
        print(line)
        if not line:
           continue
        else:
            row = {}
            row_data = line.split('      ')
            if(row_data[0]==''):
                row['class'] = row_data[1].strip()
                row['precision'] = float(row_data[2])
                row['recall'] = float(row_data[3])
                row['f1_score'] = float(row_data[4])
                row['support'] = float(row_data[5])
            else:
                row['class'] = row_data[0].strip()
                row['precision'] = float(row_data[1])
                row['recall'] = float(row_data[2])
                row['f1_score'] = float(row_data[3])
                row['support'] = float(row_data[4])
        report_data.append(row)
    return report_data

def train(data, model):
    dataframe = read_files(data)

    x = model.pre_process(dataframe.iloc[:, 1])
    y = dataframe.iloc[:, 0]

    model.train(x, y)

def save_model(model, model_name):
    joblib.dump(model, join(app.config['STORE_LOCATION'], model_name))
    return {
        "modelName": model_name,
        "modelUrl": "http://" + request.host + "/models/" + model_name,
    }

def load_model(file_name):
   model= joblib.load(file_name)
   return model

if __name__ == '__main__':
    app.run(debug=True)
