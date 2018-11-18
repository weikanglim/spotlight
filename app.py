import re

from flask import Flask, json, render_template, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import nltk
import pandas as pd
from autocorrect import spell
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize as wt
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/classification', methods = ['POST'])
def text_classifier():
       if request.method == 'POST':
        # Use Case 1 :from scratch with input file : output : trained model/classification Report
          if request.files :
              input_data = request.files['input_file']
              print(input_data.filename)
              dataframe=read_files(input_data)
              data_list=[]
              for i in range(dataframe.shape[0]):
                  data_list.append(dataframe.iloc[i, 1])

              X = data_pre_process(data_list)
              y = dataframe.iloc[:, 0]
              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
              classifier = pick_classifier(request.form.get('classifier'))
              clf = train_model(X_train, y_train, classifier)
              save_model(clf, str(request.form.get('classifier')) + "_classifier")
              clf = load_model(str(request.form.get('classifier')) + "_classifier")
              y_pred = predict(X_test, clf)
              print('accuracy %s' % accuracy_score(y_pred, y_test))
              print(classification_report(y_test, y_pred))
              if str(request.form.get('output')) == 'File_Download':
                  return send_file(str(request.form.get('classifier')) + "_classifier",
                                   attachment_filename="trained_model", as_attachment='true')
              elif str(request.form.get('output')) == 'Classification_Report':
                  accuracy = accuracy_score(y_pred, y_test)
                  data = {'accuracy': accuracy,
                          'classification report': classification_report_data(classification_report(y_test, y_pred))}
                  response = app.response_class(
                      response=json.dumps(data),
                      mimetype='application/json'
                  )
                  return response
              else:
                  return "output type is not specified"
          #elif request.form.get('input_data'):
              #input_data = request.form.get('input_data')
              #data = parse_input_data(input_data)
          else :
              return "Input data is not provided"


@app.route('/prediction', methods=['POST'])
def text_prediction():
   if request.method == 'POST':
          # Use Case 2 :Trained model and input file : output : classification Report
          if request.files and not request.form.get('input_data'):
              print("in predict 1")
              input_data=request.files['input_file']
              input_model=request.files['model_file']
              print(input_data.filename)
              dataframe=read_files(input_data)
              data_list = []
              for i in range(dataframe.shape[0]):
                  data_list.append(dataframe.iloc[i, 1])
              clf = load_model(input_model)
              X = data_pre_process(data_list)
              y = dataframe.iloc[:, 0]
              print("in predict 1 y"+str(len(y)))
              y_pred = predict(X, clf)
              print("in predict 1 y_pred"+str(len(y_pred)))
              if str(request.form.get('output')) == 'Classification_Report':
                  accuracy= accuracy_score(y_pred, y)
                  data = {'accuracy': accuracy, 'classification report': classification_report_data(classification_report(y, y_pred))}
                  response = app.response_class(
                        response=json.dumps(data),
                        mimetype='application/json'
                   )
                  return response
              else:
                  return "output type is not specified"
          # Use Case 3 :Trained model and input text data  : output : Category prediction
          elif request.files and request.form.get('input_data'):
              input_data = request.form.get('input_data')
              input_model = request.files['model_file']
              dataframe = parse_input_data(input_data)
              data_list = []
              for i in range(dataframe.shape[0]):
                  data_list.append(dataframe.iloc[i, 0])
              clf = load_model(input_model)
              X = data_pre_process(data_list)
              print(X)
              y_pred = predict(X, clf)
              list_out=[]
              for data,pred in zip(data_list,y_pred.tolist()):
                  list_out.append(data+' : '+pred)
              if str(request.form.get('output')) == 'Predict_Category':
                  response = app.response_class(
                        response=json.dumps(list_out),
                        mimetype='application/json'
                   )
                  return response
              else:
                  return "output type is not specified"
          else:
              return "no file uploaded"




def read_files(file_obj):
        file_type = file_obj.filename[file_obj.filename.rfind('.'):]
        print(file_type)
        dataset=None
        if (file_type == '.json'):
            dataset = pd.read_json(file_obj)
        elif (file_type == '.csv'):
            dataset = pd.read_csv(file_obj, encoding="ISO-8859-1")
            # print(dataset.shape[0])
        return dataset


def parse_input_data(input_data):
    list_of_lines = str(input_data).strip().splitlines()
    dataset = pd.DataFrame(list_of_lines)
    return dataset

def pick_classifier(classifier_name):
    classifier=None
    if classifier_name == 'M':
        classifier = MultinomialNB()
    elif classifier_name == 'C':
        classifier = ComplementNB()
    elif classifier_name == 'B':
        classifier = BernoulliNB()
    return classifier

def data_pre_process(dataset):
        data = []
        stemmer = PorterStemmer()
        for text_data in dataset:
            # remove non alphabatic characters
            text_data = re.sub('[^A-Za-z]', ' ', text_data)
            # make words lowercase
            text_data = text_data.lower()
            # tokenising
            tokenized_data = wt(text_data)
            # remove stop words and stemming
            text_processed = []
            for word in tokenized_data:
                if word not in set(stopwords.words('english')):
                 text_processed.append(spell(stemmer.stem(word)))
            data.append(" ".join(text_processed))
        return data
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
                row['class'] = row_data[1]
                row['precision'] = float(row_data[2])
                row['recall'] = float(row_data[3])
                row['f1_score'] = float(row_data[4])
                row['support'] = float(row_data[5])
            else:
                row['class'] = row_data[0]
                row['precision'] = float(row_data[1])
                row['recall'] = float(row_data[2])
                row['f1_score'] = float(row_data[3])
                row['support'] = float(row_data[4])
        report_data.append(row)
    return report_data

def train_model(X_train, y_train, classifier):
    nb = Pipeline([('vect', TfidfVectorizer()),
                   ('clf', classifier),
                   ])
    return nb.fit(X_train, y_train)

def predict(test_data,clf):
        predicted = clf.predict(test_data)
        return predicted


def save_model(classifier, file_name):
    print(file_name)
    joblib.dump(classifier,file_name)

def load_model(file_name):
   model= joblib.load(file_name)
   return model

if __name__ == '__main__':
    app.run( debug=True)
