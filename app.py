from flask import Flask, render_template,request,json,send_file
from werkzeug.utils import secure_filename
import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt
from autocorrect import spell
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score,recall_score,precision_recall_fscore_support
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/classification', methods = ['POST'])
def text_classifier():
   if request.method == 'POST':
      if request.files :
          input_data = request.files['input_file']
          print(input_data.filename)
          data=read_files(input_data)
          X = data_pre_process(data)
          y = data.iloc[:, 0]
          X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
          classifier = pick_classifier(request.form.get('classifier'))
          clf=train_model(X_train, y_train,classifier)
          save_model(clf, str(request.form.get('classifier'))+"_classifier")
          clf=load_model(str(request.form.get('classifier'))+"_classifier")
          y_pred=predict(X_test,clf)
          print('accuracy %s' % accuracy_score(y_pred, y_test))
          print(classification_report(y_test, y_pred))
          if str(request.form.get('output')) == 'File_Download':
             return send_file(str(request.form.get('classifier'))+"_classifier",attachment_filename="trained_model",as_attachment='true')
          elif str(request.form.get('output')) == 'Classification_Report':
             accuracy= accuracy_score(y_pred, y_test)
             data = {'accuracy': accuracy, 'classification report': classification_report_data(classification_report(y_test, y_pred))}
             response = app.response_class(
                  response=json.dumps(data),
                  mimetype='application/json'
             )
             return response
          else:
              return "output type is not specified"
      else:
          return "no file uploaded"


@app.route('/prediction', methods=['POST'])
def text_prediction():
   if request.method == 'POST':
      if request.files:
          input_data=request.files['input_file']
          input_model=request.files['model_file']
          print(input_data.filename)
          data=read_files(input_data)
          X = data_pre_process(data)
          y = data.iloc[:, 0]
          X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
          clf=load_model(input_model)
          y_pred=predict(X_test,clf)
          if str(request.form.get('output')) == 'Classification_Report':
             accuracy= accuracy_score(y_pred, y_test)
             data = {'accuracy': accuracy, 'classification report': classification_report_data(classification_report(y_test, y_pred))}
             response = app.response_class(
                  response=json.dumps(data),
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
        for i in range(dataset.shape[0]):
            text_data = dataset.iloc[i, 1]
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