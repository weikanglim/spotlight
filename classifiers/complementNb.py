from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize as wt
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from IClassifier import IClassifier
import nltk

nltk.download('stopwords')
nltk.download('punkt')


"""
This is a sample classifier that wraps the Bernoulli Naive Bayes Classifier packaged with scikit.
"""
class Complement(IClassifier):
    
    def __init__(self):
        self.classifier = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', ComplementNB())])

    def pre_process(self, data):
        processed_data = []
        stemmer = PorterStemmer()
        for text_data in data:
            # make words lowercase
            text_data = text_data.lower()
            # tokenising
            tokenized_data = wt(text_data)
            # remove stop words and stemming
            text_processed = []
            for word in tokenized_data:
                if word not in set(stopwords.words('english')):
                 text_processed.append(stemmer.stem(word))
            processed_data.append(" ".join(text_processed))

        return processed_data

    def train(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
