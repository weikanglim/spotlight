from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize as wt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from IClassifier import IClassifier
import nltk

nltk.download('stopwords')
nltk.download('punkt')


"""
This is a sample classifier that wraps the Bernoulli Naive Bayes Classifier packaged with scikit.
"""
class Multinomial(IClassifier):
    
    def __init__(self):
        self.classifier = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', MultinomialNB())])

    def train(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
