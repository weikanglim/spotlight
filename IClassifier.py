from yapsy.IPlugin import IPlugin
import abc

class IClassifier(IPlugin, metaclass=abc.ABCMeta):
    
    def pre_process(self, data):
        """
        Pre-processes data. 

        data: Array of text data

        Returns: An array of processed text data.
        """
        return data
    
    @abc.abstractmethod
    def train(self, X, y):
        """
        X: Training vectors, an array of sample data.
        y: Target values.

        Returns: self.
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        X: Set of values to make prediction on.

        Returns: An array of values, one label for each prediction.
        """