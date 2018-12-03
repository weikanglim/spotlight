from yapsy.PluginManager import PluginManager
from IClassifier import IClassifier

class ClassifierManager():

    def __init__(self):
        self.manager = PluginManager(categories_filter={ "Classifiers": IClassifier })
        self.manager.setPluginPlaces(["classifiers"])

        self.manager.locatePlugins()
        self.manager.loadPlugins()

        self.classifiers = {}

    def loadAll(self):
        self.classifiers = {}

        print("Loading classifiers...")
        for classifier in self.manager.getAllPlugins():
            print(classifier.plugin_object)
            self.classifiers[classifier.plugin_object.__class__.__name__] = classifier.plugin_object
