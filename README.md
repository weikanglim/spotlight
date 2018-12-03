# First-time Setup
Pre-requisites:
1. Python 3.X
2. npm

To initialize and obtain all required package dependencies:

- Backend

Run the following commands:

```
pip install pipenv
pipenv install
```

- Frontend

```
cd www
npm install

```

# Hosting

To run a local server:

1. Start the backend Flask server.
```
pipenv run python app.py
```

2. Start the frontend Angular server in a separate window.
```
cd www
ng serve
```

3. Open a browser and the server will be running on http://localhost:4200/.


# Development

Pre-requisite:
To ensure that the same libraries are available on the server as well as your local development environment, use the Pipfile included to ensure all packages are available. Run `pipenv shell` to obtain an environment that mimicks the server.

1. Create your classifier that implements the `IClassifier` interface for predict, train and pre_process. See *IClassifier.py* for documentation, and other sample files included in the *classifiers* folder.
2. Copy your classifier into the *classifiers* folder.
3. Add a *MyClassifier.yapsy-plugin* file to describe your classifier.
3. When you start Spotlight, the classifier will automatically be loaded for testing and training. Enjoy!