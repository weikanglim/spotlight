# First-time Setup
Pre-requisites:
1. Python 3.X
2. npm

To initialize and obtain all required package dependencies:

- Backend

```
pip install pipenv
pipenv install
```

- Frontend

```
cd www
npm install
```

# Development

To run a local development server:

1. Start the backend Flask server.
```
pipenv run python app.py
```

2. Start the frontend Angular server.
```
cd www
ng serve
```

3. Open a browser and the server will be running on http://localhost:4200/. Enjoy! 