### Installation
- clone the repo
- enter cs4210-project directory
- create python virtual environment in cs4210-project directory
- enter virtual environment
- run pip -r requirements.txt to install dependencies

## Running
- enter cs4210-project directory
- start mlflow server with:

```
mlflow server --host 127.0.0.1 --port 8080 --artifacts-destination ./models
```

- run cross validation on baseline models and log to mlflow with

```
python3 src/train_baselines.py
```
