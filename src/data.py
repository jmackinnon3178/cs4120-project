import pandas as pd
from pandas import DataFrame as df
from sklearn import model_selection

class Data:
    def __init__(self):
        self.data = pd.read_csv('./data/datasets/student-por.csv',delimiter=';')
        self.X: df = self.data[self.data.columns[:-1]]
        self.y: df = self.data[['G3']]

    def split_data(self, train_ratio: float, test_ratio: float, val_ratio: float, random_state: int):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=1 - train_ratio, random_state=random_state)
        X_val, X_test, y_val, y_test = model_selection.train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_state) 
        
        return X_train, X_test, X_val, y_train, y_test, y_val

