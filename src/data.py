import pandas as pd
from pandas import DataFrame as df
from sklearn import model_selection
from ucimlrepo import fetch_ucirepo

class Data:
    def __init__(self):
        self.dataset = fetch_ucirepo(id=320)
        self.X = self.dataset.data.features
        self.y = self.dataset.data.targets
        self.data = pd.concat([self.X, self.y], axis=1)

    def split_data(self, train_ratio: float, test_ratio: float, val_ratio: float, random_state: int):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=1 - train_ratio, random_state=random_state)
        X_val, X_test, y_val, y_test = model_selection.train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_state) 
        
        return X_train, X_test, X_val, y_train, y_test, y_val

