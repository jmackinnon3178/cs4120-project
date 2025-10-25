import pandas as pd
from pandas import DataFrame as df
from sklearn import model_selection

class Data:
    def __init__(self):
        self.data = pd.read_csv('../data/datasets/student-mat.csv',delimiter=';')
        self.X: df = self.data[self.data.columns[:-1]]
        self.y: df = self.data[['G3']]

    def split_data(self, random_state: int) -> tuple[df, df, df, df]:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, random_state=random_state)
        return X_train, X_test, y_train, y_test
