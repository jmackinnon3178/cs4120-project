import pandas as pd
from pandas import DataFrame as df
from sklearn import model_selection

class Data:
    def __init__(self):
        self.data = pd.read_csv('./data/datasets/student-por.csv',delimiter=';')
        self.X: df = self.data[self.data.columns[:-1]]
        self.y: df = self.data[['G3']]

    def split_data(self, test_size: float, random_state: int) -> tuple[df, df, df, df]:
        X_train, X_test, y_train, y_test = \
            model_selection.train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
