from sklearn import preprocessing, compose
from pandas import DataFrame as df

class Features:
    def __init__(self) -> None:
    
        self.__feature_cols_numeric = ['age','failures','absences','G1','G2']

        self.__feature_cols_categorical = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']

        self.__feature_cols_ordinal = ['Medu','Fedu','traveltime','studytime','famrel','freetime','goout','Dalc','Walc','health']

        self.preprocessor: compose.ColumnTransformer = compose.ColumnTransformer(
            transformers=[
                ('num', preprocessing.StandardScaler(), self.__feature_cols_numeric),
                ('cat', preprocessing.OneHotEncoder(), self.__feature_cols_categorical),
                ('ord', preprocessing.OrdinalEncoder(), self.__feature_cols_ordinal)
            ]
        )

    def grade_to_pass_fail(self, target: df) -> df:
        return (target >= 10).astype(int)
