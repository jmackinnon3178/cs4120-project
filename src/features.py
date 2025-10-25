from sklearn import preprocessing, compose
from data import Data

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

# X_train_encoded = preprocessor.fit_transform(X_train)
# X_test_encoded = preprocessor.transform(X_test)
#
# y_train_labeled = (y_train >= 10).astype(int)
# y_test_labeled = (y_test >= 10).astype(int)

# data = Data()
# X_train, X_test, y_train, y_test = data.split_data(1)

# feature_cols = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2']
