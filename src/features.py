from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from pandas import DataFrame as df

feature_cols_numeric = ['age','failures','absences','G1','G2']
feature_cols_categorical = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
feature_cols_ordinal = ['Medu','Fedu','traveltime','studytime','famrel','freetime','goout','Dalc','Walc','health']

lr_prep_stdscaler = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_cols_numeric),
        ('cat', OneHotEncoder(drop='if_binary'), feature_cols_categorical),
        ('ord', OrdinalEncoder(), feature_cols_ordinal)
    ],
)

dt_prep = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='if_binary'), feature_cols_categorical),
        ('ord', OrdinalEncoder(), feature_cols_ordinal),
        ('num', 'passthrough', feature_cols_numeric)
    ],
)
def grade_to_pass_fail(target: df) -> df:
    return (target >= 10).astype(int)
