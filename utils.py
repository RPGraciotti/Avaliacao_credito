# aqui você inclui as funções de utilidades e os pacotes necessários
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class Transformador(BaseEstimator, TransformerMixin):
    def __init__(self, quant_cols, cat_cols):
        self.quant_cols = quant_cols
        self.cat_cols = cat_cols
        self.enc = OneHotEncoder()
        self.scaler = MinMaxScaler()

    def fit(self, X, y = None ):
        self.enc.fit(X[self.cat_cols])
        self.scaler.fit(X[self.quant_cols])
        return self 

    def transform(self, X, y = None):
      
      X_categoricas = pd.DataFrame(data = self.enc.transform(X[self.cat_cols]).toarray(),
                                  columns = self.enc.get_feature_names(self.cat_cols))
      
      X_continuas = pd.DataFrame(data = self.scaler.transform(X[self.quant_cols]),
                                  columns = self.quant_cols)
      
      X = pd.concat([X_continuas, X_categoricas], axis = 1)

      return X