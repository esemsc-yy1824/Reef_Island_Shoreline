import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date'):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()

        if self.date_column not in X_.columns:
            return X_
        
        X_[self.date_column] = pd.to_datetime(X_[self.date_column])
        X_['date_ordinal'] = X_[self.date_column].apply(lambda x: x.toordinal())

        X_ = X_.drop(columns=[self.date_column])
        
        return X_

class WindTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        u = X_['10m_u_component_of_wind']
        v = X_['10m_v_component_of_wind']
        wind_speed = np.sqrt(u ** 2 + v ** 2)
        wind_dir = np.arctan2(v, u)
        X_['wind_speed'] = wind_speed
        X_['wind_dir_sin'] = np.sin(wind_dir)
        X_['wind_dir_cos'] = np.cos(wind_dir)
        X_ = X_.drop(columns=['10m_u_component_of_wind', '10m_v_component_of_wind'])
        return X_

class NumericScaler(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_scale):
        self.cols_to_scale = cols_to_scale
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols_to_scale])
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.cols_to_scale] = self.scaler.transform(X_[self.cols_to_scale])
        return X_
    
class MonthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, month_col='month'):
        self.month_col = month_col
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['month_sin'] = np.sin(2 * np.pi * X_[self.month_col] / 12)
        X_['month_cos'] = np.cos(2 * np.pi * X_[self.month_col] / 12)
        X_ = X_.drop(columns=[self.month_col])
        return X_
