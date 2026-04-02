import numpy as np
import pandas as pd


EPSILON = 1e-6


class SklearnFeatureEngineer:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "sensor_11" in X.columns and "sensor_12" in X.columns:
            X["sensor_11_12_gap"] = X["sensor_11"] - X["sensor_12"]

        if "sensor_20" in X.columns and "sensor_21" in X.columns:
            X["sensor_20_21_ratio"] = X["sensor_20"] / (np.abs(X["sensor_21"]) + EPSILON)

        if "sensor_15" in X.columns:
            X["sensor_15_squared"] = X["sensor_15"] ** 2

        return X