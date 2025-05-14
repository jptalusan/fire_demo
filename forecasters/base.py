"""
Base Class for all forecasters. All classes must implement the following methods:

1. fit -- fit a regression model to data
2. predict -- sample instances based on the forecasting model
3. get_regression_expr -- get a patsy expression for the regression.
4. update_model_stats -- store the model likelihood, AIC score for easy access
"""


class Forecaster:

    def __init__(self):
        self.model_params = None
        self.name = None
        self.model_stats = None

    def fit(self):
        pass

    def predict(self):
        pass

    def get_regression_expr(self):
        pass

    def update_model_stats(self):
        pass

    def get_likelihood(self):
        pass
