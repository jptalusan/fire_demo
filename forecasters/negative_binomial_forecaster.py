"""
@Author - Ayan Mukhopadhyay
Negative Binomial Regression Forecaster -- Inherits from Forecaster class
The model is based on NB2 Regression Model which assumes the form
'variance = mean + alpha * mean**2', where mean is the expected value
under a standard Poisson Model.
"""

from forecasters.base import Forecaster
import statsmodels.formula.api as smf
from patsy import dmatrices
from forecasters.poisson_reg_forecaster import Poisson_Model
import statsmodels.api as sm
import pandas as pd
from copy import deepcopy
from scipy.special import gamma
import numpy as np


def get_log_lin_lambda(df, params):
    def rate_calc(x):
        temp = 0
        for key, val in params.items():
            try:
                temp += params[key]*x[key]
            except KeyError:
                # intercept is not in the features but always 1
                temp += params[key] * 1
        # to ensure numerical stability, put a very small rate for units that have 0 rate
        if temp == 0:
            temp = 1e-10

        return np.exp(temp)

    df['sample'] = df.apply(rate_calc, axis=1)
    return df


class Negative_Binomial_Model(Forecaster):

    def __init__(self):
        self.name = 'Negative_Binomial_Regression'
        self.model_params = {}
        self.model_stats = {}

    def fit(self, df, metadata):
        """
        Fits regression model to data
        @param df: dataframe of incidents in regression format
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: _
        """
        # learn a negative binomial model for each cluster
        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]

            # first learn a poisson model with the data
            poisson_regressor = Poisson_Model()
            poisson_regressor.fit(df_cluster, metadata)

            # add the poisson rate for each example as part of the dataframe
            df_cluster['poisson_rate'] = poisson_regressor.predict(df_cluster, deepcopy(metadata))

            # add the dependent variable for the negative binomial regression
            # var = {(y_i - rate_i)^2 - y_i}/rate_i
            df_cluster['dep_ols'] = df_cluster.apply(lambda x: ((x['count'] - x['poisson_rate']) ** 2 - x['count']) /
                                                               x['poisson_rate'], axis=1)

            # get regression expression for alpha
            expr_alpha = self.get_regression_expr('alpha')
            model_alpha = smf.ols(expr_alpha, df_cluster).fit()

            # given alpha, fit the negative binomial model
            expr_count = self.get_regression_expr(param='count', features=metadata['features'])
            y_train, x_train = dmatrices(expr_count, df_cluster, return_type='dataframe')
            model_nb = sm.GLM(y_train, x_train, family=sm.families.NegativeBinomial(alpha=model_alpha.params[0])).fit()

            # save the cluster parameters
            self.model_params[temp_cluster] = {'alpha': model_alpha.params[0], 'model': model_nb, 'model_p': poisson_regressor}

        self.update_model_stats()
        print('Finished Learning {} model'.format(self.name))

    def predict(self, x_test, metadata):
        """
        Predicts E(y|x) for a set of x, where y is the concerned dependent variable.
        @param x_test: dataframe consisting of x points
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: dataframe with samples
        """
        df_samples = pd.DataFrame()
        # add intercept to features
        features = metadata['features']
        features.append('Intercept')

        clusters = x_test.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = x_test.loc[x_test.cluster_label == temp_cluster]
            df_cluster['sample'] = self.model_params[temp_cluster].predict(df_cluster[features])
            df_samples = df_samples.append(df_cluster[['cell', 'start_time', 'sample']])

    def get_regression_expr(self, param, features=None):
        """
        Creates regression expression in the form of a patsy expression
        @param features: x features to regress on
        @return: string expression
        """
        if param == 'alpha':
            return "dep_ols ~ poisson_rate - 1"
        elif param == 'count':
            expr = "count~"
            for i in range(len(features)):
                expr += features[i]
                if i != len(features) - 1:
                    expr += '+'
            return expr

    def update_model_stats(self):
        """
        Store the likelihood of the training set, AIC value.
        @return: _
        """
        train_likelihood = 0
        aic = 0
        for temp_cluster in self.model_params.keys():
            train_likelihood += self.model_params[temp_cluster]['model'].llf
            aic += self.model_params[temp_cluster]['model'].aic

        self.model_stats['train_likelihood'] = train_likelihood
        self.model_stats['aic'] = aic

    def get_likelihood(self, df, metadata):
        """
        Return the likelihood of model for the incidents in df
        @param df: dataframe of incidents to calculate likelihood on
        @param metadata: dictionary with feature names, cluster labels.
        @return: likelihood value
        """

        l = 0
        pred_features = deepcopy(metadata['features'])
        if 'Intercept' not in pred_features:
            pred_features.append('Intercept')
        if 'Intercept' not in df.columns:
            df['Intercept'] = 1

        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            # add mean rates to the dataframe
            df_cluster = get_log_lin_lambda(df_cluster, self.model_params[temp_cluster]['model'].params)
            count = df_cluster['count'].values.reshape(-1,1)
            rate = df_cluster['sample'].values.reshape(-1,1)
            alpha = self.model_params[temp_cluster]['alpha']
            l_temp = gamma(count + 1/alpha) / (gamma(1/alpha) * gamma(count+1))
            l_temp *= (1 / (1 + alpha * rate)) ** (1/alpha)
            l_temp *= ((alpha * rate) / (1 + alpha * rate)) ** count
            l_temp = np.log(l_temp)
            l += np.sum(l_temp)
            return l

