"""
@Author - Ayan Mukhopadhyay
Zero-Inflated Poisson Regression Forecaster -- Inherits from Forecaster class
"""

from forecasters.base import Forecaster
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from copy import deepcopy
import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-1 * x))


def parse_inflate_model(coef):
    coef_reg = {}
    coef_inflate = {}
    for k,v in coef.items():
        if 'inflate_' in k:
            coef_inflate[k] = v
        else:
            coef_reg[k] = v
    return coef_reg, coef_inflate


def get_multi_level_lik(df, params_reg, params_inflate, inflate_func=None):
    """
    Calculate the likelihood of a zero-inflated poisson model. The likelihood of the inflated model is calculated
    through coefficients marked with 'inflate_'
    @param df: dataframe whose likelihood needs to be calculated
    @param param: set of regression coefficients
    @return: likelihood value
    """

    def lik_calc(x):
        temp_inflate = 0
        temp_poisson = 0

        for key, val in params_inflate.items():
            feature = key.split('inflate_')[1]  # retrieve the part after "inflate_"
            temp_inflate += params_inflate[key] * x[feature]

        for key, val in params_reg.items():
            try:
                temp_poisson += params_reg[key] * x[key]
            except KeyError:
                # if intercept is not in the features
                temp_poisson += params_reg[key] * 1

        p_lambda = np.exp(temp_poisson)
        p_logistic = sigmoid(temp_inflate)

        if x['count'] == 0:  # use embedded inflated model (logistic regression usually)
            return np.log(p_logistic + (1-p_logistic) * np.exp(-1 * p_lambda))

        else:  # use embedded inflated model for P(y|x) != 0 * P(count=y|x)

            l_poisson = np.log(((p_lambda) ** x['count']) * np.exp(-1 * p_lambda) / np.math.factorial(x['count']))
            l_inflate = np.log(1 - p_logistic)
            return l_poisson + l_inflate

    l = df.apply(lambda x: lik_calc(x), axis=1)
    return sum(l)


class ZIP_Model(Forecaster):

    def __init__(self):
        self.name = 'Zero_Inflated_Poisson_Regression'
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
        # get regression expression
        expr = self.get_regression_expr(metadata['features'])
        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            y_train, x_train = dmatrices(expr, df_cluster, return_type='dataframe')

            # fit model
            # use a logistic regression model to estimate probability of zero.
            # use the same features as in the poisson
            model = sm.ZeroInflatedPoisson(endog=y_train, exog=x_train, exog_infl=x_train, inflation='logit').fit()
            self.model_params[temp_cluster] = {'coef': model.params, 'model': model}

        self.update_model_stats()
        print('Finished Learning {} model'.format(self.name))

    def predict(self, df, metadata):
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
        if 'Intercept' not in features:
            features.append('Intercept')
        # if intercept not added (e.g. if poisson lambda is needed for negative binomial learning,
        # then add the intercept term explicitly
        if 'Intercept' not in df.columns:
            df['Intercept'] = 1

        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            df_cluster['sample'] = self.model_params[temp_cluster].predict(df_cluster[features],
                                                                           exog_infl=df_cluster[features])
            df_samples = df_samples.append(df_cluster[[metadata['unit_name'], 'start_time', 'sample']])

        return df_samples

    def get_regression_expr(self, features):
        """
        Creates regression expression in the form of a patsy expression
        @param features: x features to regress on
        @return: string expression
        """
        expr = "count~"
        for i in range(len(features)):
            # patsy expects 'weird columns' to be inside Q
            if ' ' in features[i]:
                expr += "Q('" + features[i] + "')"
            else:
                expr += features[i]
            if i != len(features) - 1:
                expr += '+'
        return expr

    def get_likelihood(self, df, metadata):
        """
        Return the likelihood of model for the incidents in df
        @param df: dataframe of incidents to calculate likelihood on
        @param metadata: dictionary with feature names, cluster labels.
        @return: likelihood value
        """

        pred_features = deepcopy(metadata['features'])
        if 'Intercept' not in pred_features:
            pred_features.append('Intercept')
        if 'Intercept' not in df.columns:
            df['Intercept'] = 1

        clusters = df.cluster_label.unique()
        l = 0
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            coef_reg, coef_inflate = parse_inflate_model(self.model_params[temp_cluster]['coef'])
            l += get_multi_level_lik(df_cluster, params_reg=coef_reg, params_inflate=coef_inflate)
        return l

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
