"""
@Author - Ayan Mukhopadhyay
Simple OLS Regression -- Inherits from Forecaster class
"""

from forecasters.base import Forecaster
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from copy import deepcopy
from scipy.special import factorial


def get_poisson_lambda(df, params):
    def rate_calc(x):
        temp = 0
        for key, val in params.items():
            try:
                temp += params[x['cluster_label']][key] * x[key]
            except KeyError:
                # intercept is not in the features but always 1
                temp += params[key] * 1
        # to ensure numerical stability, put a very small rate for units that have 0 rate
        if temp == 0:
            temp = 1e-10

        return np.exp(temp)

    df['poisson_rate'] = df.apply(rate_calc, axis=1)
    return df


def create_default_meta(df, static_features=None):
    metadata = {'start_time_train': df['time'].min(), 'end_time_train': df['time'].max()}
    if static_features is None:
        static_features = list(df.columns)
        if 'cluster_label' in static_features:
            static_features.remove('cluster_label')
    metadata['features'] = static_features
    return metadata


class Simple_Model(Forecaster):

    def __init__(self):
        self.name = 'Simple_Regression'
        self.model_params = {}
        self.model_stats = {}

    def fit(self, df, metadata=None):
        """
        Fits regression model to data
        @param df: dataframe of incidents in regression format
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: _
        """
        
        # if metadata is none, use standard parameters
        if metadata is None:
            metadata = create_default_meta(df)

        # get regression expression
        expr = self.get_regression_expr(metadata['features'])
        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            y_train, x_train = dmatrices(expr, df_cluster, return_type='dataframe')

            # fit model
            model = sm.GLM(y_train, x_train,family=sm.families.Gaussian()).fit()
            self.model_params[temp_cluster] = model

        self.update_model_stats()
        print('Finished Learning {} model'.format(self.name))

        
    def predict(self, df, metadata):
        """
        Predicts E(y|x) for a set of x, where y is the concerned dependent variable.
        @param df: dataframe consisting of x points
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
            df_cluster['sample'] = self.model_params[temp_cluster].predict(df_cluster[features])
            try:
                df_samples = df_samples.append(df_cluster[[metadata['unit_name'], 'start_time', 'sample']])
            except KeyError:
                df_samples = df_cluster['sample']
            #TODO: UPDATE LIKELIHOOD METHOD
            # lik += self.get_likelihood(df_cluster['sample'].values, df_cluster['count'].values)

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
        count = df['count'].to_numpy().reshape(-1, 1)
        pred_features = deepcopy(metadata['features'])
        if 'Intercept' not in pred_features:
            pred_features.append('Intercept')
        if 'Intercept' not in df.columns:
            df['Intercept'] = 1
        rate = df.apply(lambda x: self.model_params[x['cluster_label']].
                        predict([x[pred_features]])[0], axis=1).to_numpy()
        rate = rate.reshape(-1, 1)
        l = -1 * np.sum(np.log(factorial(count))) + np.sum(np.log(rate)*count) - np.sum(rate)
        return l

    def update_model_stats(self):
        """
        Store the likelihood of the training set, AIC value.
        @return: _
        """
        train_likelihood = 0
        aic = 0
        for temp_cluster in self.model_params.keys():
            train_likelihood += self.model_params[temp_cluster].llf
            aic += self.model_params[temp_cluster].aic

        self.model_stats['train_likelihood'] = train_likelihood
        self.model_stats['aic'] = aic
