"""
@Author - Ayan Mukhopadhyay
Parent File for all regression models for incident prediction.
Currently supports -
1. Poisson Regression (Count Based)
2. Negative Binomial Regression (Count Based)
3. Parametric Survival Regression (Time Based)
"""
from forecasters.simple_reg_forecaster import Simple_Model
from forecasters.poisson_reg_forecaster import Poisson_Model
from forecasters.negative_binomial_forecaster import Negative_Binomial_Model
from forecasters.zip_forecaster import ZIP_Model
from forecasters.survival_forecaster import Survival_Model
from forecasters.utils import create_regression_df, create_regression_df_test, update_meta_categorical
import numpy as np


def learn(df, metadata, model_name):
    """
    Wrapper before data is passed to specific regression models. Converts raw incident data to a format that regression
    models can use. For example, creates counts per time period for Poisson model. Splits the data into train and test
    sets for cross-validation.
    @param df: raw dataframe of incidents
    @param metadata: metadata with start and end dates, spatial unit etc. See github documentation for details
    @param model_name: the name of the regression model
    @return: trained model and regression df (the latter if the flag @param return_reg_df is set to True)
    """
    # create dataframe for regression

    regression_data = create_regression_df(df, metadata, model_name)
    df_learn = regression_data['train']
    df_predict = regression_data['predict']
    metadata['features'] = update_meta_categorical(metadata['features'], df_features=list(df_learn.columns),
                                                   cat_col=metadata['cat_features'])
    # split into train and test
    split_point = 0.8
    mask = np.random.rand(len(df_learn)) < split_point
    df_train = df_learn[mask]
    df_test = df_learn[~mask]
    if model_name == 'Simple_Regression':
        model = Simple_Model()
        model.fit(df_train, metadata)
    elif model_name == 'Poisson_Regression':
        model = Poisson_Model()
        model.fit(df_train, metadata)
    elif model_name == 'Negative_Binomial_Regression':
        model = Negative_Binomial_Model()
        model.fit(df_train, metadata)
    elif model_name == 'Zero_Inflated_Poisson_Regression':
        model = ZIP_Model()
        model.fit(df_train, metadata)
    elif model_name == 'Survival_Regression':
        model = Survival_Model()
        model.fit(df_train, metadata)

    return {'model':model, 'df_train':df_train, 'df_predict':df_predict}


def predict(model, df, metadata):
    """
    Wrapper method before data is passed to specific predict methods for regression models
    @param model: the trained model
    @param df: dataframe of points where predictions need to be made
    @param metadata: dictionary with start and end dates for predictions, cluster labels etc
    @return: dataframe with E(y|x) appended to each row
    """
    if model.name == 'Poisson_Regression' or model.name == 'Negative_Binomial_Regression' or model.name == 'Simple_Regression':
        df_ = create_regression_df_test(df, metadata)
        df_samples = model.predict(df_, metadata)
        return df_samples
