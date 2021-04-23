import pandas as pd
import numpy as np
import plotnine as p9
import itertools as it
from sklearn import preprocessing, compose, pipeline, linear_model, metrics
import pickle

def read_weekly_data(file_loc = '../data/week_sum_demo.csv'):
    """
    Read the weekly testing data.
    """
    week_data = pd.read_csv(file_loc, dtype={'FIPS':str})
    total_prop = week_data.groupby('week').sum().eval('pos_results/days_by_10kpop')

    def week_num(df):
        df['week_num'] = np.arange(df.shape[0])
        df['ca_prop'] = total_prop.values
        return df

    week_data = week_data.groupby('FIPS').apply(week_num)
    week_data['pos_prop'] = week_data.eval('pos_results / days_by_10kpop')
    return week_data

def generate_model(pred_vars, log_transform = True, one_hot_week = False, method = "lm"):
    """
    Generate the model for transforming and predicting.
    ...
    """
    assert method in ['lm','poisson'], "method must be one of 'lm' or 'poisson'"
    if log_transform:
        ft = preprocessing.FunctionTransformer(np.log)
    else:
        ft = preprocessing.FunctionTransformer()
    
    if one_hot_week:
        model_prep = compose.ColumnTransformer(
            [
                ("onehot_categorical", preprocessing.OneHotEncoder(),
                 ["week_num"]),
                ("num_scaler", ft, pred_vars)
            ],
            remainder="drop",
        )
    else:
        model_prep = compose.ColumnTransformer(
            [   
                ("num_scaler", ft, pred_vars + ['ca_prop'])
            ],
            remainder="drop",
        )
    if method == 'lm':
        pipe = pipeline.Pipeline([
            ("preprocessor", model_prep),
            ("regressor", linear_model.LinearRegression())
        ])
    elif method == 'poisson':
        pipe = pipeline.Pipeline([
            ("preprocessor", model_prep),
            ("regressor", linear_model.PoissonRegressor(alpha=1e-12, max_iter=10000))
        ])
    return pipe

def transform_y(y, tau = 1e-1):
    return np.log(tau + y)

def inv_transform_y(z, tau = 1e-1):
    return np.exp(z) - tau

def fit_fips_lm(fips, pipe, week_data, tau = 1e-1):
    train_data = week_data.query('FIPS != "{}"'.format(fips))
    z_train = transform_y(train_data['pos_prop'], tau=tau)
    pipe.fit(train_data, z_train)
    z_pred = pipe.predict(week_data.query('FIPS == "{}"'.format(fips)))
    return inv_transform_y(z_pred, tau=tau)

def model_error_lm(week_data, pipe, tau = 1e-1, ret_data = False):
    week_data['pos_prop_pred'] = 0
    for fips in week_data['FIPS'].unique():
        week_data.loc[week_data['FIPS'] == fips,'pos_prop_pred'] = fit_fips_lm(fips, pipe, week_data, tau = tau)
    if ret_data:
        return {
            'mae': metrics.mean_absolute_error(week_data["pos_prop"],
                                               week_data['pos_prop_pred'],
                                               sample_weight=week_data["days_by_10kpop"]),
            'data': week_data
        }
    else:
        return {'mae': metrics.mean_absolute_error(week_data["pos_prop"],
                                                   week_data['pos_prop_pred'],
                                                   sample_weight=week_data["days_by_10kpop"])}
if __name__=="__main__":
    week_data = read_weekly_data()
    pred_vars = list(week_data.columns[4:11])
    # pred_vars = ['seniors','hispanic_pop','black_pop']
    tau_set = 2.0**(-np.arange(8))
    model_parms = it.product([True,False],[True,False],tau_set)
    model_results = []
    for log_t, oh_t, tau in model_parms:
        pipe = generate_model(pred_vars, log_transform=log_t, one_hot_week=oh_t)
        pred_ret = model_error_lm(week_data.copy(), pipe, tau=tau)
        
        model_results.append({
            'log_transform': log_t,
            'one_hot_week': oh_t,
            'tau': tau,
            'mae': pred_ret['mae']
        })
    model_ordered = sorted(model_results, key=lambda x: x['mae'])
    model_selected = model_ordered[0]
    with open("../data/selected_parms.pickle","wb") as parmfile:
        pickle.dump(model_selected, parmfile)
        
