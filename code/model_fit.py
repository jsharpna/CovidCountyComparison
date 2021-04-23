from model_sel import *

def fit_all_lm(pipe, week_data, tau=1e-1):
    z_train = transform_y(week_data['pos_prop'], tau=tau)
    pipe.fit(week_data, z_train)
    z_pred = pipe.predict(week_data)
    return inv_transform_y(z_pred, tau=tau)

def get_betas(pipe, week_data, tau=1e-1):
    z_train = transform_y(week_data['pos_prop'], tau=tau)
    pipe.fit(week_data, z_train)
    return pipe.get_params()['regressor'].coef_

def all_error_lm(week_data, pipe, tau = 1e-1):
    week_data['pos_prop_pred'] = fit_all_lm(pipe, week_data, tau = tau)
    return {
        'mae': metrics.mean_absolute_error(week_data["pos_prop"],
                                           week_data['pos_prop_pred'],
                                           sample_weight=week_data["days_by_10kpop"]),
        'data': week_data
    }

def pred_create(week_data, pipe, tau=1e-1, leave_out=True):
    if leave_out:
        new_data = model_error_lm(week_data.copy(), pipe, tau=tau, ret_data=True)['data']
    else:
        new_data = all_error_lm(week_data.copy(), pipe, tau=tau)['data']
    new_data['pos_pred'] = new_data.eval('pos_prop_pred * days_by_10kpop')
    return new_data

def pred_iter(week_data, pipe, tau=1e-1, leave_out=True, special_fips = None):
    for fips in week_data['FIPS'].unique():
        new_var = pred_create(week_data.query(f"FIPS != '{fips}'").copy(), pipe, tau = tau, leave_out = leave_out)
        sum_var = new_var.groupby('FIPS').sum()
        sum_var['hold_out_fips'] = fips
        if special_fips:
            fips_data = new_var.query(f'FIPS == "{special_fips}"')
            yield (sum_var, fips_data)
        else:
            yield sum_var

def coef_jackknife(week_data, pipe, tau=1e-1):
    Betas = []
    for fips in week_data['FIPS'].unique():
        betas = get_betas(pipe, week_data.query(f"FIPS != '{fips}'").copy(), tau = tau)
        Betas.append(betas)
    return np.array(Betas)

if __name__=="__main__":
    week_data = read_weekly_data()
    pred_vars = list(week_data.columns[4:11])
    with open("../data/selected_parms.pickle", 'rb') as parmfile:
        model_selected = pickle.load(parmfile)

    print(model_selected)
    pipe = generate_model(pred_vars,
                          log_transform=model_selected['log_transform'],
                          one_hot_week=model_selected['one_hot_week'])
    pred_data, fips_data = pd.DataFrame(), pd.DataFrame()
    spec_fips = '06113'
    for df, fips_df in pred_iter(week_data, pipe, tau=model_selected['tau'], leave_out=False, special_fips = spec_fips):
        pred_data = pd.concat((pred_data, df))
        fips_data = pd.concat((fips_data, fips_df))
    pred_data.to_csv('../data/pred_data.csv')
    fips_data.to_csv('../data/yolo_data.csv')
    
    pd.DataFrame(coef_jackknife(week_data, pipe, tau=model_selected['tau'])).to_csv('../data/beta_jackknife.csv')
