import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .condition_fun import x_variable


def vif(dt, x=None):
    dt = dt.copy()
    if isinstance(x, str):
        x = [x]
    if x is not None:
        dt = dt[x]

    x = x_variable(dt, x=x)
    X_train = dt.loc[:, x].copy()
    X_train['const'] = 1
    vif_json = {'variable': [], 'vif': []}
    for idx, v in enumerate(x):
        vif_json['variable'].append(v)
        vif_json['vif'].append(variance_inflation_factor(X_train, idx))

    return pd.DataFrame(vif_json)
