import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

class GrangerCausalityResult:
    def __init__(self, summary_df, logs):
        self.summary = summary_df
        self.details = logs
        self.method='Granger Causality'
        self.metric='F-statistic'

    def log(self):
        for line in self.details:
            print(line)


def check_stationarity(series, significance_level=0.05):
    kpss_p = kpss(series.dropna(), regression='ct')[1]
    adf_p = adfuller(series.dropna())[1]
    return kpss_p > significance_level and adf_p < significance_level


def make_stationary(series, max_diff=5):
    diffed = series.copy()
    num_diffs = 0

    while not check_stationarity(diffed) and num_diffs < max_diff:
        diffed = diffed.diff().dropna()
        num_diffs += 1

    return diffed, num_diffs


def get_optimal_lag(data, max_lag=30):
    model = VAR(data)
    order = model.select_order(maxlags=max_lag).selected_orders['aic']
    return order if order > 0 else 1


def granger_causality(simulation, source, target, mode='individual', max_diff=5, max_lag=30):
    logs = []
    results = []

    for col in source:
        logs.append(f"\nAnalyzing {col} → {target}")
        X, x_diff = make_stationary(simulation[col], max_diff)
        Y, y_diff = make_stationary(simulation[target], max_diff)

        # Shift and align after differencing
        X = X.shift(y_diff)
        Y = Y.shift(x_diff)
        data = pd.concat([X, Y], axis=1).dropna()
        data.columns = ['Source', 'Target']

        if data.shape[0] < max_lag + 2:
            logs.append(f"⚠️ Skipped {col}: Not enough data after differencing.")
            continue

        try:
            lag = get_optimal_lag(data, max_lag)
            logs.append(f"Optimal lag: {lag}")
        except Exception as e:
            logs.append(f"⚠️ Failed lag selection for {col}: {str(e)}")
            continue

        try:
            gc_res = grangercausalitytests(data, maxlag=lag, verbose=False)
            f_stat = gc_res[lag][0]['ssr_chi2test'][0]
            p_val = gc_res[lag][0]['ssr_chi2test'][1]
            results.append([f'lagged_{col}', f_stat, p_val])
            logs.append(f"Granger test: F={f_stat:.3f}, p={p_val:.3e}")
        except Exception as e:
            logs.append(f"⚠️ Failed Granger test for {col}: {str(e)}")
 

    if results:
        summary_df = pd.DataFrame(results, columns=['Variables', 'F-statistic', 'P-value'])
        summary_df['Sensitivity rank'] = summary_df['F-statistic'].rank(ascending=False, method='min')
    else:
        summary_df = pd.DataFrame(columns=['Variables', 'F-statistic', 'P-value', 'Sensitivity rank'])
        logs.append("❌ No valid results.")


    return GrangerCausalityResult(summary_df, logs)
