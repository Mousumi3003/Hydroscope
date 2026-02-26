import numpy as np
import statsmodels.api as sm
import pandas as pd

class LagSelector:
    """
    A class to select the best lag for a predictor variable (e.g., precipitation) with respect to 
    a response variable (e.g., runoff) using correlation, AIC, and BIC from time series data.

    Attributes:
    -----------
    simulation : pd.DataFrame
        A DataFrame containing at least the columns specified in precip_col and runoff_col.
    precip_col : str
        Name of the predictor (e.g., precipitation) column.
    runoff_col : str
        Name of the response (e.g., runoff) column.
    max_lag : int
        Maximum lag to consider.
    lag_range : range
        Range of lags from 0 to max_lag.
    correlations : list
        Correlation values at each lag.
    aic_values : list
        AIC values at each lag.
    bic_values : list
        BIC values at each lag.
    """

    def __init__(self, simulation_df, precip_col='P', runoff_col='Q', max_lag=30):
        self.simulation = simulation_df.copy()
        self.precip_col = precip_col
        self.runoff_col = runoff_col
        self.max_lag = max_lag
        self.lag_range = range(0, max_lag + 1)
        self.correlations = []
        self.aic_values = []
        self.bic_values = []

    def compute_correlation_lag(self):
        """
        Computes the lag with the highest absolute correlation between lagged predictor and response.

        Returns:
        --------
        int
            Lag index with the maximum absolute correlation.
        """
        for lag in self.lag_range:
            lagged_name = 'lagged_' + self.precip_col
            self.simulation[lagged_name] = self.simulation[self.precip_col].shift(lag)
            correlation = self.simulation.corr().loc[self.runoff_col, lagged_name]
            self.correlations.append(correlation)
        best_lag_corr = np.argmax(np.abs(self.correlations))
        return best_lag_corr

    def compute_aic_bic_lags(self):
        """
        Computes AIC and BIC values for linear models at each lag.

        Returns:
        --------
        tuple
            Lags corresponding to lowest AIC and BIC.
        """
        for lag in self.lag_range:
            lagged_name = 'lagged_' + self.precip_col
            self.simulation[lagged_name] = self.simulation[self.precip_col].shift(lag)
            simulation_lagged = self.simulation.dropna(subset=[lagged_name])

            X = sm.add_constant(simulation_lagged[lagged_name])
            y = simulation_lagged[self.runoff_col]

            model_ = sm.OLS(y, X).fit()
            self.aic_values.append(model_.aic)
            self.bic_values.append(model_.bic)

        best_lag_aic = self.lag_range[np.argmin(self.aic_values)]
        best_lag_bic = self.lag_range[np.argmin(self.bic_values)]
        return best_lag_aic, best_lag_bic

    def get_best_lag(self):
        """
        Determines the best lag by selecting the minimum among best correlation, AIC, and BIC lags.

        Returns:
        --------
        int
            Best overall lag.
        """
        best_corr_lag = self.compute_correlation_lag()
        best_aic_lag, best_bic_lag = self.compute_aic_bic_lags()
        best_lag = min(best_corr_lag, best_aic_lag, best_bic_lag)

        # Store final lagged predictor
        self.simulation['lagged_' + self.precip_col] = self.simulation[self.precip_col].shift(best_lag)
        return best_lag
