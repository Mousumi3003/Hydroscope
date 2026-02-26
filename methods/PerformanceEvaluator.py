import numpy as np

class PerformanceEvaluator:
    """
    Evaluates statistical goodness-of-fit metrics between observed and simulated data.
    """

    @staticmethod
    def nse(observed, simulated):
        """Nash-Sutcliffe Efficiency (NSE)"""
        a = np.array(observed)
        b = np.array(simulated)
        u = np.sum((a - b) ** 2)
        v = np.sum((a - np.mean(a)) ** 2)
        return 1 - u / v if v != 0 else (1 if u == 0 else -np.inf)

    @staticmethod
    def pbias(observed, simulated):
        """Percent Bias (PBIAS)"""
        observed = np.array(observed)
        simulated = np.array(simulated)
        return 100 * np.sum(simulated - observed) / np.sum(observed)

    @staticmethod
    def rmse(observed, simulated):
        """Root Mean Square Error (RMSE)"""
        observed = np.array(observed)
        simulated = np.array(simulated)
        return np.sqrt(np.mean((simulated - observed) ** 2))

    @staticmethod
    def kge(observed, simulated):
        """Kling-Gupta Efficiency (KGE)"""
        obs = np.array(observed)
        sim = np.array(simulated)
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    @staticmethod
    def mae(observed, simulated):
        """Mean Absolute Error (MAE)"""
        return np.mean(np.abs(np.array(observed) - np.array(simulated)))

    @staticmethod
    def r_squared(observed, simulated):
        """Coefficient of Determination (R²)"""
        observed = np.array(observed)
        simulated = np.array(simulated)
      
        return np.corrcoef(observed,simulated)[0,1]
