import numpy as np

class PredictivePerformance:
    """
    Computes Predictive Asymmetry (Ap) values using various causality methods:
    CCM, Granger, Mutual Information, SHAP, and Transfer Entropy.
    """

    def __init__(self, results, req_df=None, target=None, bins=10):
        self.method = results.method
        self.metric = results.metric
        self.results = results.summary if hasattr(results, 'summary') else results
        self.req_df = req_df
        self.target = target
        self.bins = bins

    def shannon_entropy(self, data_list, bins):
        hist = np.histogramdd(np.vstack(data_list).T, bins=bins)[0]
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))

    def compute_ap(self):

                raise ValueError("Missing data for Mutual Information.")
            H_y = self.shannon_entropy([self.req_df[self.target].values], self.bins)
            H_y = max(H_y, 1e-6)
            Ap = 1 - mi / H_y
            return max(0, min(1, Ap))

 
        return results
