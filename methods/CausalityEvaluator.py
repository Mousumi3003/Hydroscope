import numpy as np
import pandas as pd
from hydroscope.utils.funcutils import shannon_entropy, mutual_information


class CausalityEvaluator:
    def __init__(self, source_vars, target_var, bins=10, seed=42):
        """
        Unified evaluator for computing Ap and Af.

        Parameters:
        - source_vars (list): List of input/source variable names.
        - target_var (str): Name of the output/target variable.
        - bins (int): Number of bins for discretization.
        - seed (int): Random seed for reproducibility.
        """
        self.source_vars = source_vars
        self.target_var = target_var
        self.bins = bins
        self.seed = seed
        self.ap_results = None
        self.af_results = None

    def compute_ap_df(self, df_obs, df_mod):
        """
        Compute Ap values between observed and modelled dataframes.

        Parameters:
        - df_obs (pd.DataFrame): Observed data
        - df_mod (pd.DataFrame): Modelled data

        Returns:
        - pd.DataFrame with Ap values per variable
        """
        common_vars = df_obs.columns.intersection(df_mod.columns)
        results = []

        for var in self.source_vars:
            x = df_obs[var].dropna()
            y = df_mod[var].dropna()
            x, y = x.align(y, join='inner')

            if len(x) < 2:
                continue

            H_x = shannon_entropy(x.values, bins=self.bins)
            MI = mutual_information(x.values, y.values, bins=self.bins, seed=self.seed)
            Ap = 1 - (MI / H_x) if H_x > 0 else np.nan

            results.append({'variable': var, 'H_x': H_x, 'MI': MI, 'Ap': Ap})

        self.ap_results = pd.DataFrame(results)
        return self.ap_results

    # def compute_ap(self, result_obs, result_mod):
    #     """
    #     Compute Ap values from mutual information result objects for observed and modeled data.
    
    #     Parameters:
    #     - result_obs: Result object from observed data (must be MI method)
    #     - result_mod: Result object from modeled data (must be MI method)
    
    #     Returns:
    #     - pd.DataFrame with columns: variable, MI_obs, MI_mod, H_x, Ap
    #     """
    
    #     if result_obs.metric != 'Mutual Information' or result_mod.metric != 'Mutual Information':
    #         raise ValueError("Both result objects must use 'Mutual Information' metric.")
    
    #     summary_obs = result_obs.summary
    #     summary_mod = result_mod.summary
    
    #     # Merge MI results by variable
    #     merged = pd.merge(
    #         summary_obs[['Variables', 'Mutual Information']],
    #         summary_mod[['Variables', 'Mutual Information']],
    #         on='Variables',
    #         suffixes=('_obs', '_mod')
    #     )
    
    #     # Compute H_x from the observed data in result_obs
    #     df_obs = result_obs.data  # assuming your result object stores the raw data here
    #     if df_obs is None:
    #         raise ValueError("result_obs must contain the original observed DataFrame as `data` attribute.")
    
    #     results = []
    #     for _, row in merged.iterrows():
    #         var = row['Variables']
    #         MI = row['Mutual Information_mod']
    #         H_x = shannon_entropy(df_obs[var].dropna().values, bins=self.bins)
    #         Ap = 1 - (MI / H_x) if H_x > 0 else np.nan
    
    #         results.append({
    #             'variable': var,
    #             'MI_obs': row['Mutual Information_obs'],
    #             'MI_mod': row['Mutual Information_mod'],
    #             'H_x': H_x,
    #             'Ap': Ap
    #         })
    
    #     self.ap_results = pd.DataFrame(results)
    #     return self.ap_results




    def compute_af(self, result_obs, result_mod):
        """
        Compute Af values from precomputed result summaries for various causal metrics.
    
        Parameters:
        - result_obs: Result object from observed data
        - result_mod: Result object from modeled data
    
        Returns:
        - pd.DataFrame with Af results for the appropriate method
        """
        
        # Ensure that both results refer to the same method/metric
        assert result_obs.method == result_mod.method, (
            f"Mismatched methods: {result_obs.method} vs {result_mod.method}"
        )
    
        method = result_obs.method
        metric= result_obs.metric
    
        # Mutual Information
        if metric == 'Mutual Information':
            summary_obs = result_obs.summary
            summary_mod = result_mod.summary
    
            merged = pd.merge(
                summary_obs[['Variables', 'Mutual Information']],
                summary_mod[['Variables', 'Mutual Information']],
                on='Variables',
                suffixes=('_obs', '_mod')
            )
    
            merged['Af'] = merged['Mutual Information_mod'] - merged['Mutual Information_obs']
            merged.rename(columns={
                'Variables': 'variable',
                'Mutual Information_obs': 'MI_obs',
                'Mutual Information_mod': 'MI_mod'
            }, inplace=True)
    
            self.af_results = merged[['variable', 'MI_obs', 'MI_mod', 'Af']]
            return self.af_results
    
        # Transfer Entropy
        elif metric == 'Transfer Entropy':
            summary_obs = result_obs.summary
            summary_mod = result_mod.summary
    
            merged = pd.merge(
                summary_obs[['Variables', 'Transfer Entropy']],
                summary_mod[['Variables', 'Transfer Entropy']],
                on='Variables',
                suffixes=('_obs', '_mod')
            )
    
            merged['Af'] = merged['Transfer Entropy_mod'] - merged['Transfer Entropy_obs']
            merged.rename(columns={
                'Variables': 'variable',
                'Transfer Entropy_obs': 'TE_obs',
                'Transfer Entropy_mod': 'TE_mod'
            }, inplace=True)
    
            self.af_results = merged[['variable', 'TE_obs', 'TE_mod', 'Af']]
            return self.af_results
    
        # CCM (Convergent Cross Mapping)
        elif method == 'CCM':
            summary_obs = result_obs.summary
            summary_mod = result_mod.summary
    
            merged = pd.merge(
                summary_obs[['Variables', 'R-squared']],
                summary_mod[['Variables', 'R-squared']],
                on='Variables',
                suffixes=('_obs', '_mod')
            )
    
            merged['Af'] = merged['R-squared_mod'] - merged['R-squared_obs']
            merged.rename(columns={
                'Variables': 'variable',
                'R-squared_obs': 'R2_obs',
                'R-squared_mod': 'R2_mod'
            }, inplace=True)
    
            self.af_results = merged[['variable', 'R2_obs', 'R2_mod', 'Af']]
            return self.af_results
    
        # SHAP Sensitivity
        elif method == 'SHAP':
            summary_obs = result_obs.summary
            summary_mod = result_mod.summary
    
            variables = summary_obs['Variables'].values
            shap_obs = result_obs.shap_values
            shap_mod = result_mod.shap_values
    
            if shap_obs.shape != shap_mod.shape:
                raise ValueError("Observed and modeled SHAP values must have the same shape")
    
            shap_diff = np.abs(shap_mod - shap_obs)
    
            af_mean = shap_diff.mean(axis=0)
            af_median = np.median(shap_diff, axis=0)
    
            af_df = pd.DataFrame({
                'variable': variables,
                'Af': af_mean,
                'Af_median': af_median
            })
    
            self.af_results = af_df
            return self.af_results
    
        else:
            raise NotImplementedError(f"Unsupported method type for Af computation: {method}")
            
 

    def compute_af_df(self, df_obs, df_mod):
        common_vars = df_obs.columns.intersection(df_mod.columns)
        results = []
    
        for var in self.source_vars:
            if var not in common_vars:
                continue
    
            # Observed
            x_obs = df_obs[var].dropna()
            y_obs = df_obs[self.target_var].dropna()
            x_obs, y_obs = x_obs.align(y_obs, join='inner')
    
            if len(x_obs) < 2:
                continue
    
            MI_obs = mutual_information(x_obs.values, y_obs.values, bins=self.bins, seed=self.seed)
    
            # Modeled
            x_mod = df_mod[var].dropna()
            y_mod = df_mod[self.target_var].dropna()
            x_mod, y_mod = x_mod.align(y_mod, join='inner')
    
            if len(x_mod) < 2:
                continue
    
            MI_mod = mutual_information(x_mod.values, y_mod.values, bins=self.bins, seed=self.seed)
    
            Af = MI_mod - MI_obs
            alpha = 1e-8
    
            results.append({
                'variable': var,
                'MI_mod': MI_mod,
                'MI_obs': MI_obs,
                'Af': Af,
                'alpha': alpha
            })
    
        af_result = pd.DataFrame(results)
        return af_result
