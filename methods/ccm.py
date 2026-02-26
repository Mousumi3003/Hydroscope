import pandas as pd
from causal_ccm.causal_ccm import ccm

class CCMCausalityResult:
    def __init__(self, summary_df, logs):
        self.summary = summary_df
        self.details = logs
        self.method='CCM'
        self.metric='R-squared'

    def log(self):
        for line in self.details:
            print(line)


def ccmCausality(simulation, source, target, best_lag, bins=10, mode='individual'):
    logs = []
    df = simulation.copy()

    # Create lagged versions of source variables
    for var in source:
        lagged_var = f'lagged_{var}'
        df[lagged_var] = df[var].shift(best_lag)
        logs.append(f"Lagged {var} by {best_lag} as {lagged_var}.")

    columns = [f'lagged_{var}' for var in source]
    req_df = df[columns + [target]].dropna()
    logs.append(f"Cleaned data shape after dropping NA: {req_df.shape}")

    if mode == 'individual':
        results = []
        for i, col in enumerate(columns):
            try:
                causality = ccm(req_df.iloc[:, i], req_df.iloc[:, -1]).causality()
                r_squared, p_value = causality
                logs.append(f"CCM causality for {col} → {target}: R² = {r_squared:.4f}, p = {p_value:.4g}")
                results.append([col, r_squared, p_value])
            except Exception as e:
                logs.append(f"⚠️ CCM causality failed for {col}: {e}")
    

        if results:
            summary_df = pd.DataFrame(results, columns=['Variables', 'R-squared', 'P-value' ])
            summary_df['Sensitivity rank'] = summary_df['R-squared'].rank(ascending=False, method='first')
            logs.append("Assigned unique sensitivity ranks using method='first'.")
        else:
            summary_df = pd.DataFrame(columns=['Variables', 'R-squared', 'P-value', 'Sensitivity rank'])
            logs.append("❌ No valid CCM results.")

    else:
        summary_df = pd.DataFrame(columns=['Variables', 'R-squared', 'P-value', 'Sensitivity rank'])
        logs.append("⚠️ Only 'individual' mode is supported currently.")

    return CCMCausalityResult(summary_df, logs)
