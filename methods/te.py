import numpy as np
import pandas as pd


class TransferEntropyResult:
    def __init__(self, summary_df, logs):
        self.summary = summary_df
        self.details = logs
        self.method='Infomation theory'
        self.metric='Transfer Entropy'

    def log(self):
        for line in self.details:
            print(line)


def shannon_entropy(data, bins=10):
    hist = np.histogramdd(np.array(data).T, bins=bins)[0]
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def conditional_mutual_information(df, source, target, condition, bins=10):
    df = df.dropna()
    x = df[source].values
    y = df[target].values
    z = df[condition].values

    H_xz = shannon_entropy([x, z], [bins, bins])
    H_yz = shannon_entropy([y, z], [bins, bins])
    H_z = shannon_entropy([z], bins)
    H_xyz = shannon_entropy([x, y, z], [bins] * 3)

    return H_xz + H_yz - H_z - H_xyz



def transfer_entropy(simulation, source_vars, target_var, best_lag, bins=10):
    logs = []
    results = []

    df = simulation.copy()

    # Lag the source variables
    for var in source_vars:
        df[f'lagged_{var}'] = df[var].shift(best_lag)
        logs.append(f"Lagged {var} by {best_lag} steps.")

    # For each source, compute TE
    for var in source_vars:
        lagged = f'lagged_{var}'

        temp_df = df[[lagged, target_var]].copy()
        temp_df['condition'] = df[target_var].shift(1)
        temp_df = temp_df.dropna()

        if temp_df.empty:
            logs.append(f"⚠️ Skipped {var}: no data after lagging and conditioning.")
            continue

        try:
            te = conditional_mutual_information(temp_df, lagged, target_var, 'condition', bins)
            logs.append(f"TE({var} → {target_var}) = {te:.4f}")
            results.append([lagged, te])
        except Exception as e:
            logs.append(f"⚠️ Failed TE calculation for {var}: {e}")

    if results:
        summary_df = pd.DataFrame(results, columns=['Variables', 'Transfer Entropy'])
        summary_df['Sensitivity rank'] = summary_df['Transfer Entropy'].rank(ascending=False, method='first')
    else:
        summary_df = pd.DataFrame(columns=['Variables', 'Transfer Entropy', 'Sensitivity rank'])
        logs.append("❌ No valid TE results.")

    return TransferEntropyResult(summary_df, logs)

