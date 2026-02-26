import numpy as np
import pandas as pd
import random

class MutualInformationResult:
    def __init__(self, summary_df, logs,data):
        self.summary = summary_df
        self.details = logs
        self.method='Infomation theory'
        self.metric='Mutual Information'
        self.data=data

    def log(self):
        for line in self.details:
            print(line)


def shannon_entropy(x, bins=10):
    c = np.histogramdd(x, bins)[0]
    p = c / np.sum(c)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mutual_information(dfi, bins=10, reshuffle=0):
    x = dfi.iloc[:, 0].values
    y = dfi.iloc[:, -1].values
    if reshuffle:
        random.shuffle(x)
        random.shuffle(y)
    H_x = shannon_entropy([x], [bins])
    H_y = shannon_entropy([y], [bins])
    H_xy = shannon_entropy([x, y], [bins, bins])
    return H_x + H_y - H_xy


def mutualinformation(simulation, source, target, best_lag, bins=10):
    logs = []
    df = simulation.copy()

    # Create lagged variables
    for var in source:
        lagged_var = f'lagged_{var}'
        df[lagged_var] = df[var].shift(best_lag)
        logs.append(f"Lagged variable {var} by {best_lag} as {lagged_var}.")

    columns = [f'lagged_{var}' for var in source]
    req_df = df[columns + [target]].dropna()
    logs.append(f"Cleaned data shape after dropping NA: {req_df.shape}")

    results = []
    for col in columns:
        mi = mutual_information(req_df[[col, target]], bins=bins)
        results.append([col, mi])
        logs.append(f"Mutual Information between {col} and {target}: {mi:.4f}")
   
   
    if results:
        summary_df = pd.DataFrame(results, columns=['Variables', 'Mutual Information'])
        summary_df['Sensitivity rank'] = summary_df['Mutual Information'].rank(ascending=False, method='first')
        logs.append("Assigned unique sensitivity ranks using method='first'.")
    else:
        summary_df = pd.DataFrame(columns=['Variables', 'Mutual Information', 'Sensitivity rank'])
        logs.append("❌ No mutual information could be calculated.")

    return MutualInformationResult(summary_df, logs,df)
