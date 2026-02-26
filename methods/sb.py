import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import shap

class SHAPSensitivityResult:
    def __init__(self, summary_df, logs, shap_values=None, shap_data=None, model_perf=None):
        self.summary = summary_df
        self.details = logs
        self.method = 'SHAP'
        self.metric = 'Slope'
        self.shap_values = shap_values
        self.shap_data = shap_data
        self.model_perf = model_perf or {}

    def log(self):
        for line in self.details:
            print(line)



def fitML(X, Y, method='RF', params=None):
    if params is None:
        params = {}

    if method == 'RF':
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.25, random_state=42)

        param_grid = {
            'n_estimators': [25, 50, 100, 150],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [3, 6, 9],
            'max_leaf_nodes': [3, 6, 9]
        }

        random_search = RandomizedSearchCV(RandomForestRegressor(), param_grid, n_iter=10)
        random_search.fit(X_train, y_train)

        rf = random_search.best_estimator_
        rf.fit(X, Y)
        return rf
    else:
        raise ValueError(f"Unsupported method: {method}")


from sklearn.metrics import r2_score

def SHAPsensitivity(simulation, source, target, best_lag=1, n=365*5, random_state=1, method='RF'):
    logs = []

    df = simulation.copy()

    # Create lagged variables
    lagged_source = []
    for var in source:
        lagged_var = f'lagged_{var}'
        df[lagged_var] = df[var].shift(best_lag)
        lagged_source.append(lagged_var)
        logs.append(f"Lagged {var} by {best_lag} as {lagged_var}.")

    df = df[lagged_source + [target]].dropna()
    logs.append(f"Prepared DataFrame shape after lagging and NA removal: {df.shape}")

    try:
        X_sampled = df[lagged_source].sample(n=n, random_state=random_state)
        Y_sampled = df[target].loc[X_sampled.index]
        logs.append(f"Sampled {n} rows for SHAP analysis.")
    except Exception as e:
        logs.append(f"❌ Sampling failed: {e}")
        return SHAPSensitivityResult(pd.DataFrame(), logs)

    # Train model + track performance
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_sampled, Y_sampled, test_size=0.25, random_state=42)

        model = fitML(X_sampled, Y_sampled, method=method)
        logs.append(f"Trained {method} model.")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        model_perf = {'train_r2': train_r2, 'test_r2': test_r2}
        logs.append(f"Model performance - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    except Exception as e:
        logs.append(f"❌ Model training failed: {e}")
        return SHAPSensitivityResult(pd.DataFrame(), logs)

    # SHAP values
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sampled)
        logs.append("Computed SHAP values.")
    except Exception as e:
        logs.append(f"❌ SHAP computation failed: {e}")
        return SHAPSensitivityResult(pd.DataFrame(), logs)

    # Sensitivity from SHAP slopes
    results = []
    for i, col in enumerate(X_sampled.columns):
        try:
            s = shap_values[:, i]
            v = np.array(X_sampled[col]).reshape(-1, 1)
            mask = ~np.isnan(s) & ~np.isnan(v).flatten() & ~np.isinf(v).flatten()
            v_clean, s_clean = v[mask], s[mask]

            if len(s_clean) >= 5 and not np.all(v_clean == 0):
                tel = TheilSenRegressor().fit(v_clean, s_clean)
                slope = tel.coef_[0]
                results.append([col, slope])
                logs.append(f"Computed slope for {col}: {slope:.4f}")
            else:
                logs.append(f"⚠️ Skipped {col} due to invalid data.")
        except Exception as e:
            logs.append(f"⚠️ Failed to compute slope for {col}: {e}")

    if results:
        summary_df = pd.DataFrame(results, columns=['Variables', 'Slope'])
        summary_df['Sensitivity rank'] = summary_df['Slope'].rank(ascending=False, method='first')
        logs.append("Assigned unique sensitivity ranks using method='first'.")
    else:
        summary_df = pd.DataFrame(columns=['Variables', 'Slope', 'Sensitivity rank'])
        logs.append("❌ No valid SHAP slopes were calculated.")

    return SHAPSensitivityResult(summary_df, logs, shap_values, X_sampled, model_perf)
