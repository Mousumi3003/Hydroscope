import numpy as np
import pandas as pd

def model(
    data,
    alpha=0.5,
    beta=0.8,
    gamma=0.9,
    K=0.5,
    c=0.1,
    threshold1=0.1,
    threshold2=0.1,
    Srzmax=500,
    GWmax=3000,
    theta=20,
    ep=3,
    lag=0.5,
    frac=0.8,
    Srz=200,
    GW=1500,
    porosity=0.01,
    b=1,         # default value assumed, required by model logic
    d=0          # default value assumed, required by model logic
):
    """
    Run the MGH hydrological model (variant 4 with exponential percolation).

    Parameters:
        data (DataFrame): Must contain 'PPT', 'PET', and 'SF' columns.
        All other arguments are model parameters with defaults.

    Returns:
        DataFrame: Hydrological outputs per time step.
    """


    length = len(data)
    Srz_arr = np.zeros(length + 1)
    GW_arr = np.zeros(length + 1)
    Qsurf = np.zeros(length)
    ET = np.zeros(length)
    Qbase = np.zeros(length)
    Qperc = np.zeros(length + 1)
    Qrunoff = np.zeros(length)
    Q = np.zeros(length)
    S = np.zeros(length)
    Qstorage = np.zeros(length)

    # Initial conditions
    Srz_arr[0] = Srz
    GW_arr[0] = GW

    for i in range(length):
        rel_srz = Srz_arr[i] / Srzmax
        storage_expr = rel_srz**b + c

        # Storage capacity logic
        if storage_expr > 1:
            S[i] = 1
        elif storage_expr - d < threshold1:
            S[i] = 0
        else:
            S[i] = storage_expr

        # Evapotranspiration
        ET[i] = (rel_srz) ** beta * data['PET'].iloc[i]

        # Surface runoff
        Qsurf[i] = data['PPT'].iloc[i] * S[i] ** alpha

        # Percolation (exponential form)
        Qperc[i] = K * (1 - np.exp(-gamma * rel_srz))

        # Update Srz
        Srz_arr[i + 1] = Srz_arr[i] + (data['PPT'].iloc[i] - ET[i] - Qsurf[i] - Qperc[i])
        Srz_arr[i + 1] = np.clip(Srz_arr[i + 1], 0, Srzmax)

        # Base flow
        if GW_arr[i] / GWmax > threshold2:
            Qbase[i] = theta * (GW_arr[i] / GWmax) ** ep
        else:
            Qbase[i] = 0

        # Runoff routing
        Qrunoff[i] = frac * Qsurf[i]
        Qstormflow = (1 - frac) * Qsurf[i]
        Qperc[i] += Qstormflow

        # Update groundwater
        GW_arr[i + 1] = GW_arr[i] + (Qperc[i] - Qbase[i]) / porosity
        GW_arr[i + 1] = np.clip(GW_arr[i + 1], 0, GWmax)

        # Storage-based lag
        if i > 0:
            Qrunoff[i] = (Qstorage[i - 1] + Qrunoff[i]) * (1 - np.exp(-lag))
        Qstorage[i] = (Qstorage[i - 1] + Qsurf[i]) * np.exp(-lag)

        # Total flow
        Q[i] = Qrunoff[i] + Qbase[i]

    # Output DataFrame
    return pd.DataFrame({
        'Srz': Srz_arr[:-1],
        'S': S,
        'ET': ET,
        'GW': GW_arr[:-1],
        'P': data['PPT'].values[:length],
        'Qsim': Q,
        'Qobs': data['SF'].values[:length],
        'Qsurf': Qsurf,
        'Qbase': Qbase,
        'Qrunoff': Qrunoff,
        'Qperc': Qperc[:-1],
        'Qstorage': Qstorage,
    }, index=data.index)
