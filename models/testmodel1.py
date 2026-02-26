import numpy as np
import pandas as pd


# Define parameters in a dictionary
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
    # b=1,
    # d=0
):
    """
    Run the MGH hydrological model.

    Parameters:
    data (DataFrame): Input data with 'PPT', 'PET', and 'SF'.
    Other parameters are scalar model parameters with defaults (see function signature).

    Returns:
    DataFrame: Model outputs per time step.
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
        # Storage capacity calculation
        storage_expr = (Srz_arr[i] / Srzmax)  + c# (Srz_arr[i] / Srzmax) ** b + c
        if storage_expr > 1:
            S[i] = 1
        elif storage_expr  < threshold1:
            S[i] = 0
        else:
            S[i] = storage_expr

        # Evapotranspiration
        ET[i] = (Srz_arr[i] / Srzmax) ** beta * data['PET'].iloc[i]

        # Surface runoff
        Qsurf[i] = data['PPT'].iloc[i] * S[i] ** alpha

        # Percolation
        Qperc[i] = K * (Srz_arr[i] / Srzmax) ** gamma

        # Update root zone storage
        Srz_arr[i + 1] = Srz_arr[i] + (data['PPT'].iloc[i] - ET[i] - Qsurf[i] - Qperc[i])
        Srz_arr[i + 1] = np.clip(Srz_arr[i + 1], 0, Srzmax)

        # Baseflow
        if GW_arr[i] / GWmax > threshold2:
            Qbase[i] = theta * (GW_arr[i] / GWmax) ** ep
        else:
            Qbase[i] = 0

        # Runoff and stormflow
        Qrunoff[i] = frac * Qsurf[i]
        Qstormflow = (1 - frac) * Qsurf[i]
        Qperc[i] += Qstormflow

        # Update groundwater
        GW_arr[i + 1] = GW_arr[i] + (Qperc[i] - Qbase[i]) / porosity
        GW_arr[i + 1] = np.clip(GW_arr[i + 1], 0, GWmax)

        # Lag effect
        if i > 0:
            Qrunoff[i] = (Qstorage[i - 1] + Qrunoff[i]) * (1 - np.exp(-lag))
        Qstorage[i] = (Qstorage[i - 1] + Qsurf[i]) * np.exp(-lag)
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
