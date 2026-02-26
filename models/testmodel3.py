import numpy as np
import pandas as pd

def model(
    data,
    alpha=0.5,
    beta=0.8,
    gamma=0.9,
    K=0.5,
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
    porosity=0.01
):

    length = len(data)
    Srz_arr = np.zeros(length + 1)
    GW_arr = np.zeros(length + 1)
    Qsurf = np.zeros(length)
    ET = np.zeros(length)
    Qbase = np.zeros(length)
    Qperc = np.zeros(length + 1)
    Qrunoff = np.zeros(length)
    Q = np.zeros(length)
    # S = np.zeros(length)
    Qstorage = np.zeros(length)

    # Initial conditions
    Srz_arr[0] = Srz
    GW_arr[0] = GW

    for i in range(length):
        # Storage state (used only for tracking or optional modifications)
        # S[i] = max(0, min(1, Srz_arr[i] / Srzmax))

        # Surface runoff based on Srz threshold
        if Srz_arr[i] / Srzmax > threshold1:
            Qsurf[i] = data['PPT'].iloc[i] * ((Srz_arr[i] / Srzmax) - threshold1) ** alpha
        else:
            Qsurf[i] = 0
            
        # sat = Srz_arr[i] / Srzmax
        # rain = data['PPT'].iloc[i]
        
        # if sat > threshold1:
        #     Qsurf[i] = (rain ** alpha) * (sat - threshold1) ** beta
        # else:
        #     Qsurf[i] = 0


        # Evapotranspiration (simplified linear form)
        ET[i] = beta * (Srz_arr[i] / Srzmax) * data['PET'].iloc[i]

        # Percolation
        Qperc[i] = K * (Srz_arr[i] / Srzmax) ** gamma

        # Update Srz
        Srz_arr[i + 1] = Srz_arr[i] + (data['PPT'].iloc[i] - ET[i] - Qsurf[i] - Qperc[i])
        Srz_arr[i + 1] = np.clip(Srz_arr[i + 1], 0, Srzmax)

        # Baseflow from GW
        if GW_arr[i] / GWmax > threshold2:
            Qbase[i] = theta * (GW_arr[i] / GWmax) ** ep
        else:
            Qbase[i] = 0

        # Split runoff
        Qrunoff[i] = frac * Qsurf[i]
        Qstormflow = (1 - frac) * Qsurf[i]
        Qperc[i] += Qstormflow

        # Update GW
        GW_arr[i + 1] = GW_arr[i] + (Qperc[i] - Qbase[i]) / porosity
        GW_arr[i + 1] = np.clip(GW_arr[i + 1], 0, GWmax)

        # Lagged runoff and storage
        if i > 0:
            Qrunoff[i] = (Qstorage[i - 1] + Qrunoff[i]) * (1 - np.exp(-lag))
        Qstorage[i] = (Qstorage[i - 1] + Qsurf[i]) * np.exp(-lag)

        # Total discharge
        Q[i] = Qrunoff[i] + Qbase[i]

    # Output
    return pd.DataFrame({
        'Srz': Srz_arr[:-1],
        # 'S': S,
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
