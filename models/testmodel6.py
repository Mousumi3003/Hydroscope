import numpy as np
import pandas as pd

def model(data,
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
                  Srz_init=200,
                  GW_init=1500,
                  porosity=0.01,
                  b=1.0,
                  d=0.0,
                  use_exp_perc=False,
                  use_two_gw_stores=False,
                  Qsurf_version="v1"):
    length = len(data)
    Srz = np.zeros(length + 1)
    GW = np.zeros(length + 1)
    Qsurf = np.zeros(length)
    ET = np.zeros(length)
    Qbase = np.zeros(length)
    Qperc = np.zeros(length + 1)
    Qrunoff = np.zeros(length)
    Q = np.zeros(length)
    S = np.zeros(length)
    Qstorage = np.zeros(length)

    if use_two_gw_stores:
        GW_shallow = np.zeros(length + 1)
        GW_deep = np.zeros(length + 1)

    Srz[0] = Srz_init
    GW[0] = GW_init

    for i in np.arange(length):
        sat_ratio = Srz[i] / Srzmax
        storage_term = sat_ratio ** b + c
        if storage_term > 1:
            S[i] = 1
        elif storage_term - d < threshold1:
            S[i] = 0
        else:
            S[i] = storage_term

        ET[i] = beta * sat_ratio * data['PET'].iloc[i]

        rain = data['PPT'].iloc[i]
        if Qsurf_version == "v3":
            if sat_ratio > threshold1:
                Qsurf[i] = (rain ** alpha) * (sat_ratio - threshold1) ** beta
            else:
                Qsurf[i] = 0
        else:
            Qsurf[i] = rain * S[i] ** alpha

        if use_exp_perc:
            Qperc[i] = K * (1 - np.exp(-gamma * sat_ratio))
        else:
            Qperc[i] = K * sat_ratio ** gamma

        Srz[i + 1] = Srz[i] + (rain - ET[i] - Qsurf[i] - Qperc[i])
        Srz[i + 1] = np.clip(Srz[i + 1], 0, Srzmax)

        if use_two_gw_stores:
            Qbase_shallow = theta * (GW_shallow[i] / GWmax)
            Qbase_deep = theta * 0.5 * (GW_deep[i] / GWmax)
            GW_shallow[i + 1] = GW_shallow[i] + (Qperc[i] - Qbase_shallow) / porosity
            GW_deep[i + 1] = GW_deep[i] + (Qbase_shallow - Qbase_deep) / porosity
            Qbase[i] = Qbase_deep
        else:
            if GW[i] / GWmax > threshold2:
                Qbase[i] = theta * (GW[i] / GWmax) ** ep
            else:
                Qbase[i] = 0

        Qrunoff[i] = frac * Qsurf[i]
        Qstormflow = (1 - frac) * Qsurf[i]
        Qperc[i] += Qstormflow

        GW[i + 1] = GW[i] + (Qperc[i] - Qbase[i]) / porosity
        GW[i + 1] = np.clip(GW[i + 1], 0, GWmax)

        if i > 0:
            Qrunoff[i] = (Qstorage[i - 1] + Qrunoff[i]) * (1 - np.exp(-lag))
        Qstorage[i] = (Qstorage[i - 1] + Qsurf[i]) * np.exp(-lag)
        Q[i] = Qrunoff[i] + Qbase[i]

    output_data = pd.DataFrame({
        'Srz': Srz[:-1],
        'S': S,
        'ET': ET,
        'GW': GW[:-1],
        'P': data['PPT'].values[:length],
        'Qsim': Q,
        'Qobs': data['SF'].values[:length],
        'Qsurf': Qsurf,
        'Qbase': Qbase,
        'Qrunoff': Qrunoff,
        'Qperc': Qperc[:-1],
        'Qstorage': Qstorage,
    }, index=data.index)

    return output_data
