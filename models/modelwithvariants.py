import numpy as np
import pandas as pd


# Define parameters in a dictionary
def modelwithvaraints(
    data,
    variant=1,
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
    S = np.zeros(length)
    Qstorage = np.zeros(length)

    # Initial conditions
    Srz_arr[0] = Srz
    GW_arr[0] = GW

    for i in range(length):
        
        # ---------------------
        # 1. Saturation (S)
        # ---------------------
        if variant in [3, 7, 8, 9, 10]:
            # Alternate S representation
            storage_expr = max(0, min(1, (Srz_arr[i] - threshold1 * Srzmax) / ((1 - threshold1) * Srzmax)))
        else:
            storage_expr = (Srz_arr[i] / Srzmax) + c
    
        if storage_expr > 1:
            S[i] = 1
        elif storage_expr < threshold1:
            S[i] = 0
        else:
            S[i] = storage_expr
    
        # ---------------------
        # 2. Evapotranspiration (ET)
        # ---------------------
        if variant in [2, 7, 8, 9, 10]:
            # Alternate ET: soil moisture stress-limited PET
            ET[i] = min(data['PET'].iloc[i], Srz_arr[i] / (Srzmax + 1e-6))
        else:
            # Default nonlinear ET
            ET[i] = (Srz_arr[i] / Srzmax) ** beta * data['PET'].iloc[i]
    
        # ---------------------
        # 3. Surface Runoff (Qsurf)
        # ---------------------
        if variant in [4, 8, 9, 10]:
            # Alternate surface runoff: linear threshold-based
            if S[i] > threshold1:
                Qsurf[i] = data['PPT'].iloc[i] * (S[i] - threshold1)
            else:
                Qsurf[i] = 0
        else:
            Qsurf[i] = data['PPT'].iloc[i] * S[i] ** alpha
    
        # ---------------------
        # 4. Percolation (Qperc)
        # ---------------------
        if variant in [5, 9, 10]:
            # Alternate percolation: exponential function
            Qperc[i] = K * (1 - np.exp(-Srz_arr[i] / Srzmax))
        else:
            Qperc[i] = K * (Srz_arr[i] / Srzmax) ** gamma
    
        # ---------------------
        # 5. Update root zone storage
        # ---------------------
        Srz_arr[i + 1] = Srz_arr[i] + (data['PPT'].iloc[i] - ET[i] - Qsurf[i] - Qperc[i])
        Srz_arr[i + 1] = np.clip(Srz_arr[i + 1], 0, Srzmax)
    
        # ---------------------
        # 6. Baseflow (Qbase)
        # ---------------------
        if variant in [6, 10]:
            # Alternate baseflow: linear threshold-based
            if GW_arr[i] / GWmax > threshold2:
                Qbase[i] = theta * (GW_arr[i] / GWmax - threshold2)
            else:
                Qbase[i] = 0
        else:
            if GW_arr[i] / GWmax > threshold2:
                Qbase[i] = theta * (GW_arr[i] / GWmax) ** ep
            else:
                Qbase[i] = 0
    
        # ---------------------
        # 7. Stormflow + Runoff Routing
        # ---------------------
        Qrunoff[i] = frac * Qsurf[i]
        Qstormflow = (1 - frac) * Qsurf[i]
        Qperc[i] += Qstormflow  # combine stormflow with percolation
    
        # ---------------------
        # 8. Update groundwater storage
        # ---------------------
        GW_arr[i + 1] = GW_arr[i] + (Qperc[i] - Qbase[i]) / porosity
        GW_arr[i + 1] = np.clip(GW_arr[i + 1], 0, GWmax)
    
        # ---------------------
        # 9. Lagged runoff
        # ---------------------
        if i > 0:
            Qrunoff[i] = (Qstorage[i - 1] + Qrunoff[i]) * (1 - np.exp(-lag))
        Qstorage[i] = (Qstorage[i - 1] + Qsurf[i]) * np.exp(-lag)
    
        # ---------------------
        # 10. Total streamflow
        # ---------------------
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
