params_FilteredBollingerBand = [
    {
        "ema_lookback": n,
        "vol_lookback": 48,
        "avg_vol_lookback": 6000,
        "width": 1,
        "ema_filter_lookback": n * 8,
    }
    for n in range(3,121,3)
]


for variation in params_FilteredBollingerBand:
    print(variation,end=',\n')