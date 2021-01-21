import numpy as np


def read_energy_data(is_train):
    if is_train:
        with open('./data/SoDa_HC3-METEO_lat0.329_lon32.499_2005-01-01_2005-12-31_1833724734.csv', 'r') as f:
            readlines = f.readlines()
            result = list(readlines)[32:]

    else:
        with open('./data/SoDa_HC3-METEO_lat0.329_lon32.499_2006-01-01_2006-12-31_1059619648.csv', 'r') as f:
            readlines = f.readlines()
            result = list(readlines)[32:]
    GHI = np.zeros(len(result))
    for i in range(len(GHI)):
        GHI[i] = int(result[i].split(";")[2])
    return GHI
