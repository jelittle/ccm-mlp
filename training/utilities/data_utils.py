import numpy as np

def normalize_cct(cct):
    #normalize cct to -1 to 1, with 1000K as -1 and 10000K as 1
    cct = np.clip(cct, 1000, 10000)
    cct = (cct - 1000) / 9000 * 2 - 1
    return cct
def denormalize_cct(cct):
    #denormalize cct from -1 to 1, with 1000K as -1 and 10000K as 1
    cct = (cct + 1) / 2 * 9000 + 1000
    return cct