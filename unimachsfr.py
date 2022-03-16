import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf

def sfr_from_vMpeak(vMpeak,z):
    a = 1/(1+z);
    logV = 2.151 - 1.658*(1-a) + 1.680*np.log(1+z) - 0.233*z
    loge = 0.109 - 3.441*(1-a) + 5.079*np.log(1+z) - 0.781*z
    alpha = -5.598 - 20.731*(1-a) + 13.455*np.log(1+z) - 1.321*z
    beta = -1.911 + 0.395*(1-a) + 0.747*z
    logg = -1.699 + 4.206*(1-a) - 0.809*z
    delta = 0.055
    logv = np.log10(vMpeak) - logV
    v = 10**logv
    return 10**loge*(1/(v**alpha+v**beta)+10**logg*np.exp(-logv**2/(2*delta**2)))

def fquench(vMpeak,z):
    a = 1/(1+z);
    Qmin = np.maximum(0,-1.944+(-2.419)*(1-a));
    logVQ = 2.248 - 0.018*(1-a) + 0.124*z;
    sigVQ = 0.227 + 0.037*(1-a) - 0.107*np.log(1+z);
    return Qmin + (1-Qmin)*(0.5+0.5*erf((np.log10(vMpeak)-logVQ)/(2**0.5*sigVQ)))

def vMpeak_from_Mh(Mh,z):
    a = 1/(1+z);
    M200kms = 1.64e12/((a/0.378)**(-0.142)+(a/0.378)**(-1.79)) # MSol
    return 200*(Mh/M200kms)**0.3 # km/s

def sfr_from_Mh(Mh,z,quench=True,Csigma=None):
    sfr_sf = sfr_from_vMpeak(vMpeak_from_Mh(Mh,z),z)
    result = sfr_sf*(1-quench*fquench(vMpeak_from_Mh(Mh,z),z))
    if 'Csigma' not in globals().keys() or Csigma == None:
        return result;
    else:
        return result/Csigma(z);