# This is a file module which contains classes and functions 
# which calculate the cosmological model of LCDM + stiff + constant N_eff

import os, yaml, math
import numpy as np
from numpy import concatenate as cat
#from scipy import interpolate
from pathlib import Path

from global_param import *
from functions import int_FD


class LCDM_SN:
    """
    Cosmological model: LCDM + stiff component + constant Delta N_eff

    The free/base parameters of the model are: 
    'Omega_bh2', 'Omega_ch2', 'H0', 'DN_eff', 'A_s', 'r', 'n_t', 'cr', 'T_re', 'DN_re', 'kappa10'.
    - [H0] = km s^-1 Mpc^-1, [T_re] = GeV.
    - Set cr > 0 if the consistency relation is assumed, otherwise set cr <= 0 and provide (n_t, DN_re).
    - DN_re is the number of e-folds from the end of inflation to the end of reheating, 
      assuming a^{-3} matter-like evolution.
    - kappa10 := rho_stiff/rho_photon at 10 MeV.

    There are three ways to instantiate a model with desired base parameters: 
    1. input a yaml file,  2. input a dictionary, 
    3. specify the parameters by keyword arguments.
    
    They will be stored in the 'obj_name.cosmo_param' attribute as a dictionary
    and can be modified later on.
  
    """
  
    def __init__(self, *args, 
                 **kwargs):
        if len(args) > 0:
            if type(args[0]) == str:
                if os.path.exists(args[0]): 
                    if Path(args[0]).suffix == 'yml' or 'yaml':
                        with open(args[0], 'r') as stream:
                            try:
                                self.cosmo_param = yaml.safe_load(stream)
                            except yaml.YAMLError as exc:
                                print(exc)
            elif type(args[0]) == dict:
                self.cosmo_param = args[0].copy()
        else:
            self.cosmo_param = {'Omega_bh2': 0.0223828, 'Omega_ch2': 0.1201075, 'H0': 67.32117, 'DN_eff': 0.,
                                'A_s': 2.100549e-9, 'r': 0., 'n_t': 0., 'cr': 0,
                                'T_re': 1e12, 'DN_re': 0, 'kappa10': 0., 
                               }    # default values: Planck 2018 Plik best fit, no consistency relation, no stiff matter
            if len(kwargs) > 0:
                for key in kwargs:
                    if key in self.cosmo_param:
                        self.cosmo_param[key] = kwargs[key]
                        
        
    @property  
    def derived_param(self):
        derived_dict = {
            'h': self.cosmo_param['H0']/100,
            'H_0': self.cosmo_param['H0']/(10*parsec),     # s^-1
        
            'Omega_mh2': self.cosmo_param['Omega_bh2'] + self.cosmo_param['Omega_ch2'],                  # baryons + CDM     
            'Omega_sh2': Omega_ph2 * self.cosmo_param['kappa10'] * (1e-2*a_10/TCMB_GeV)**4 * a_10**2,    # Omega_stiff*h^2 at the present

            'A_t': self.cosmo_param['r'] * self.cosmo_param['A_s'],
            'nt': input_nt(self.cosmo_param),

            'kappa_s': self.cosmo_param['kappa10'] * (1e-2/T_i)**4 * np.exp(6*(N_i-N_10))                # kappa_stiff(T_i) for AlterBBN
        }

        if self.cosmo_param['T_re'] >= T_max:
            derived_dict['N_re'] = N_max + math.log(self.cosmo_param['T_re']/T_max)
            derived_dict['rho_re'] = rho_th[-1]
        else:
            derived_dict['N_re'] = spl_T_N(math.log10(self.cosmo_param['T_re']))
            derived_dict['rho_re'] = spl_rho(derived_dict['N_re'])
        derived_dict['f_re'] = np.exp(derived_dict['N_re']) * math.sqrt(TCMB_GeV**4*derived_dict['rho_re'] \
            + self.cosmo_param['kappa10']*1e-8*np.exp(2*derived_dict['N_re']-6*N_10))*1e9 / (6*math.sqrt(5)*M_Pl*hbar)   # Hz, frequency at the end of reheating

        derived_dict['N_inf'] = None
        if self.cosmo_param['cr'] > 0:
            if self.cosmo_param['r'] > 0:
                derived_dict['V_inf'] = (1.5*derived_dict['A_t'])**.25 * math.pi**.5 * M_Pl
                # ( GeV)^4, energy scale of single field, slow-roll inflaion
                
                Delta_N = math.log(M_Pl)*4/3 - derived_dict['N_re']*4/3 + math.log(45/2*derived_dict['A_t'] \
                        / (TCMB_GeV**4*derived_dict['rho_re'] + self.cosmo_param['kappa10']*1e-8*np.exp(2*derived_dict['N_re']-6*N_10)))/3   
                # Lookback number of e-folds from the end of inflation to the end of reheating, a^{-3} matter-like evolution

                if Delta_N >= 0:
                    derived_dict['N_inf'] = derived_dict['N_re'] + Delta_N
                    self.cosmo_param['DN_re'] = Delta_N
                else:
                    print('V_inf smaller than rho(T_re)! Adjust relevant input parameters: r, T_re, kappa10.')
            else:
                print('r cannot be set to zero in a single-field, slow-roll inflation. Use positive r!')
        else:
            derived_dict['N_inf'] = derived_dict['N_re'] + self.cosmo_param['DN_re']
        
        return derived_dict   

    
    
    
    def gen_expansion(self):
        """
        Generate the expansion history of an extended Lambda-CDM model 
        with an extra stiff matter and a constant Delta_Neff (which can
        mimic the effect of the primordial SGWB)

        N_re and N_inf are LOOKBACK numbers of e-folds 
        at the end of reheating and inflation, respectively
        N_re = ln (T_re/T_CMB), N_BBN < N_re < N_inf

        Run this function only when self.derived_param['N_inf'] is not None.
        
        """
    
        #####    Import cosmology    #####

        Omh2 = self.derived_param['Omega_mh2']; Osh2 = self.derived_param['Omega_sh2']
        
        Oerh2 = Omega_ph2 * 7/8 * (4/11)**(4/3) * self.cosmo_param['DN_eff']         # Omega_{extra rad}*h^2
        Otrh2 = Omega_orh2 + Oerh2                                                   # total radiation after e+e- annihilation
        Otreh2 = Omega_ph2 * rho_th[-1] + Oerh2                                      # total radiation before any SM phase transition
        
        OLh2 = self.derived_param['h']**2 - Omh2 - Omega_mnuh2 - Omega_nh2*2/3 - Omega_ph2 - Oerh2 - Osh2     # Omega_Lambda*h^2

    
        #####   Construct output arrays    #####
    
        len_inf = math.floor(self.derived_param['N_inf']*100)+1; Nv = np.arange(0, len_inf)*.01
        # Nv is equivalent to ln(a), its present-day value is set at Nv[-1] = N_inf.
        Sv = np.zeros_like(Nv); fv = np.zeros_like(Nv)  # fv proportional to f_H = aH/(2*pi)

    
        #####    Main loop: Calculating expansion history    #####  

        index_re = len_inf-1 - math.floor(self.derived_param['N_re']*100)

        for i in range(index_re, len_inf, 1):
            eN = math.exp(Nv[-1]-Nv[i]); e3N = math.exp(3.0*(Nv[-1]-Nv[i]))  # 1/a and 1/a^3
            nu = nu_today / eN          #  (m_nu*c^2)/(kB*T_nu) 
            if (nu > 100):              #  massive neutrinos become highly non-relativistic
                H2 = Omh2 + Omega_mnuh2 + (Omega_ph2+2/3*Omega_nh2+Oerh2)*eN + Osh2*e3N + OLh2/e3N    # h^2 * e^{3(N-N_end)} = h^2 * a^3
                Sv[i] = (Omh2 + Omega_mnuh2 + 4/3*(Omega_ph2+2/3*Omega_nh2+Oerh2)*eN + 2*Osh2*e3N)/H2
            elif (nu <= 100) and (nu >= 0.1):
                [rho_nu, p_nu] = int_FD(nu)
                H2 = Omh2 + (Omega_ph2+(2/3+rho_nu/3)*Omega_nh2+Oerh2)*eN + Osh2*e3N + OLh2/e3N
                Sv[i] = (Omh2 + 4/3*(Omega_ph2+2/3*Omega_nh2+Oerh2)*eN + (rho_nu+p_nu)*Omega_nh2/3*eN + 2*Osh2*e3N)/H2
            elif (nu < 0.1) and (Nv[i] > Nv[-1]-N_fin):     # massive neutrinos become highly relativistic
                H2 = Omh2 + Otrh2*eN + Osh2*e3N + OLh2/e3N  
                Sv[i] = (Omh2 + 4/3*Otrh2*eN + 2*Osh2*e3N)/H2
            elif (Nv[i] <= Nv[-1]-N_fin) and (Nv[i] >= Nv[-1]-N_max):   # Use lookup table for thermal history, 20 keV ~- 10^6 GeV. 
                rho_i = spl_rho(Nv[-1]-Nv[i])
                rhop_i = spl_rhop(Nv[-1]-Nv[i])
                H2 = Omh2 + (Omega_ph2*rho_i + Oerh2)*eN + Osh2*e3N + OLh2/e3N
                Sv[i] = (Omh2 + (Omega_ph2*rhop_i + 4/3*Oerh2)*eN + 2*Osh2*e3N)/H2
            else:
                H2 = Omh2 + Otreh2*eN + Osh2*e3N + OLh2/e3N
                Sv[i] = (Omh2 + 4/3*Otreh2*eN + 2*Osh2*e3N)/H2

            fv[i] = -0.5*Nv[i] + 0.5*math.log(H2)    #  N + ln H

        Sv[0:index_re] = 1
        fv[0:index_re] = fv[index_re] - 0.5*(Nv[0:index_re] - Nv[index_re])
        # reheating is assumed to be MD since the end of inflation

        # set fv = 0 for the reference frequency f_yr. 
        # fv here is really \tilde f_H := ln (f_H/f_yr)
        f0 = fv[-1]; f_inf = fv[0]; fmin = math.floor(fv.min()-f_inf)
        Delta_f = math.log(2*math.pi*f_yr/self.derived_param['H_0'])
        fv = fv - f0 - Delta_f


        #####   Construct the vector of sampled frequencies to calculate tensor transfer functions,
        #####   chosen empirically -- more points around transition!

        if self.derived_param['N_inf']>self.derived_param['N_re']:
            f = np.arange(0, (self.derived_param['N_inf']-self.derived_param['N_re'])/5) * (-2)     # before the peak, during reheating
            f = cat((f, f[-1]-np.arange(1,53)*.2), axis=None)                                       # around the peak
        else:
            f = -np.arange(1,50)*.2                                                                 # after the peak, instantaneous reheating
            
        if (self.cosmo_param['kappa10']>0):
            f = cat((f, -np.arange(1, self.derived_param['N_re']-math.log(Otreh2/Osh2)/2+4)+f[-1]), axis=None)     # after the peak, before T_sr
            f = cat((f, -np.arange(1,15)*.5+f[-1]), axis=None)                  # around T_sr (the ankle), through BBN
        
        f = cat((f, f[-1]-2*np.arange(1,(f[-1]-fmin-10.0)/2)), axis=None)
        f = cat((f, np.arange(1,11)*(-.5)+f[-1]), axis=None); f = cat((f, np.arange(1,16)*(-.2)+f[-1]), axis=None)
        f = f + f_inf - f0 - Delta_f
        f = cat((f, np.arange(1,11)*(-.5)+f[-1]), axis=None); f = f[np.where(f>-26)]    # ~= Delta_f 
        
        
        self.f = f * math.log10(math.exp(1)) + math.log10(f_yr)       # Convert output frequency in log10(f/Hz)
        self.Nv = Nv
        self.N = Nv - Nv[-1]
        self.sigma = Sv
        self.f_hor = fv * math.log10(math.exp(1)) + math.log10(f_yr)


def input_nt(params):
    """
    Input the tensor spectral index. 
    Use the inflationary consistency relation or the provided n_t.    
    """
    if params['cr'] > 0:
        return -params['r']/8        # consistency relation
    else:
        return params['n_t']