# This is a file module which contains classes and functions 
# which calculate the cosmological model of LCDM + stiff + constant N_eff

import os, yaml, math
import numpy as np
from numpy import concatenate as cat
from scipy import interpolate
from pathlib import Path

from global_param import *
from functions import int_FD


class LCDM_SN:
    """
    Cosmological model: LCDM + stiff component + constant Delta N_eff

    The free/base parameters of the model are: 
    'Omega_bh2', 'Omega_ch2', 'H0', 'DN_eff', 'A_s', 'r', 'T_re', 'T_sr'.
    [H0] = km s^-1 Mpc^-1, [T_re] = GeV, [T_sr] = GeV.

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
            self.cosmo_param = {'Omega_bh2': 0.0223828, 'Omega_ch2': 0.1201075, 'H0': 67.32117, 'DN_eff': 0,
                                'A_s': 2.100549e-9, 'r': 0, 
                                'T_re': 1e12, 'T_sr': 1e12, 
                               }                               # default values: Planck 2018 Plik best fit 
            if len(kwargs) > 0:
                for key in kwargs:
                    if key in self.cosmo_param:
                        self.cosmo_param[key] = kwargs[key]
                        
        
    @property  
    def derived_param(self):
        derived_dict = {
            'h': self.cosmo_param['H0']/100,
            'H_0': self.cosmo_param['H0']/(10*parsec),     # s^-1
        
            'Omega_mh2': self.cosmo_param['Omega_bh2'] + self.cosmo_param['Omega_ch2'],      # baryon + CDM
            
            'Omega_trh2': Omega_orh2 + Omega_nh2 * self.cosmo_param['DN_eff']/Neff0,     # total radiation (including extra rad)
            'Omega_treh2': Omega_oreh2 + Omega_nh2 * self.cosmo_param['DN_eff']/Neff0,   # total rho_rad before e+e- annihilation
        
            'Omega_sh2': Omega_oreh2 * (kB*Tnu/(1e9*self.cosmo_param['T_sr']))**2,
        
            'nt': -self.cosmo_param['r']/8,                                     # consistency relation
            'A_t': self.cosmo_param['r'] * self.cosmo_param['A_s'],
        
            'N_re': math.log(1e9*self.cosmo_param['T_re']/(kB*Tnu)),            # assume neutrinos are decoupled from the beginning
        }
        
        derived_dict['Omega_m'] = derived_dict['Omega_mh2']/derived_dict['h']**2
        
        derived_dict['Omega_mnu'] = Omega_mnuh2/derived_dict['h']**2            # the massive neutrino eigenstate
        
        derived_dict['Omega_s'] = derived_dict['Omega_sh2']/derived_dict['h']**2
        # stiff-radiation transition must occur before e+e- annihilation      
        
        derived_dict['V_inf'] = (1.5*derived_dict['A_t'])**.25*math.pi**.5*M_Pl/1e9          # (~*10^9 GeV)^4, energy scale of inflaion
        try:                                   # lookback number of e-foldings at the end of inflation. If r=0, set N_inf=60 by default
            derived_dict['N_inf'] = math.log((1.5*derived_dict['A_t']*15/(43/8))**.25 * M_Pl/(kB*Tnu))    # g_*/2 = 43/8 before e+e- annihilation
        except ValueError as exp_r:
            #print(exp_r, ', caused by setting r=0 in a single-field model')
            derived_dict['N_inf'] = 60       
        
        return derived_dict
      

    
    
    def gen_expansion(self):
        """
        Generate the expansion history of an extended Lambda-CDM model 
        with an extra stiff matter and a constant Delta_Neff (which can
        mimic the effect of the primordial SGWB)

        N_re and N_inf are LOOKBACK numbers of e-foldings 
        at the end of reheating and inflation, respectively
        N_re = ln (T_re/T_CMB), N_BBN < N_re < N_inf
        
        """
    
        #####    Import cosmology    #####

        Op = Omega_ph2/self.derived_param['h']**2              # photons 
        Onu = Omega_nh2/self.derived_param['h']**2             # relativistic neutrinos
        Oer  = Onu * self.cosmo_param['DN_eff']/Neff0          # extra radiation
        Or = self.derived_param['Omega_trh2']/self.derived_param['h']**2       # total radiation (before neutrinos become non-relativistic)
        Ore = self.derived_param['Omega_treh2']/self.derived_param['h']**2

        Om = self.derived_param['Omega_m']; Omnu = self.derived_param['Omega_mnu']; Os = self.derived_param['Omega_s']
        OL = 1 - Om - Omnu - Onu*2/3 - Op - Oer - Os         # Omega_Lambda at the present

    
        #####   Construct output arrays    #####
    
        N_inf = math.floor(self.derived_param['N_inf']*100); Nv = np.arange(0, N_inf+1)*.01
        # Nv equivalent to N = ln a. Present-day value set equal to N_inf
        Sv = np.zeros_like(Nv); fv = np.zeros_like(Nv)  # fv proportional to f_H = aH/(2*pi)

        len_Nre = math.floor(self.derived_param['N_re']*100)
        index_re = len(Nv)-1-len_Nre
    

        #####    Load thermal history    #####

        Nref = np.log(np.array([x.split()[0] for x in open('th.txt').readlines()], dtype=float)) + Nv[-1]
        Tref = np.array([x.split()[1] for x in open('th.txt').readlines()], dtype=float) * 1e-6     # T_photon in MeV
        Sref = np.array([x.split()[2] for x in open('th.txt').readlines()], dtype=float)
        Eref = np.array([x.split()[3] for x in open('th.txt').readlines()], dtype=float)
        #  Entropy and energy density of photons, elections and positrons

        spline_Sref = interpolate.InterpolatedUnivariateSpline(Nref, Sref)
        spline_Eref = interpolate.InterpolatedUnivariateSpline(Nref, Eref)    

        # Compute the ratio of rho_stiff to rho_photon at T=1 MeV   
        spline_N = interpolate.InterpolatedUnivariateSpline(np.flip(Tref), np.flip(Nref))
        spline_S = interpolate.InterpolatedUnivariateSpline(np.flip(Tref), np.flip(Sref))
        N_MeV = spline_N(1)
        S_MeV = spline_S(1)

        self.stiff_to_photon_MeV = Os * math.exp(2.0*(Nv[-1]-N_MeV)) / (Op/S_MeV**(4/3))
        
    
        #####    Main loop: Calculating expansion history    #####  

        for i in range(index_re, len(Nv), 1):
            eN = math.exp(Nv[-1]-Nv[i]); e3N = math.exp(3.0*(Nv[-1]-Nv[i]))  # 1/a and 1/a^3
            nu = nu_today / eN          #  (m_nu*c^2)/(kB*T) 
            if (nu > 100):              #  massive neutrinos become highly non-relativistic
                H2 = Om + Omnu + (Op+2/3*Onu+Oer)*eN + Os*e3N + OL/e3N    # H^2 * e^{3(N-N_end)} = H^2 * a^3
                Sv[i] = (Om + Omnu + 4/3*(Op+2/3*Onu+Oer)*eN + 2*Os*e3N)/H2
            elif (nu <= 100) and (nu >= 0.1):
                [rho_nu, p_nu] = int_FD(nu)
                H2 = Om + (Op+(2/3+rho_nu/3)*Onu+Oer)*eN + Os*e3N + OL/e3N
                Sv[i] = (Om + 4/3*(Op+2/3*Onu+Oer)*eN + (rho_nu+p_nu)*Onu/3*eN + 2*Os*e3N)/H2
            elif (nu < 0.1) and (Nv[i] > Nref[-1]):     # massive neutrinos become highly relativistic
                H2 = Om + Or*eN + Os*e3N + OL/e3N  
                Sv[i] = (Om + 4/3*Or*eN + 2*Os*e3N)/H2
            elif (Nv[i] < Nref[0]):
                H2 = Om + Ore*eN + Os*e3N + OL/e3N
                Sv[i] = (Om + 4/3*Ore*eN + 2*Os*e3N)/H2
            else:
                Sref_i = spline_Sref(Nv[i])
                Eref_i = spline_Eref(Nv[i])
                H2 = Om + (Op * Eref_i/Sref_i**(4/3) + Onu + Oer)*eN + Os*e3N + OL/e3N
                Sv[i] = (Om + 4/3*(Op * Sref_i**(-1/3) + Onu + Oer)*eN + 2*Os*e3N)/H2

            fv[i] = -0.5*Nv[i] + 0.5*math.log(H2)    #  N + ln H

        Sv[0:index_re] = 1
        fv[0:index_re] = fv[index_re] - 0.5*(Nv[0:index_re] - Nv[index_re])
        # reheating is assumed to be MD since the end of inflation

        # set fv = 0 for the reference frequency f_yr. 
        # fv here is really \tilde f_H = ln (f_H/f_yr)
        f0 = fv[-1]; f_inf = fv[0]; fmin = math.floor(fv.min()-f_inf)
        Delta_f = math.log(2*math.pi*f_yr/self.derived_param['H_0'])
        fv = fv - f0 - Delta_f


        #####   Construct the vector of sampled frequencies to calculate transfer
        #####   functions, chosen empirically -- more points around transition!

        f = np.arange(0, (N_inf/100-self.derived_param['N_re'])/5) * (-2)   # before the peak, during reheating
        f = cat((f, f[-1]-np.arange(1,53)*.2), axis=None)                   # around the peak
        f = cat((f, -np.arange(1, self.derived_param['N_re']-math.log(Ore/Os)/2+4)+f[-1]), axis=None)  # after the peak, before T_sr
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