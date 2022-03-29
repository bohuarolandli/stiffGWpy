# This is a file module which contains classes and functions 
# which calculate the cosmological model of LCDM + stiff + primordial SGWB

import os, yaml, math
import numpy as np
from numpy import concatenate as cat
from scipy import interpolate, integrate
from pathlib import Path

from global_param import *
from functions import int_FD, solve_SGWB
from LCDM_stiff_Neff import LCDM_SN


class LCDM_SG(LCDM_SN):
    """
    Cosmological model: LCDM + stiff component + SGWB. It is a derived class of LCDM_SN.
    
    The free/base parameters of the model are: 
    'Omega_bh2', 'Omega_ch2', 'H0', 'DN_eff', 'A_s', 'r', 'T_re', 'T_sr'.
    [H0] = km s^-1 Mpc^-1, [T_re] = GeV, [T_sr] = GeV.

    There are three ways to instantiate a model with desired base parameters: 
    1. input a yaml file,  2. input a dictionary, 
    3. specify the parameters by keyword arguments.
    
    They will be stored in the 'obj_name.cosmo_param' attribute as a dictionary
    and can be modified later on.
    
    Once a successful 'SGWB_iter' calculation has been performed for an instance, 
    if for some reason you would like to reuse this instance with a new set of 
    parameters, you may modify the parameters by changing the values in the 
    'obj_name.cosmo_param[]' dictionary. However, in this case, MAKE SURE that 
    you set 'obj_name.cosmo_param['DN_eff']' at the desired value (e.g., 0) and 
    run 'obj_name.reset()' right after those modifications in order to reset 
    the status of the SGWB calculation.
    
    """
    def __init__(self, *args, 
                 **kwargs):
        LCDM_SN.__init__(self, *args, **kwargs)
        self.reset()    
      
        
    def reset(self):
        #LCDM_SN.derived_param(self)        
        self.DN_eff_orig = None
        self.SGWB_converge = False     # Whether the SGWB has been successfully computed     
        
        
    @property
    def Ogw_min_NG12(self):
        """ 95% lower limit of log10(Omega_GW) from NANOGrav 12.5yr data """
        return math.log10(2*math.pi**2/3) + 2*math.log10(f_yr/self.derived_param['H_0'])+2*f_NG12 + 2*hc_NG12
        

        
        
    def run_SGWB(self):
        """
        Integrate the equation of motion of primordial tensor fluctuations 
        for selected frequencies and given expansion history
        """
        
        P_t = self.derived_param['A_t'] * (10**self.f/f_piv)**self.derived_param['nt']    # Primordial tensor power spectrum

        n_f = len(self.f); n_v = len(self.Nv)
        # Each element in the list is a frequency channel
        N_hc = []    # integration begins when the horizon crossing starts to occur
        Th = []      # Tensor transfer function
        Ogw = []     # dOmega_GW / dlnf
        Opgw = []    # dOmega_pGW / dlnf
        Oj = []      # \dot T_h * T_h / 3H * P_t
    
        for i in range(n_f):

            j = 0
            while ((self.f[i]+3) < self.fv[j]) and (j<n_v):  # solver begins at k/aH < 10^{-3}
                j = j+1
            if (j >= 1): j = j-1   

            z0 = (self.f[i]-self.fv[j])*math.log(10)         # convert to ln(2*pi*f*c/aH)
            result = solve_SGWB(self.Nv, self.Sv, j, z0)
            
            N_this = self.Nv[(self.Nv >= result.sol.t_min) & (self.Nv <= result.sol.t_max)]
            [zf_this, xf_this, yf_this] = result.sol(N_this)
            Th_this = np.divide(yf_this, np.exp(zf_this))
            Oj_this = np.multiply(xf_this, Th_this)/3 * P_t[i]
            Ogw_this = (np.power(xf_this,2) + np.power(yf_this,2))/24 * P_t[i] + Oj_this
            Opgw_this = (-5*np.power(xf_this,2) + 7*np.power(yf_this,2))/72 * P_t[i]
            
            # high-frequency, post-horizon-reentry regime
            if (N_this[-1]<self.Nv[-1]):
                N_hf = self.Nv[self.Nv > N_this[-1]]
                f_hf = self.fv[self.Nv > N_this[-1]]
        
                coeff = math.sqrt(0.5*(xf_this[-1]**2+yf_this[-1]**2))
                Th_hf = coeff * np.exp(-zf_this[-1] + N_this[-1] - N_hf)      # the rms T_h, time-averaged
                xf_hf = np.multiply(Th_hf, np.power(10, self.f[i]-f_hf))      # the rms x_f, time-averaged
                Oj_hf = -np.power(Th_hf,2)/3 * P_t[i]
                Opgw_hf = np.power(xf_hf,2)/36 * P_t[i]
                Ogw_hf = Opgw_hf * 3 + Oj_hf
                
                N_this = cat((N_this, N_hf), axis=None)
                Th_this = cat((Th_this, Th_hf), axis=None)
                Oj_this = cat((Oj_this, Oj_hf), axis=None)
                Ogw_this = cat((Ogw_this, Ogw_hf), axis=None)
                Opgw_this = cat((Opgw_this, Opgw_hf), axis=None)
                
            N_hc.append(N_this); Th.append(Th_this)
            Ogw.append(Ogw_this); Opgw.append(Opgw_this); Oj.append(Oj_this)
        
        self.N_hc = N_hc; self.Th = Th
        self.Ogw = Ogw; self.Opgw = Opgw; self.Oj = Oj

        
             
    def SGWB_iter(self):
        """
        Main numerical scheme: 
        Iteration method that yields self-consistent cosmology including the stiff-amplified primordial SGWB, 
        for which the extra radiation due to the SGWB is mimicked by a constant Delta N_eff.
        
        """
        
        if (self.SGWB_converge == False):
            Omega_nu = Omega_nh2/self.derived_param['h']**2
        
            self.DN_eff_orig = self.cosmo_param['DN_eff']
            self.DN_gw = 0; DN_gw_new = 0
            while True:    # main iteration
                self.gen_expansion()
                self.run_SGWB()
                self.int_SGWB()

                DN_gw_new = Neff0 * self.g2[-1] / Omega_nu
                   
                #print(DN_gw_new, self.DN_gw)
                if (self.DN_eff_orig + DN_gw_new > 5):
                    print('Total N_eff too large! Shorten the stiff era to lessen GW amplification.')
                    self.cosmo_param['DN_eff'] = self.DN_eff_orig
                    self.DN_eff_orig = None
                    return
                else:
                    self.cosmo_param['DN_eff'] = self.DN_eff_orig + DN_gw_new
                    if (self.DN_gw !=0 and abs(DN_gw_new/self.DN_gw - 1) <= 1e-4):
                        break
                
                self.DN_gw = DN_gw_new
                

            print(DN_gw_new, self.DN_gw, self.cosmo_param['DN_eff']) 
            self.SGWB_converge = True           
            self.hubble = math.log10(2*math.pi) + self.fv + math.log10(math.exp(1)) * (self.Nv[-1]-self.Nv)    # log10(H/s^-1), H = 2pi * fv / a 
            self.DN_gw = Neff0/Omega_nu * np.multiply(self.g2, np.exp(2*(self.fv-self.fv[-1])*math.log(10)+2*(self.Nv-self.Nv[-1])))
            # Obtain the entire evolution of the DN_eff due to SGWB. Now DN_gw[-1] + DN_eff_orig = cosmo_param['DN_eff']

            self.get_today()
            


    def get_today(self):
        """
        Obtain the present-day SGWB energy spectrum
        """
        self.Ogw_today=np.empty(0); self.Opgw_today=np.empty(0); self.Oj_today=np.empty(0)
        for i in range(len(self.f)): 
            self.Ogw_today = np.append(self.Ogw_today, self.Ogw[i][-1])
            self.Opgw_today = np.append(self.Opgw_today, self.Opgw[i][-1])
            self.Oj_today = np.append(self.Oj_today, self.Oj[i][-1])
                                
            
        
    def int_SGWB(self):
        """
        Integrate the SGWB spectrum to obtain the bolometric energy fractions at each moment.
        This step is important since it calculates the Delta N_eff due to the primordial SGWB.
        """
        self.g2 = np.zeros_like(self.Nv)      # total Omega_gw
        self.w2 = np.zeros_like(self.Nv)      # total Omega_pgw
        
        # Patch SGWB output arrays into a matrix for integration
        M_N_hc = -np.ones((len(self.f),len(self.Nv)))
        M_Ogw=np.empty((0,len(self.Nv))); M_Opgw=np.empty((0,len(self.Nv))); M_Oj=np.empty((0,len(self.Nv)))
        for i in range(len(self.f)):
            Ogw_evo = np.zeros((1,len(self.Nv))); Opgw_evo = np.zeros((1,len(self.Nv))); Oj_evo = np.zeros((1,len(self.Nv)))
            
            mask = np.isin(self.Nv,self.N_hc[i]); M_N_hc[i,mask]=self.N_hc[i]
            np.place(Ogw_evo, mask, self.Ogw[i]); M_Ogw = cat((M_Ogw, Ogw_evo))
            np.place(Opgw_evo, mask, self.Opgw[i]); M_Opgw = cat((M_Opgw, Opgw_evo))
            np.place(Oj_evo, mask, self.Oj[i]); M_Oj = cat((M_Oj, Oj_evo))
            
        M_N_hc = np.transpose(M_N_hc); M_Ogw = np.transpose(M_Ogw); M_Opgw = np.transpose(M_Opgw); M_Oj = np.transpose(M_Oj)
        
        for i in range(len(self.Nv)):
            ind_int = M_N_hc[i]>=0               # indices of frequencies/modes which start to enter or have entered the horizon
            f_int = np.flip(self.f[ind_int])     # in units of log10(f/Hz)
            Ogw_int = np.flip(M_Ogw[i,ind_int]); Opgw_int = np.flip(M_Opgw[i,ind_int]); Oj_int = np.flip(M_Oj[i,ind_int])
            
            # do not consider the negative super-horizon contribution from Omega_j, for the moment...    
            self.g2[i] = integrate.simpson(Ogw_int-Oj_int, f_int) * math.log(10)
            self.w2[i] = integrate.simpson(Opgw_int, f_int) * math.log(10)
            
            #fx = np.arange(f_int.min(), f_int.max(), .01)
            #spline_gw = interpolate.InterpolatedUnivariateSpline(f_int, Ogw_int-Oj_int)
            #spline_pgw = interpolate.InterpolatedUnivariateSpline(f_int, Opgw_int)   # Opgw can be negative
            #self.g2[i] = integrate.simpson(spline_gw(fx), fx) * math.log(10)         
            #self.w2[i] = integrate.simpson(spline_pgw(fx), fx) * math.log(10)