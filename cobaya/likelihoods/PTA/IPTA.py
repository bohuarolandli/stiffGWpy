from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError, get_logger
import numpy as np
import os, math
import astropy.units as u
from scipy import interpolate
from scipy.stats import gaussian_kde as kde

class IPTA(Likelihood):
    
    def initialize(self):
        """
        Initializes the class (called from __init__, before other initializations).
        Prepare any computation, importing any necessary code, files, etc.
        """
        self.data = np.loadtxt(self.sample_file)
        self.freqs = np.loadtxt(self.freq_file)
    
    def close(self):
        pass
    
            
    def get_requirements(self):
        """
        return dictionary specifying quantities that are always needed and calculated by a theory code
        """
        return {'f': None, 'omGW_stiff': None, 'hubble': None}
    
    
    def logp(self, _derived=None, **params_values):
        """
        The default implementation of the Likelihood class does the calculation in this 'logp()' function, 
        which is called by 'Likelihood.calculate()' to save the log likelihood into "state['logp']" 
        (the latter may be more convenient if you also need to calculate some derived parameters).
        
        'logp()' can take a dictionary (as keyword arguments) of nuisance parameter values, 'params_values', 
        (if there is any), and returns a log-likelihood.
        """
        f_theory = self.provider.get_result('f'); f_theory = np.flip(f_theory)                  # log10(f/Hz)
        Ogw_theory = self.provider.get_result('omGW_stiff'); Ogw_theory = np.flip(Ogw_theory)   # log10(Omega_GW(f))
        H_0 = self.provider.get_result('hubble')                                                # s^{-1}
        
        return self.log_likelihood(f_theory, Ogw_theory, H_0, **params_values)

    
    def log_likelihood(self, f_theory, Ogw_theory, H_0, **data_params):
        """
        where the calculation is actually done, independently of Cobaya
        Here f_theory must be increasing.
        """
        yr = u.yr.to(u.s)                # s, one Julian year
        T_base = 1/self.freqs[0]         # s, baseline of the PTA data
        
        Nf = self.Nfreqs                 # Number of work frequencies, i = 1 - Nf
        work_freqs = self.freqs[:Nf]     # work frequencies in Hz
        # Caveat: limiting frequency bins may lead to the survival of some stiff-amplified models 
        # which would otherwise be ruled out by high-frequency data (i=Nf+1 - Ntot)

        # Primordial SGWB
        rho_prim = np.zeros_like(work_freqs)
        if f_theory[-1] >= math.log10(work_freqs[0]):
            cond = (f_theory > -13) & (f_theory < -5) 
            f_t = f_theory[cond]; Ogw_t = Ogw_theory[cond]
            spec_prim = interpolate.CubicSpline(f_t, Ogw_t)

            cond = (np.log10(work_freqs)<=f_t[-1])
            # Calculate theoretical Omega_GW ONLY for PTA frequency bins in its range, i.e., <= f_end
            Ogw_prim = spec_prim(np.log10(work_freqs[cond]))        
            rho_prim[cond] = np.divide(np.power(10., Ogw_prim) * H_0**2, 8*np.pi**4 * work_freqs[cond]**5 * T_base)  # s^2
      
        # SGWB from SMBHBs
        log10hc_BH = data_params['A_BBH']      # log10(h_c) at f_yr
        gamma_BH = data_params['gamma_BBH']
        rho_BH = np.power(10., log10hc_BH*2) * np.power(work_freqs*yr, -gamma_BH) * yr**3 / (12*np.pi**2 * T_base)   # s^2
        
        log10rho_Model = np.log10(np.sqrt(rho_prim + rho_BH))    # log10(delay/s) = log10(sqrt(rho/s^2))
        #print(log10rho_Model)      

        
        log10rho = self.data
        log10rho_max = np.max(log10rho, axis=0); log10rho_min = np.min(log10rho, axis=0)
        KDE = {i: kde(log10rho[:,i]) for i in range(Nf)}
        
        logL = 0
        for i in range(Nf):
            if log10rho_Model[i]>log10rho_max[i]:
                logL += KDE[i].logpdf(log10rho_max[i])[0]
            elif log10rho_Model[i]<log10rho_min[i]:
                logL += KDE[i].logpdf(log10rho_min[i])[0]
            else:
                logL += KDE[i].logpdf(log10rho_Model[i])[0]
        
        return logL