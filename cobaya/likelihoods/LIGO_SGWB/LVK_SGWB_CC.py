from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError, get_logger
import numpy as np
import os
from scipy import interpolate


class LVK_SGWB_CC(Likelihood):
    
    def initialize(self):
        """
        Initializes the class (called from __init__, before other initializations).
        Prepare any computation, importing any necessary code, files, etc.

        e.g. here we load some SGWB data file, with default CC_file set in the .yaml,
        or overridden when running Cobaya.
        """
        self.data = np.loadtxt(self.CC_file)
    
    def close(self):
        pass
        
    def get_requirements(self):
        """
        return dictionary specifying quantities that are always needed and calculated by a theory code
        """
        return {'f': None, 'omGW_stiff': None,}

    
    def logp(self, _derived=None, **params_values):
        """
        The default implementation of the Likelihood class does the calculation in this 'logp()' function, 
        which is called by 'Likelihood.calculate()' to save the log likelihood into "state['logp']" 
        (the latter may be more convenient if you also need to calculate some derived parameters).
        
        'logp()' can take a dictionary (as keyword arguments) of nuisance parameter values, 'params_values', 
        (if there is any), and returns a log-likelihood.
        """
        f_theory = self.provider.get_result('f'); f_theory = np.flip(f_theory)
        Ogw_theory = self.provider.get_result('omGW_stiff'); Ogw_theory = np.flip(Ogw_theory)
        
        #if _derived is not None:
        #    _derived['N_eff'] = self.provider.get_param('Delta_Neff_GW') + 3.044
        
        return self.log_likelihood(f_theory, Ogw_theory, **params_values)

    
    def log_likelihood(self, f_theory, Ogw_theory, **data_params):
        """
        where the calculation is actually done, independently of Cobaya
        Here f_theory must be increasing.
        """
        f_LVK = np.log10(self.data[:,0])
        Cf_LVK = self.data[:,1]
        sigma_LVK = self.data[:,2]
        
        Ogw_Model = np.zeros_like(Cf_LVK)
        if f_theory[-1]>=f_LVK[0]:   # Calculate theoretical Omega_GW ONLY for LIGO frequency bins in its range, i.e., <= f_end
            f_t = f_theory[(f_theory >= -5)]; Ogw_t = Ogw_theory[(f_theory >= -5)]
            #print(f_t, Ogw_t)
            spec = interpolate.interp1d(f_t, Ogw_t, kind='cubic')
            
            cond = (f_LVK<=f_theory[-1])
            Ogw_Model[cond] = np.power(10., spec(f_LVK[cond]))  
            
        #print(Ogw_Model)
             
        chi2_array = np.square(np.divide((Cf_LVK-Ogw_Model), sigma_LVK))
        chi2 = sum(chi2_array)
        
        return -chi2 / 2