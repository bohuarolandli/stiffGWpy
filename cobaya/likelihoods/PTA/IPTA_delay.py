from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError, get_logger
import numpy as np
import os
from scipy import interpolate
from scipy.stats import gaussian_kde as kde

class IPTA_delay(Likelihood):
    
    def initialize(self):
        """
        Initializes the class (called from __init__, before other initializations).
        Prepare any computation, importing any necessary code, files, etc.
        """
        self.data = np.loadtxt(self.sample_file)
        self.freq = np.log10(np.loadtxt(self.freq_file)[:13])
    
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
        f_theory = self.provider.get_result('f'); f_theory = np.flip(f_theory)
        Ogw_theory = self.provider.get_result('omGW_stiff'); Ogw_theory = np.flip(Ogw_theory)
        H_0 = self.provider.get_result('hubble')
        
        return self.log_likelihood(f_theory, Ogw_theory, H_0, **params_values)

    
    def log_likelihood(self, f_theory, Ogw_theory, H_0, **data_params):
        """
        where the calculation is actually done, independently of Cobaya
        Here f_theory must be increasing.
        """
        IPTA_Ogw_low = self.data - 2*np.log10(H_0)
        KDE = {i: kde(IPTA_Ogw_low[:,i]) for i in range(13)}

        cond = (f_theory >= -11) & (f_theory <= -5) 
        f_t = f_theory[cond]; Ogw_t = Ogw_theory[cond]

        spec = interpolate.interp1d(f_t, Ogw_t, kind='cubic')
        Ogw_Model = spec(self.freq)  
        #print(Ogw_Model)
             
        logL = 0
        for i in range(13):
            logL += KDE[i].logpdf(Ogw_Model[i])[0]      
        
        return logL