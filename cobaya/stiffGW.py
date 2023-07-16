from cobaya.theory import Theory
from cobaya.tools import load_module
from cobaya.log import LoggedError, get_logger
import numpy as np
import sys, os
from mpi4py import MPI
#from pathlib import Path

class stiffGW(Theory):
    
    def initialize(self):
        """called from __init__ to initialize"""
        stiff_SGWB_path = os.path.dirname(__file__) + '/../'
        stiff_SGWB = load_module('stiff_SGWB', path=stiff_SGWB_path)
        self.stiffGW_model = stiff_SGWB.LCDM_SG()
        #self.comm = MPI.COMM_WORLD
        #self.rank = self.comm.Get_rank()
        self.log.info("Initialized!")

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using theory.Provider class instance.
        It is used to return any dependencies (requirements of this theory) 
        via methods like "provider.get_X()" and "provider.get_param(‘Y’)".
        """
        self.provider = provider
        
    def close(self):
        pass

    
    def get_requirements(self):
        """
        Return dictionary of quantities that are always needed by this component 
        and should be calculated by another component or provided by input parameters.
        """
        return {'Omega_bh2': None, 'Omega_ch2': None, 'H0': None, 'DN_eff': None, 
                'A_s': None, 'r': None, 'n_t': None, 'f_end': None, 'cr': None, 
                'T_re': None, 'kappa10': None}

#    def must_provide(self, **requirements):
#        if 'A' in requirements:
#            # e.g. calculating A requires B computed using same kmax (default 10)
#            return {'B': {'kmax': requirements['A'].get('kmax', 10)}}
        
    def get_can_provide(self):
        return ['f', 'omGW_stiff', 'hubble',]

    def get_can_provide_params(self):
        return ['Delta_Neff_GW', ]

    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        The 'Theory.calculate()' method takes a dictionary 'params_values_dict' 
        of the parameter values as keyword arguments and saves all needed results 
        in the 'state' dictionary (which is cached and reused as needed).        
        """
        
        # Set parameters
        self.stiffGW_model.reset()
        args = {p: v for p, v in params_values_dict.items()}
        self.log.debug("Setting parameters: %r", args)
        #print(self.rank, ": ", params_values_dict)
        for key in self.stiffGW_model.cosmo_param:
            if key in params_values_dict:
                self.stiffGW_model.cosmo_param[key] = params_values_dict[key]
        
        # Compute!
        self.stiffGW_model.SGWB_iter()
        
        if self.stiffGW_model.SGWB_converge:
            state['f'] = self.stiffGW_model.f                                              # Output frequency in log10(f/Hz)
            state['omGW_stiff'] = np.log10(self.stiffGW_model.Ogw_today - self.stiffGW_model.Oj_today)  # log10(Omega_GW(f))
            # Ignoring the negative super-horizon contribution from Omega_j, for the moment...
            state['hubble'] = self.stiffGW_model.derived_param['H_0']         # H_0 in units of s^-1    
            
            if want_derived:
                state['derived'] = {'Delta_Neff_GW': self.stiffGW_model.DN_gw[-1], # Delta N_eff due to the primordial SGWB today
                                    }         
        else:
            self.log.debug("SGWB calculation not converged, mostly due to too much stiff amplification. "
                           "Assigning 0 likelihood and going on.")
            return False

        
#    def get_A(self, normalization=1):
#        return self.current_state['A'] * normalization