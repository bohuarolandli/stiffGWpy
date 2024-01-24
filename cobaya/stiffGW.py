from cobaya.theory import Theory
from cobaya.tools import load_module
from cobaya.log import LoggedError, get_logger
from multiprocessing import freeze_support
import numpy as np
import astropy.units as u
from scipy import interpolate
import sys, os, math
from mpi4py import MPI
#from pathlib import Path

class stiffGW(Theory):
    speed = 0.1
    params = {'Delta_Neff_GW': {'derived': True, 'latex': '\Delta N_\mathrm{eff,GW}'},
              'Delta_Neff': {'derived': True, 'latex': '\Delta N_\mathrm{eff,tot}'},
              'log10hc_prim_fyr': {'derived': True, 'latex': '\log_{10}h_{c,\mathrm{prim}}'},
              'f_end': {'derived': True, 'latex': 'f_\mathrm{end}'},
             }
    
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
                'A_s': None, 'r': None, 'n_t': None, 'cr': None, 
                'T_re': None, 'DN_re': None, 'kappa10': None}

#    def must_provide(self, **requirements):
#        if 'A' in requirements:
#            # e.g. calculating A requires B computed using same kmax (default 10)
#            return {'B': {'kmax': requirements['A'].get('kmax', 10)}}
        
    def get_can_provide(self):
        return ['f', 'omGW_stiff', 'hubble', 'kappa_s', 'kappa_r',]

    def get_can_provide_params(self):
        return ['Delta_Neff_GW', 'Delta_Neff', 'log10hc_prim_fyr', 'f_end',]

    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        The 'Theory.calculate()' method takes a dictionary 'params_values_dict' 
        of the parameter values as keyword arguments and saves all needed results 
        in the 'state' dictionary (which is cached and reused as needed).        
        """
        
        # Set parameters
        self.stiffGW_model.reset()
        #args = {p: v for p, v in params_values_dict.items()}
        #self.log.debug("Setting parameters: %r", args)
        #print(self.rank, ": ", params_values_dict)
        for key in self.stiffGW_model.cosmo_param:
            if key in params_values_dict:
                self.stiffGW_model.cosmo_param[key] = params_values_dict[key]
        
        # Compute!
        sys.path.append(os.path.abspath('../'))
        if __name__ == 'stiffGW':
            self.stiffGW_model.SGWB_iter()

        
        if self.stiffGW_model.SGWB_converge:
            state['f'] = self.stiffGW_model.f                                    # Output frequency in log10(f/Hz)
            state['omGW_stiff'] = self.stiffGW_model.log10OmegaGW                # log10(Omega_GW(f))
            state['hubble'] = self.stiffGW_model.derived_param['H_0']            # H_0 in units of s^-1
            state['kappa_s'] = self.stiffGW_model.derived_param['kappa_s']       # kappa_stiff(T_i) for AlterBBN
            state['kappa_r'] = self.stiffGW_model.kappa_r                        # kappa_rad(T_i) for AlterBBN, related to Delta_Neff
            
            if want_derived:
                yr = u.yr.to(u.s); log10f_yr = -math.log10(yr)
                if self.stiffGW_model.f[0] >= log10f_yr:
                    f_t = np.flip(state['f']); Ogw_t = np.flip(state['omGW_stiff'])
                    spec_prim = interpolate.CubicSpline(f_t[f_t>-13], Ogw_t[f_t>-13])
                    omGW_stiff_fyr = spec_prim(log10f_yr)    # log10(Omega_GW(f_yr))
                else:    
                    omGW_stiff_fyr = -100.
                
                state['derived'] = {'Delta_Neff_GW': self.stiffGW_model.DN_gw[-1],            # Delta N_eff due to the primordial SGWB today
                                    'Delta_Neff': self.stiffGW_model.cosmo_param['DN_eff'],   # Total Delta N_eff after GW calculation
                                    'log10hc_prim_fyr': omGW_stiff_fyr/2 + math.log10(math.sqrt(1.5)*state['hubble']/math.pi)-log10f_yr,
                                    # log10(h_c(f_yr)) of the primordial SGWB
                                    'f_end': np.power(10., self.stiffGW_model.f[0]),          # Hz, UV cutoff frequency
                                   }
        else:
            #self.log.debug("SGWB calculation not converged, mostly due to total N_eff too large. Assigning 0 likelihood and going on.")
            return False

        
#    def get_A(self, normalization=1):
#        return self.current_state['A'] * normalization