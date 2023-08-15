from cobaya.theory import Theory
from cobaya.tools import load_module
from cobaya.log import LoggedError, get_logger
import numpy as np
import sys, os
import uuid
from mpi4py import MPI
#from pathlib import Path

class alterbbn(Theory):
    speed = 1
    params = {'Yp': {'derived': True, 'latex': 'Y_\mathrm{p}'},
              'DH': {'derived': True, 'latex': '[\mathrm{D}/\mathrm{H}]'},
             }
    
    def initialize(self):
        """called from __init__ to initialize"""
        self.model_uid = str(uuid.uuid1())      
        alterbbn_path = os.path.dirname(__file__) + '/../../../alterbbn_v2.2/'
#        alterbbn_path = '/work/bohuali/alterbbn_v2.2/'
        alter_stiff = load_module('alter_stiff', path=alterbbn_path)
        self.BBNstiff_model = alter_stiff.BBN_stiff(self.model_uid)    
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
        self.BBNstiff_model.cleanup()

    
    def get_requirements(self):
        """
        Return dictionary of quantities that are always needed by this component 
        and should be calculated by another component or provided by input parameters.
        """
        return {'kappa_s': None, 'kappa_r': None, 'Omega_bh2': None,}

#    def must_provide(self, **requirements):
#        if 'A' in requirements:
#            # e.g. calculating A requires B computed using same kmax (default 10)
#            return {'B': {'kmax': requirements['A'].get('kmax', 10)}}
        
    def get_can_provide(self):
        return ['Y_p', 'D_to_H']

    def get_can_provide_params(self):
        return ['Yp', 'DH',]

    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        The 'Theory.calculate()' method takes a dictionary 'params_values_dict' 
        of the parameter values as keyword arguments and saves all needed results 
        in the 'state' dictionary (which is cached and reused as needed).        
        """
        
        # Set parameters
        args = {p: v for p, v in params_values_dict.items()}
        self.log.debug("Setting parameters: %r", args)

        try:
            kappa_s = self.provider.get_result('kappa_s')
            kappa_r = self.provider.get_result('kappa_r') 
            self.BBNstiff_model.calculateAbundances(kappa_s, kappa_r, args['Omega_bh2'], 
                                                   failsafe = 3, fast = True)
        except AttributeError as e:
            self.BBNstiff_model.calculateAbundances(args['kappa_s'], args['kappa_r'], args['Omega_bh2'],
                                                   failsafe = 3, fast = True)
        
        state['Y_p'] = self.BBNstiff_model.abundances[0]        # Y_p
        state['D_to_H'] = self.BBNstiff_model.abundances[1]     # [D/H]
        
        if want_derived:
            state['derived'] = {'Yp': state['Y_p'], 
                                'DH': state['D_to_H'],}
        
        self.close()
        
  
#    def get_Y_p(self, normalization=1):
#        return self.current_state['Y_p'] * normalization