from cobaya.theory import Theory
from cobaya.tools import load_module
from cobaya.log import LoggedError, get_logger
import numpy as np
import sys, os
import uuid
from mpi4py import MPI
#from pathlib import Path

class par_stiff(Theory):
    
    def initialize(self):
        """called from __init__ to initialize"""
        self.path = '/Users/bohuali/Documents/bbn_codes/parthenope3.0/'
        self.model_uid = str(uuid.uuid1())
        self.input_card = 'input_' + self.model_uid + '.card'
        self.parthenope_input = 'in_' + self.model_uid
            
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
        os.system('rm *'+self.model_uid+'*')

    
    def get_requirements(self):
        """
        Return dictionary of quantities that are always needed by this component 
        and should be calculated by another component or provided by input parameters.
        """
        return ['tau_n', 'Omega_bh2', 'dnnu', 'kappa',]

#    def must_provide(self, **requirements):
#        if 'A' in requirements:
#            # e.g. calculating A requires B computed using same kmax (default 10)
#            return {'B': {'kmax': requirements['A'].get('kmax', 10)}}
        
    def get_can_provide(self):
        return ['Y_p', 'D_to_H']

#    def get_can_provide_params(self):
#        return ['Yp', 'DH',]

    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        The 'Theory.calculate()' method takes a dictionary 'params_values_dict' 
        of the parameter values as keyword arguments and saves all needed results 
        in the 'state' dictionary (which is cached and reused as needed).        
        """
        
        # Set parameters
        args = {p: v for p, v in params_values_dict.items()}
        self.log.debug("Setting parameters: %r", args)

        with open(self.path + 'input.card', 'r') as f:
            contents = f.readlines()
        
        eta10 = 273.3036 * params_values_dict['Omega_bh2'] * (1 + 7.16958e-3*.25)
        contents.insert(1, 'TAU ' + str(params_values_dict['tau_n'])+'\n')
        contents.insert(2, 'ETA10 ' + str(eta10)+'\n')
        contents.insert(3, 'DNNU ' + str(params_values_dict['dnnu'])+'\n')
        contents.insert(4, 'KAPPA ' + str(params_values_dict['kappa'])+'\n')
        contents.insert(5, 'FILES ' + 'abun_'+self.model_uid + ' evo_'+self.model_uid + ' info_'+self.model_uid+'\n')

        with open(self.input_card, 'w') as f:
            f.writelines(contents)
            
        # Compute!
        with open(self.parthenope_input, 'w') as f:
            contents = ['c\n',]
            contents.append(self.input_card)
            f.writelines(contents)
        
        os.system(self.path + 'parthenope3.0 < ' + self.parthenope_input + ' > parthenope.log')
        
        with open('abun_'+self.model_uid, 'r') as f:
            lines = f.readlines()
            result = lines[-1].split()
        
        state['Y_p'] = D_to_E(result[-1])       # Y_p
        state['D_to_H'] = D_to_E(result[-2])    # [D/H]
        
#        if want_derived:
#            state['derived'] = {'Yp': state['Y_p'], 
#                                'DH': state['D_to_H'],}
        
        self.close()
        
 
  
#    def get_Y_p(self, normalization=1):
#        return self.current_state['Y_p'] * normalization


def D_to_E(string):
    val = string.split('D')
    return float(val[0])*10**int(val[1])