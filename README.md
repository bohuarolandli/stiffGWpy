# stiffGWpy
Python package that computes the self-consistent expansion history of Lambda-CDM+stiff component+primordial SGWB

Author: Bohua Li, rolandliholmes@gmail.com

Description
-------------
The code computes the expansion history of the extended Lambda-CDM model 
with (1) the stochastic gravitational-wave background (SGWB) from inflation 
and (2) an additional stiff component. The presence of a possible early stiff era
may cause amplification of the primordial SGWB relative to that in the base Lambda-CDM model.
For given cosmological parameters, the coupled dynamical system of the Friedmann equation 
and the tensor wave equation is solved iteratively, which accounts for the backreaction 
from the stiff-amplified SGWB on the background expansion history.


Dependencies
--------------------
Python packages: numpy, scipy


Basic Usage
-----------------------------------
We provide two classes, 'LCDM_SN' and 'LCDM_SG', for the 'LCDM+stiff+const Delta N_eff' 
and 'LCDM+stiff+SGWB' models, respectively. The latter is a derived class of the former 
and uses the former model to mimic its expansion history.

After creating and initializing an LCDM_SG instance, one may calculate the coupled background evolution 
using its 'SGWB_iter' method. Detailed usage (initialization, input, output, etc.) of both classes and all methods
can be found in relevant docstrings. An example run is as follows.

from stiff_SGWB import LCDM_SG as sg

model = sg(r = 1e-2,
           T_re = 2e3,
           T_sr = 1e-2,
          )

model.cosmo_param['H0'] = 68

model.SGWB_iter()
