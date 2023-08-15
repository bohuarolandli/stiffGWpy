# stiffGWpy
Python package that computes the self-consistent expansion history of &Lambda;CDM+stiff component+primordial SGWB

Author: Bohua Li, rolandliholmes@gmail.com

Description
-------------
The code computes the expansion history of the extended &Lambda;CDM model 
with (1) the stochastic gravitational-wave background (SGWB) from inflation 
and (2) an additional stiff component. The presence of a possible early stiff era
may cause amplification of the primordial SGWB relative to that in the base &Lambda;CDM model.
For given cosmological parameters, the coupled dynamical system of the Friedmann equation 
and the tensor wave equation is solved iteratively, which accounts for the backreaction 
from the stiff-amplified SGWB on the background expansion history.


Dependencies
--------------------
Python packages: numpy, scipy


Basic Usage
-----------------------------------
Two classes, 'LCDM_SN' and 'LCDM_SG', are provided for the '&Lambda;CDM+stiff+const &Delta;N_eff' 
and '&Lambda;CDM+stiff+SGWB' models, respectively. The latter is a derived class of the former 
and uses the former model to mimic its expansion history.

After creating and initializing an LCDM_SG instance, one may calculate the coupled background evolution 
using its 'SGWB_iter' method. 


### Input ###

The free/base parameters of both types of models are: 

'Omega_bh2',  'Omega_ch2',  'H0',  'DN_eff',  'A_s',  'r', 'n_t', 'cr', 'T_re', 'DN_re', 'kappa10'.

    - [H0] = km s^-1 Mpc^-1, [T_re] = GeV.
    - Set cr > 0 if the consistency relation is assumed, otherwise set cr <= 0 and provide (n_t, DN_re).
    - DN_re is the number of e-folds from the end of inflation to the end of reheating, 
      assuming a^{-3} matter-like evolution.
    - kappa10 := rho_stiff/rho_photon at 10 MeV.

There are three ways to initialize a model with desired base parameters:  
<pre>
    1. input a yaml file,    2. input a dictionary,  
    3. specify the parameters by keyword arguments.  
</pre>
They will be stored in the 'obj_name.cosmo_param' attribute as a dictionary and can be modified later on.  

Derived parameters of the model are stored in the 'obj_name.derived_param' property as a dictionary.  

An example run is as follows.

```
from stiff_SGWB import LCDM_SG as sg

model = sg(r = 1e-2,
           cr = 1,
           T_re = 2e3,
           kappa10 = 1e-2,
          )

model.cosmo_param['H0'] = 68

model.SGWB_iter()
```

### Output ###

Output attributes after successfully running SGWB_iter():
    
- f:             sampled frequencies of the SGWB today, log10(f/Hz) 
- log10OmegaGW:  present-day energy spectrum of the primordial SGWB, log10(Omega_GW(f))
- f_grid:        Uniformly-spaced frequency grid for training emulator
- log10OmegaGW_grid:  log10(Omega_GW(f)) at the uniform-spaced frequencies
- kappa_r:       Total extra radiation (e.g., SGWB) parameterized as kappa_rad(T_i) to be passed to AlterBBN
- DN_eff_orig:   original value of &Delta;N_eff from the input when the run is successful, otherwise set to None.
<br/>

- N:             = ln a, number of e-foldings, the time coordinate for variables below (N=0 is the present)
- sigma:         = 1+w, which characterizes the expansion history 
- f_hor:         comoving frequency of the current mode that fills the horizon, log10(f_hor/Hz), f_hor = aH/(2*pi)
- hubble:        evolution of the Hubble parameter, log10(H/s^-1). 
- DN_gw:         evolution of &Delta;N_eff due to the primordial SGWB. 


Other attributes:

- SGWB_converge: True if SGWB_iter() successfully converges, False if not (abort when the total &Delta;N_eff > 5).


\
Other details of both classes and all methods can be found in relevant docstrings. 
