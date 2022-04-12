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

The free/base parameters of both types of models are: 'Omega_bh2', 'Omega_ch2', 'H0', 'DN_eff', 'A_s', 'r', 'T_re', 'T_sr',  
where [H0] = km s^-1 Mpc^-1, [T_re] = GeV, [T_sr] = GeV.

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
           T_re = 2e3,
           T_sr = 1e-2,
          )

model.cosmo_param['H0'] = 68

model.SGWB_iter()
```

### Output ###

Output attributes after successfully running obj_name.SGWB_iter():
    
- f:             sampled frequencies for SGWB calculation, log10(f/Hz) 
- Ogw_today:     present-day energy spectrum of the primordial SGWB
- Nv:            number of e-foldings since the end of inflation
- Sv:            = 1+w, which characterizes the expansion history 
- fv:            frequency of the mode that fills the Hubble radius, log10(fv/Hz)
- hubble:        evolution of the Hubble parameter, log10(H /s^-1). Last element is the present-day value.
- stiff_to_photon_MeV: ratio of the stiff component energy density to that of photons at T = 1 MeV, to be passed to AlterBBN
- DN_eff_orig:   original value of &Delta;N_eff from the input if the run is successful, otherwise set to None
- DN_gw:         evolution of &Delta;N_eff due to the primordial SGWB. Last element is the present-day value.
- SGWB_converge: True if SGWB_iter() is run and successfully converged, False if not (when the total &Delta;N_eff > 5, excluded)


\
Other details of both classes and all methods can be found in relevant docstrings. 
