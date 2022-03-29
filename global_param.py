import numpy as np
import math
from functions import int_FD

#######   Constants. Do not change.   #######

# Base

kB = 8.6173324e-5     # eV K^-1
hbar = 6.58211928e-16 # eV s
c = 29979245800       # cm s^-1
eV = 1.783e-33        # g*c^2
GN = 6.674e-8         # cm^3 g^-1 s^-2
parsec = 3.0857e18    # cm
yr = 31557600         # seconds in a Julian year

# Derived

rc = math.pi**2/(15*(hbar*c)**3)         # eV^-3 cm^-3, radiation constant = pi^2/(15*hbar^3*c^3)
m_Pl = math.sqrt(hbar*c**3/(GN*eV))      # eV/c^2, Planck mass = sqrt(hbar*c/G)
M_Pl = m_Pl/math.sqrt(8*math.pi)         # eV/c^2, reduced Planck mass
f_piv = 0.05 *c/(2*math.pi*1e6*parsec)   # s^-1, CMB pivot scale = 0.05 Mpc^-1


#######   Cosmology Parameters   #######

# Base

TCMB = 2.7255            # K

Neff0 = 3.046
m_nu = 60                # meV, for a single massive neutrino eigenstate

T_np = 1.293             # MeV   neutron/proton ratio freeze-out
N_np = math.log(T_np*1e6/kB/TCMB)
tau_n = 880.2            # neutron lifetime, used in PArthENoPE
T_D = 1/14               # MeV   Deuterium synthesis
N_D = math.log(T_D*1e6/kB/TCMB)
N_BBN = [18.2, 23.3]     # ~[0.02 3] MeV  to safely include the whole BBN process

# Derived

Tnu = TCMB * (4/11)**(1/3)        # thermal temperature shared by all neutrinos, actual T off by (Neff0/3)^(1/4)
nu_today = m_nu*1e-3 / (kB*Tnu)

Omega_ph2 = rc * (kB*TCMB)**4 * (8*math.pi*GN*eV/3) *(0.1*parsec)**2         # photons
Omega_nh2 = Omega_ph2 * 7/8 * (4/11)**(4/3) * Neff0                          # relativistic ordinary neutrinos
Omega_orh2 = Omega_ph2 + Omega_nh2                                           # ordinary radiation
Omega_oreh2 = Omega_ph2 * (4/11)**(1/3) + Omega_nh2                          # ordinary radiation before e+e- annihilation

[rho_nu0, p_nu0] = int_FD(nu_today)
Omega_mnuh2 = Omega_nh2/3 * rho_nu0          # the massive neutrino eigenstate



#####     BBN observational data    #####

Neff_A = (2.86 + 0.57)*Neff0/3      # 95% upper bound, BBN, Aver+ 2015 + Cooke+ 2014
Neff_I = 3.58 + 0.40                # 95% upper bound, BBN, Izotov+ 2014 + Cooke+ 2014

Neff_l = 2.99 + 0.43                # 95% upper bound, CMB+BAO+Y_P(Aver+15), low-z acoustic scales + Silk damping scale
Neff_lnp = 2.97 + 0.58              # same as above, CMB+BAO, without Y_P



#####    Gravitational-wave data    #####

f_yr = 1/yr     # reference frequency used for PTA, in Hz
f_LIGO = 25     # reference frequency used by LIGO-Virgo, in Hz

#A_NG12 = log10(1.92e-15); A_NG12_upper = log10(2.67e-15); A_NG12_lower = log10(1.37e-15); alpha_NG12 = -2/3; 

alpha_NG12 = -.5            # Spectral index for stiff-amplified SGWB
A_NG12_upper = -14.417 
A_NG12_lower = -14.756 
f_NG12 = np.array((-2, 0, 1))                                  # log10(f/f_yr)
hc_NG12 = A_NG12_lower + alpha_NG12*f_NG12                     # 95% lower limit of log10(h_c) from NANOGrav 12.5yr data

Ogw_LIGO3 = 6.6e-9     #  95% upper limit of Omega_GW from LIGO-Virgo O3, marginalized over spectral index