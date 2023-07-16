import numpy as np
import os, sys, math
from scipy import interpolate
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const
import astropy.units as u
from functions import int_FD

#######   Constants. Do not change.   #######

# Base

kB = const.k_B.to(u.eV/u.K).value          # eV K^-1
hbar = const.hbar.to(u.eV*u.s).value       # eV s
c = const.c.to(u.cm/u.s).value             # cm s^-1
eV = (u.eV/(const.c**2)).to(u.g).value     # g*c^2
GN = const.G.value*1e3                     # cm^3 g^-1 s^-2
parsec = u.parsec.to(u.cm)                 # cm
yr = u.yr.to(u.s)                          # seconds in a Julian year

# Derived

rc = math.pi**2/(15*(hbar*c)**3)           # eV^-3 cm^-3, radiation constant = pi^2/(15*hbar^3*c^3)
m_Pl = math.sqrt(hbar*c**3/(GN*eV))*1e-9   # GeV/c^2, Planck mass = sqrt(hbar*c/G)
M_Pl = m_Pl/math.sqrt(8*math.pi)           # GeV/c^2, reduced Planck mass
f_piv = 0.05 *c/(2*math.pi*1e6*parsec)     # s^-1, CMB pivot scale = 0.05 Mpc^-1


#######   Cosmological Parameters   #######

# Base

TCMB = cosmo.Tcmb0.value  # K
TCMB_GeV = 1e-9*kB*TCMB   # GeV

m_nu = 60                 # meV, for a single massive neutrino eigenstate

Neff0 = 3.044
tau_n = 878.4             # s, neutron lifetime from PDG 2022

# Derived

Omega_ph2 = rc * (kB*TCMB)**4 * (8*math.pi*GN*eV/3) *(0.1*parsec)**2         # photons
Omega_nh2 = Omega_ph2 * 7/8 * (4/11)**(4/3) * Neff0                          # relativistic SM neutrinos
Omega_orh2 = Omega_ph2 + Omega_nh2                                           # SM radiation after e+e- annihilation

Tnu = TCMB * (4/11)**(1/3)                # effective thermal temperature shared by all neutrinos today
nu_today = m_nu*1e-3 / (kB*Tnu)
[rho_nu0, p_nu0] = int_FD(nu_today)
Omega_mnuh2 = Omega_nh2/3 * rho_nu0       # the massive neutrino eigenstate


######      Thermal history      ######

# I. Neutrino decoupling regime, 20 keV ~- 10 MeV. From fortepiano

fpdata = np.loadtxt(os.path.dirname(__file__) + '/benchmark_th.dat')
T_fp = fpdata[:,2]            # GeV
z_fp = fpdata[:,1]; N_fp = fpdata[:,4]; N_fin_fp = N_fp[-1]
rho_fp = fpdata[:,5]; p_fp = fpdata[:,6]

spl_T_z_fp = interpolate.CubicSpline(np.log10(np.flip(T_fp)), np.flip(z_fp)); z_fp_10 = spl_T_z_fp(-2)
spl_T_rho_fp = interpolate.CubicSpline(np.log10(np.flip(T_fp)), np.flip(rho_fp))

spl_T_N_fp = interpolate.CubicSpline(np.log10(np.flip(T_fp)), np.flip(N_fp))
N_10 = spl_T_N_fp(-2); a_10 = np.exp(-N_10)
coeff_fp = Omega_ph2 / (np.pi**2/15 * z_fp[-1]**4)

spl_N_rho_fp = interpolate.CubicSpline(np.flip(N_fp), np.flip(rho_fp))
spl_N_p_fp = interpolate.CubicSpline(np.flip(N_fp), np.flip(p_fp))

# II. Pre-neutrino-decoupling, 10 MeV ~- 10^6 GeV, SM in thermal equilibrium. From Saikawa & Shirai 2020

g_sm = np.loadtxt(os.path.dirname(__file__) + '/eos2020.dat')
T_sm = g_sm[:,0]              # GeV
gs = g_sm[:,1]; g = g_sm[:,2]

spl = interpolate.CubicSpline(np.log10(T_sm[1:]), g[1:])
spl_s = interpolate.CubicSpline(np.log10(T_sm[1:]), gs[1:]); gs_10 = spl_s(-2)

# Lookback number of e-folds, N = -ln(a), assuming entropy conservation
N_sm = np.log(T_sm[1:]/TCMB_GeV) + np.log(gs[1:]/gs[0])/3
N_max = N_sm[-1]; T_max = T_sm[-1]

g_max = g[-1]; gs_fin = gs[1]; gs_max = gs[-1]
Omega_oreh2 = Omega_ph2 * g_max/2 * (gs_fin/gs_max)**(4/3)    # SM radiation before any SM phase transition

z3 = gs[0]*np.reciprocal(gs[1:])               # z^3 := (a*T_photon/T_photon,0)^3 = g_*s,0/g_*s
spline_N_z3 = interpolate.CubicSpline(N_sm, z3, bc_type="natural")
spline_N_g = interpolate.CubicSpline(N_sm, g[1:], bc_type="natural")


#####    Gravitational-wave data    #####

f_yr = 1/yr     # Hz, reference frequency used for PTA
f_LIGO = 25     # Hz, reference frequency used by LIGO-Virgo

#A_NG12 = log10(1.92e-15); A_NG12_upper = log10(2.67e-15); A_NG12_lower = log10(1.37e-15); alpha_NG12 = -2/3; 

alpha_NG12 = -.5            # Spectral index for stiff-amplified SGWB
A_NG12_upper = -14.417 
A_NG12_lower = -14.756 
f_NG12 = np.array((-2, 0, 1))                                  # log10(f/f_yr)
hc_NG12 = A_NG12_lower + alpha_NG12*f_NG12                     # 95% lower limit of log10(h_c) from NANOGrav 12.5yr data

Ogw_LIGO3 = 6.6e-9     #  95% upper limit of Omega_GW from LIGO-Virgo O3, marginalized over spectral index


#####     BBN observational data, obsolete    #####

Neff_A = (2.86 + 0.57)*Neff0/3      # 95% upper bound, BBN, Aver+ 2015 + Cooke+ 2014
Neff_I = 3.58 + 0.40                # 95% upper bound, BBN, Izotov+ 2014 + Cooke+ 2014

Neff_l = 2.99 + 0.43                # 95% upper bound, CMB+BAO+Y_P(Aver+15), low-z acoustic scales + Silk damping scale
Neff_lnp = 2.97 + 0.58              # same as above, CMB+BAO, without Y_P

T_np = 1.293              # MeV   neutron/proton ratio freeze-out
N_np = math.log(T_np*1e6/kB/TCMB)
T_D = 1/14                # MeV   Deuterium synthesis
N_D = math.log(T_D*1e6/kB/TCMB)
N_BBN = [18.2, 23.3]      # ~[0.02 3] MeV  to safely include the whole BBN process