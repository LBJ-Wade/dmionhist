from numpy import *

# Fundamental constants
mp          = 0.938e9                     # proton mass in eV
me          = 510998.9                    # electron mass in eV
hbar        = 6.58211951e-16              # hbar in eV s
c           = 299792458e2                 # speed of light in cm/s
kB          = 8.6173324e-5                # Boltzmann constant in eV/K
alpha       = 1/137.035999139             # fine structure constant

# Atomic and optical physics

thomsonXSec = 6.652458734e-25                             # Thomson scattering cross section             
stefBoltz   = pi**2/(60*(hbar**3)*(c**2))        # Stefan-Boltzmann constant
rydberg     = 13.60569253                                 # 1 Rydberg
lyaEng      = rydberg*3/4                                 # Lyman-alpha energy
lyaFreq   = lyaEng/(2*pi*hbar)   # Lyman-alpha frequency
Lambda2s    = 8.23                                        # 2s->1s decay lifetime

# Hubble

h    = 0.6727
H0   = 100*h*3.241e-20                    # Hubble constant in s

# Omegas

omegaM      = 0.3156 
omegaRad    = 8e-5
omegaLambda = 0.6844
omegaB      = 0.0225/(h**2)
omegaDM     = 0.1198/(h**2)

# Densities

rhoCrit     = 1.05375e4*(h**2)            # in eV/cm^3
rhoDM       = rhoCrit*omegaDM
rhoB        = rhoCrit*omegaB
nB          = rhoB/mp
YHe         = 0.245                       # Helium mass abundance from the PDG
nH          = (1-YHe)*nB
nHe         = (YHe/4)*nB
nA          = nH + nHe

def hubblerates(rs, H0=H0, omegaM=omegaM, omegaRad=omegaRad, omegaLambda=omegaLambda): 
	return H0*sqrt(omegaRad*rs**4 + omegaM*rs**3 + omegaLambda)

def dtdz(rs, H0=H0, omegaM=omegaM, omegaRad=omegaRad, omegaLambda=omegaLambda):

	return 1/(rs*hubblerates(rs, H0, omegaM, omegaRad, omegaLambda))

def TCMB(rs): 

	return 0.235e-3 * rs