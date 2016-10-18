from numpy import *

# Fundamental constants
mp          = 0.938e9                     # proton mass in eV
me          = 510998.9                    # electron mass in eV
hbar        = 6.58211951e-16              # hbar in eV s
c           = 299792458e2                 # speed of light in cm/s
kB          = 8.6173324e-5                # Boltzmann constant in eV/K
alpha       = 1/137.035999139             # fine structure constant
ele         = 1.60217662e-19

# Atomic and optical physics

thomsonXSec = 6.652458734e-25             # Thomson scattering cross section             
stefBoltz   = pi**2/(60*(hbar**3)*(c**2)) # Stefan-Boltzmann constant
rydberg     = 13.60569253                 # 1 Rydberg
lyaEng      = rydberg*3/4                 # Lyman-alpha energy
lyaFreq     = lyaEng/(2*pi*hbar)          # Lyman-alpha frequency
Lambda2s    = 8.23                        # 2s->1s decay width, s^-1
bohrRad     = hbar*c/(me*alpha)           # Bohr radius
eleRad      = bohrRad*alpha**2            # Classical electron radius

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

# Cosmology functions

def hubblerates(rs, H0=H0, omegaM=omegaM, omegaRad=omegaRad, omegaLambda=omegaLambda): 
	return H0*sqrt(omegaRad*rs**4 + omegaM*rs**3 + omegaLambda)

def dtdz(rs, H0=H0, omegaM=omegaM, omegaRad=omegaRad, omegaLambda=omegaLambda):

	return 1/(rs*hubblerates(rs, H0, omegaM, omegaRad, omegaLambda))

def TCMB(rs): 

	return 0.235e-3 * rs

# Atomic Cross-Sections

def photoionxsec(eng, species):

	engThres = {'H0':rydberg, 'He0':24.6, 'He1':4*rydberg}

	indAbove = where(eng > engThres[species])
	xsec = zeros(eng.size)

	if species == 'H0' or species =='He1': 
		eta = zeros(eng.size)
		eta[indAbove] = 1./sqrt(eng[indAbove]/engThres[species] - 1.)
		xsec[indAbove] = (2.**9*pi**2*eleRad**2/(3.*alpha**3)
			* (engThres[species]/eng[indAbove])**4 
			* exp(-4*eta[indAbove]*arctan(1./eta[indAbove]))
			/ (1.-exp(-2*pi*eta[indAbove]))
			)
	elif species == 'He0':
		x = zeros(eng.size)
		y = zeros(eng.size)

		sigma0 = 9.492e2*1e-18      # in cm^2
		E0     = 13.61              # in eV
		ya     = 1.469
		P      = 3.188
		yw     = 2.039
		y0     = 4.434e-1
		y1     = 2.136

		x[indAbove]    = (eng[indAbove]/E0) - y0
		y[indAbove]    = sqrt(x[indAbove]**2 + y1**2)
		xsec[indAbove] = (sigma0*((x[indAbove] - 1)**2 + yw**2) 
			*y[indAbove]**(0.5*P - 5.5)
			*(1 + sqrt(y[indAbove]/ya))**(-P)
			)

	return xsec 

def photoionrate(eng,rs,xH,xe,species=None):
	"""Returns the photoionization rate at a particular redshift, given some ionization history.

	Parameters
	----------
	eng : ndarray
		Energies at which the photoionization rate is to be obtained. 
	rs : float
		Redshift at which the photoionization rate is to be obtained. 
	xH : float
		Ionization fraction n_H+/n_H. 
	xe : float
		Ionization fraction n_e/n_H = n_H+/n_H + n_He+/n_H.
	species : str, optional
		A string that must be one of ``'H0'``, ``'He0'`` or ``'He1'``. Determines which photoionization rate is returned. The default value is ``None``, which returns all of the rates in a dict. 
	
	Returns
	-------
	ionrate : dict
		Returns a dictionary with keys ``'H0'``, ``'He0'`` and ``'He1'``, each with an ndarray of the same length as `eng`.

	"""
	xHe = xe - xH

	ionrate = { ('H0' : photoionxsec(eng,'H0') *nH*(1-xH)     *rs**3*c, 
		        'He0' : photoionxsec(eng,'He0')*(nHe - xHe*nH)*rs**3*c, 
		        'He1' : photoionxsec(eng,'He1')*xHe*nH        *rs**3*c)
	}
	if species is not None:
		return ionrate[species]
	else:
		return ionrate




