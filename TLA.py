from numpy import *
from cosmo import *

def comptonCMB(xe, Tm, rs): 
	# Compton cooling rate
	return (xe/(1 + xe + nHe/nH)) * (TCMB(rs) - Tm)*32*thomsonXSec*stefBoltz*TCMB(rs)**4/(3*me)

def KLyman(rs, omegaM=omegaM, omegaRad=omegaRad, omegaLambda=omegaLambda): 
	# Rate at which Lya-photons cross the line
	return (c/lyaFreq)**3/(8*pi*hubblerates(rs, H0, omegaM, omegaRad, omegaLambda))

def alphae(Tm): 
	# Case-B recombination coefficient
	return 1e-13 * (4.309 * (1.16405*Tm)**(-0.6166))/(1 + 0.6703*(1.16405*Tm)**0.5300)

def betae(Tr):
	# Case-B photoionization coefficient
	thermlambda = c*(2*pi*hbar)/sqrt(2*pi*(mp*me/(me+mp))*Tr)
	return alphae(Tr) * exp(-(rydberg/4)/Tr)/(thermlambda**3) 

def CPeebles(xe,rs):
	# Peebles C-factor 
	num = Lambda2s*(1-xe) + 1/(KLyman(rs) * nH * rs**3)
	den = Lambda2s*(1-xe) + 1/(KLyman(rs) * nH * rs**3) + betae(TCMB(rs))*(1-xe)
	return num/den

def getTLADE(fz, injRate):

	def TLADE(var, rs):

		def xe(y): 
			return 0.5 + 0.5*tanh(y)

		Tm, y = var

		# dvardz = ([
		# 	(2*Tm/rs - 
		# 	dtdz(rs)*(comptonCMB(xe(y), Tm, rs))),
		# 	(2*cosh(y)**2) * dtdz(rs) * (CPeebles(xe(y),rs)*
		# 		(alphae(Tm)*xe(y)**2*nH*rs**3 - 
		# 			betae(TCMB(rs))*(1-xe(y))*exp(-lyaEng/Tm)))])

	
		dvardz = ([
			(2*Tm/rs - 
			dtdz(rs)*(comptonCMB(xe(y), Tm, rs) + 
				1/(1 + xe(y) + nHe/nH)*2/(3*nH*rs**3)*fz['Heat'](rs)*injRate(rs))),
			(2*cosh(y)**2) * dtdz(rs) * (CPeebles(xe(y),rs)*
				(alphae(Tm)*xe(y)**2*nH*rs**3 - 
					betae(TCMB(rs))*(1-xe(y))*exp(-lyaEng/Tm)) -
				fz['HIon'](rs)*injRate(rs)/(13.6*nH*rs**3) - 
				(1 - CPeebles(xe(y),rs))*fz['HLya'](rs)*injRate(rs)/(lyaEng*nH*rs**3)
				)])
		
		return dvardz

	return TLADE

