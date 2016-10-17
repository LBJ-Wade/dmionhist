from numpy import *
from physics import *
from scipy.interpolate import interp1d

def sigma1DNorm(chan,rs):
	if chan == 'sWave':
		return 1.
	elif chan == 'pWave':	
		return rs*1e-11/sqrt(100.)

def getstructform(chan,structFormType):

	if chan == 'sWave':
		structFormFName = (
			{'rho_eff_Einasto_subs':'structFormData/rho_eff_Einasto_subs.txt',
			 'rho_eff_Einasto_no_subs':'structFormData/rho_eff_Einasto_no_subs.txt',
			 'rho_eff_NFW_subs':'structFormData/rho_eff_NFW_subs_corrected.txt',
			 'rho_eff_NFW_no_subs':'structFormData/rho_eff_NFW_no_subs.txt'}
			)
	elif chan == 'pWave':
		structFormFName = (
			{'rho_eff_Einasto_subs':'structFormData/rho_eff_Einasto_subs_pwave.txt',
			 'rho_eff_NFW_subs':'structFormData/rho_eff_NFW_subs_pwave.txt'
			}
			)

	a = loadtxt(structFormFName[structFormType])

	# Convert to 1 + z
	rhoEff = vstack((a[:,0],a[:,2]))
	rhoEff[0,:] += 1

	# Convert the density to eV, and also the fact that the densities are normalized to rhoM instead of rhoDM
	rhoEff[1,:] *= 1e9*rhoDM/1.50389e3

	interpRhoEff = interp1d(rhoEff[0,:],rhoEff[1,:])

	maxrsInterp = 52 

	def structform(rs):
		if isinstance(rs,ndarray):
			rho = zeros(rs.size)
			interpRhoInd = where(rs <= maxrsInterp)
			anaRhoInd = where(rs > maxrsInterp)
			if chan == 'sWave':
				rho[interpRhoInd] = interpRhoEff(rs[interpRhoInd])
				rho[anaRhoInd]    = rhoDM*rs[anaRhoInd]**3
			elif chan == 'pWave':
				rho[interpRhoInd] = interpRhoEff(rs[interpRhoInd])*rs[interpRhoInd]
				rho[anaRhoInd]    = rhoDM*rs[anaRhoInd]**3*sigma1DNorm(chan,rs[anaRhoInd])
		elif isinstance(rs,float) or isinstance(rs,int):
			if rs <= maxrsInterp: 
				if chan == 'sWave':
					rho = interpRhoEff(rs)
				elif chan == 'pWave':
					rho = interpRhoEff(rs)*rs
			else: 
				if chan == 'sWave':
					rho = rhoDM*rs**3
				if chan == 'pWave':
					rho = rhoDM*rs**3*sigma1DNorm(chan,rs)
		return rho

	return structform 		

