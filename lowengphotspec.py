from numpy     import *
from spectrum  import *
from physics   import *
from utilities import *

atoms = ['H0', 'He0', 'He1']

def getionphotspec(lowerBound, xH, xe, dlnz, lowEngPhotSpec, lowEngPhotAtomSpec):
	"""Gets the photoionizing photons from spec, after applying the new cross-sections to photons below lowerbound. May be able to bump this up to the main high-energy code, so it could be obsolete."""

	rs  = lowEngPhotSpec.rs
	engVec = lowEngPhotSpec.eng

	belowBound  = where(engVec <= lowerBound)
	aboveBound  = where(engVec > lowerBound)
	filterAbove = ones(engVec.size)[aboveBound]

	ionRate = {atom: photoionrate(rs, engVec, xH, xe, atom) for atom in atoms}
	tau     = {atom: ionRate[atom]*dlnz/hubblerates(rs) for atom in atoms}
	sumTau  = sum([tau[atom] for atom in atoms], axis=0)

	scatterProb = {atom: divignorebyzero((1. - exp(-sumTau))*tau[atom],sumTau) for atom in atoms}
	
	specBelow = {atom: scatterProb[atom]*lowEngPhotSpec for atom in atoms}
	specAbove = {atom: lowEngPhotAtomSpec[atom] for atom in atoms}

	#return {atom: concatenate(specBelow[atom][belowBound],specAbove[atom][aboveBound]) for atom in atoms}
	return {atom: specBelow[atom] for atom in atoms}

def lowengphotspec(xHArr, xeArr, lowerboundArr, lowengphot, lowengphot_elementgrid):
	"""Runs the low energy photon deposition to get the correct number of photons going into ionization at each redshift, as well as the leftover low energy photon spectrum at each redshift. 


	"""



	# for i in arange(lowengphot.rs): xH, xe, rs, lowerBound, spec, specH0, specHe0, specHe1 in zip(xHArr, xeArr, lowerboundArr, lowengphot.spectrumList, *[atom: lowengphot_elementgrid[atom].spectrumList for atom in atoms]):

	# 	xH = xHArr[i]
	# 	xe = xeArr[i]
		

	# 	ionSpecByAtom = {'H0':specH0, 'He0':specHe0, 'He1':specHe1}

	# 	photIonSpec = getionphotspec(lowerBound, xH, xe, spec, ionSpecByAtom)

	# 	totPhotIonSpec = sum(photIonSpec[atom] for atom in atoms)

	# 	spec[where(spec.eng > lowerBound)] = 0

	# 	spec[where(spec.eng <= lowerBound)] += -totPhotIonSpec[where(spec.eng <= lowerBound)]

	# 	spec.redshift()

	return



# TO-DO: Write the function to get the low-energy result: photoionization + redshifting of photons below threshold. 