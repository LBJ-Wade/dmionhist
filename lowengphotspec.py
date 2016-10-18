from numpy    import *
from spectrum import *
from physics  import *

def getlowengphotdep(eng, rs, lowerbound, lowengphot, lowengphot_elementgrid):

	lowengphot_elementgrid[where(eng < rydberg),:,:] = 0
	ionSpectra = [Spectrum(eng,lowengphot_elementgrid[:,species,z],z) for species in arange(3) for z in rs]
	ionSpectra += 

# TO-DO: Write the function to get the low-energy result: photoionization + redshifting of photons below threshold. 