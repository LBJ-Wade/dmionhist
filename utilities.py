from numpy import *

def arrays_equal(ndarrayList):
	same = True
	ind = 0
	while same and ind < len(ndarrayList)-1:
		same = same & array_equal(ndarrayList[ind],ndarrayList[ind+1])
		ind += 1
	return same

def logspaced(arr):
	return ptp(diff(log(arr))) < 1e-5

def divignorebyzero(a,b,val=0):
	with errstate(divide='ignore', invalid='ignore'): 
		c = true_divide(a,b)
		c[~ isfinite(c)] = val
	return c