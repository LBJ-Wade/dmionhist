from numpy import *

def arrays_equal(ndarrayList):
	same = True
	ind = 0
	while same and ind < len(ndarrayList)-1:
		same = same & array_equal(ndarrayList[ind],ndarrayList[ind+1])
		ind += 1
	return same