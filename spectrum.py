"""Definition for the class `Spectrum`."""

from numpy import *
from scipy.interpolate import interp1d

class Spectrum:
	"""Structure for photon and electron spectra with log-binning in energy.
	
	Parameters
	----------
	eng, dNdE : array_like 
		Abscissa for the spectrum and spectrum stored as dN/dE. Must be log-spaced. 
	rs : float
		The redshift of the spectrum. 
	underflowSwitch : bool, optional
		Sets whether to keep track of underflow. Defaults to ``False``. 

	Attributes
	----------
	length : int
		The length of the `eng` and `dNdE`. 
	underflow : dict
		The underflow total number of particles and total energy, initialized to {'N':0., 'eng':0.}. 
	binWidth : float
		The *log* bin width. 
	binBoundary : ndarray
		The boundary of each energy bin. Has one more entry than `length`. 

	"""


	def __init__(self, eng, dNdE, rs, underflowSwitch=False):
		
		if eng.size != dNdE.size:
			raise TypeError("abscissa and spectrum need to be of the same size.")
		if not all(diff(eng) > 0): 
			raise TypeError("abscissa must be ordered in increasing energy.")
		
		self.eng             = eng
		self.dNdE            = dNdE
		self.rs              = rs 
		self.length          = eng.size
		self.underflowSwitch = underflowSwitch
		if underflowSwitch == True:
			self.underflow = {'N':0., 'eng':0.}

		binWidth         = log(eng[1]) - log(eng[0])
		self.binWidth    = binWidth

		binBoundary = sqrt(eng[:-1]*eng[1:])
		lowLim = exp(log(eng[0])  - binWidth/2)
		uppLim = exp(log(eng[-1]) + binWidth/2)
		binBoundary = insert(binBoundary,0,lowLim)
		binBoundary = append(binBoundary,uppLim)

		self.binBoundary = binBoundary


	def __add__(self, other):
		"""Adds two `Spectrum` instances together, or an array to `dNdE`. 
		
		Parameters
		----------
		other : Spectrum or ndarray

		Returns
		-------
		Spectrum
			New `Spectrum` instance which is the sum of the array with `dNdE`. 

		Raises
		------
		TypeError
			The abcissae are different for the two `Spectrum`. 
			The redshifts are different for the two `Spectrum`. 
			`other` is not a `Spectrum` or ``ndarray``. 

		"""
		if isinstance(other,Spectrum):
			if not array_equal(self.eng, other.eng):
				raise TypeError("abscissae are different for the two spectra.")
			if self.rs != other.rs:
				raise TypeError("redshifts are different for the two spectra.")
			return Spectrum(self.eng, self.dNdE + other.dNdE, self.rs, self.underflowSwitch)
		elif isinstance(other,ndarray) and other.ndim == 1 and other.size == self.length: 
			return Spectrum(self.eng, self.dNdE + other     , self.rs, self.underflowSwitch)
		else: 
			raise TypeError("adding an object that is not a list or is the wrong length.")

	def __mul__(self, other):
		if (isinstance(other,float) or isinstance(other,int) or (isinstance(other,ndarray) and other.ndim == 1) 
			):
			return Spectrum(self.eng, other*self.dNdE, self.rs, self.underflowSwitch)
		else:
			raise TypeError("can only multiply scalars or ndarrays. Please use Spectrum.contract for matrix multiplication.")

	def __rmul__(self, other):
		if (isinstance(other,float) or isinstance(other,int) or (isinstance(other,ndarray) and other.ndim == 1) 
			):
			return Spectrum(self.eng, other*self.dNdE, self.rs, self.underflowSwitch)
		else:
			raise TypeError("can only multiply scalars or ndarrays. Please use Spectrum.contract for matrix multiplication.")

	def contract(self, mat):
		if isinstance(mat,ndarray) or isinstance(mat,list):
			self.dNdE = dot(mat,self.dNdE)
		else:
			raise TypeError("can only contract lists or ndarrays.")

	def totN(self, type='all', low=None, upp=None):
		dNdlogE     = self.eng*self.dNdE
		logBinWidth = self.binWidth
		length      = self.length

		if type == 'bin':
			lowBound = max([0,low])
			uppBound = min([upp,length])
			return sum(dNdlogE[lowBound:uppBound])*logBinWidth

		if type == 'eng':
			logBinBound = log(self.binBoundary)
						
			lowEngBinInd  = interp(log(low),logBinBound,arange(logBinBound.size), left=-1, right=length+1)
			uppEngBinInd  = interp(log(upp),logBinBound,arange(logBinBound.size), left=-1, right=length+1)
			
			NFullBins = self.totN(type='bin', low=int(ceil(lowEngBinInd)), upp=int(floor(uppEngBinInd)))
			NPartBins = 0
			if lowEngBinInd > 0 and lowEngBinInd < length and floor(lowEngBinInd) == floor(uppEngBinInd):
				NPartBins += dNdlogE[int(floor(lowEngBinInd))]*(uppEngBinInd - lowEngBinInd)*logBinWidth
			else:
				if lowEngBinInd > 0 and lowEngBinInd < length:
					NPartBins += dNdlogE[int(floor(lowEngBinInd))]*(ceil(lowEngBinInd)-lowEngBinInd)*logBinWidth
				if uppEngBinInd > 0 and uppEngBinInd < length:
					NPartBins += dNdlogE[int(floor(uppEngBinInd))]*(uppEngBinInd - floor(uppEngBinInd))*logBinWidth
			
			return NFullBins+NPartBins

		if type == 'all':
			if self.underflowSwitch:
				return sum(dNdlogE)*logBinWidth + self.underflow['N']
			else: 
				return sum(dNdlogE)*logBinWidth

	def toteng(self, type='all', low=None, upp=None):
		eng         = self.eng
		dNdlogE     = self.eng*self.dNdE
		logBinWidth = self.binWidth
		length      = self.length

		if type == 'bin':
			lowBound = max([0,low])
			uppBound = min([upp,length])
			return dot(self.eng[lowBound:uppBound],dNdlogE[lowBound:uppBound])*logBinWidth

		if type == 'eng':
			logBinBound = log(self.binBoundary)
			
			lowEngBinInd  = interp(log(low), logBinBound, arange(logBinBound.size), left=-1, right=length+1)
			uppEngBinInd  = interp(log(upp), logBinBound, arange(logBinBound.size), left=-1, right=length+1)

			engFullBins = self.toteng(type='bin', low=int(ceil(lowEngBinInd)), upp=int(floor(uppEngBinInd)))
			engPartBins = 0
			if lowEngBinInd > 0 and lowEngBinInd < length and floor(lowEngBinInd) == floor(uppEngBinInd):
				engPartBins += eng[int(floor(lowEngBinInd))]*dNdlogE[int(floor(lowEngBinInd))]*(uppEngBinInd - lowEngBinInd)*logBinWidth
			else:
				if lowEngBinInd > 0 and lowEngBinInd < length:
					engPartBins += eng[int(floor(lowEngBinInd))]*dNdlogE[int(floor(lowEngBinInd))]*(ceil(lowEngBinInd)-lowEngBinInd)*logBinWidth
				if uppEngBinInd > 0 and uppEngBinInd < length:
					engPartBins += eng[int(floor(uppEngBinInd))]*dNdlogE[int(floor(uppEngBinInd))]*(uppEngBinInd - floor(uppEngBinInd))*logBinWidth
			
			return engFullBins+engPartBins	

		if type == 'all':
			if self.underflowSwitch:
				return dot(self.eng,dNdlogE)*logBinWidth + self.underflow['eng']
			else:
				return dot(self.eng,dNdlogE)*logBinWidth

	def redshift(self, rsOut):

	 	prevBinEngBound = (self.rs/rsOut)*self.binBoundary

	 	if self.underflowSwitch:
	 		if self.rs < rsOut:
	 			raise NotImplementedError("Underflow is not supported for blueshifting.")
	 		self.underflow['N']   +=   self.totN(type='eng', low=self.binBoundary[0], upp=prevBinEngBound[0])
	 		self.underflow['eng'] += self.toteng(type='eng', low=self.binBoundary[0], upp=prevBinEngBound[0])*rsOut/self.rs

	 	self.dNdE = [self.totN(type='eng', low=prevBinEngBound[i], upp=prevBinEngBound[i+1]) for i in arange(self.length)]/(self.binWidth*self.eng)
	 	self.rs   = rsOut

