"""Definition for the class `Spectrum`."""

from numpy     import *
import utilities as utils 

class Spectrum:
	"""Structure for photon and electron spectra with log-binning in energy.
	
	Parameters
	----------
	eng, dNdE : array_like 
		Abscissa for the spectrum and spectrum stored as dN/dE. Must be log-spaced. 
	rs : float
		The redshift of the spectrum. 

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

	#__array_priority__ must be larger than 0, so that radd can work. Otherwise, ndarray + Spectrum works by iterating over the elements of ndarray first, which isn't what we want. 
	__array_priority__ = 1

	def __init__(self, eng, dNdE, rs):
		
		if eng.size != dNdE.size:
			raise TypeError("abscissa and spectrum need to be of the same size.")
		if not all(diff(eng) > 0): 
			raise TypeError("abscissa must be ordered in increasing energy.")
		
		self.eng             = eng
		self.dNdE            = dNdE
		self.rs              = rs 
		self.length          = eng.size
		self.underflow       = {'N':0., 'eng':0.}

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
			if not array_equal(self.rs, other.rs):
				raise TypeError("redshifts are different for the two spectra.")
			newSpectrum = Spectrum(self.eng, self.dNdE + other.dNdE, self.rs)
			newSpectrum.underflow['N'] = self.underflow['N'] + other.underflow['N']
			newSpectrum.underflow['eng'] = self.underflow['eng'] + other.underflow['eng']
			return newSpectrum
		elif isinstance(other,ndarray) and other.ndim == 1 and other.size == self.length: 
			return Spectrum(self.eng, self.dNdE + other     , self.rs)
		else: 
			raise TypeError("adding an object that is not a list or is the wrong length.")

	def __radd__(self, other):
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
			if not array_equal(self.rs, other.rs):
				raise TypeError("redshifts are different for the two spectra.")
			newSpectrum = Spectrum(self.eng, self.dNdE + other.dNdE, self.rs)
			newSpectrum.underflow['N'] = self.underflow['N'] + other.underflow['N']
			newSpectrum.underflow['eng'] = self.underflow['eng'] + other.underflow['eng']
			return newSpectrum
		elif isinstance(other,ndarray) and other.ndim == 1 and other.size == self.length: 
			return Spectrum(self.eng, self.dNdE + other     , self.rs)
		else: 
			print(isinstance(other,ndarray))
			print(other.ndim == 1)
			print(other.size == self.length)
			raise TypeError("adding an object that is not a list or is the wrong length.")

	def __mul__(self, other):
		if issubdtype(type(other),float) or issubdtype(type(other),integer):
			newSpectrum = Spectrum(self.eng, other*self.dNdE, self.rs)
			newSpectrum.underflow['N'] = self.underflow['N']*other
			newSpectrum.underflow['eng'] = self.underflow['eng']*other
			return newSpectrum
		elif isinstance(other,ndarray) and other.ndim == 1 and other.size == self.length:
			return Spectrum(self.eng, other*self.dNdE, self.rs)
		else:
			raise TypeError("can only multiply scalars or ndarrays. Please use Spectrum.contract for matrix multiplication.")

	def __rmul__(self, other):
		if issubdtype(type(other),float) or issubdtype(type(other),integer):
			newSpectrum = Spectrum(self.eng, other*self.dNdE, self.rs)
			newSpectrum.underflow['N'] = self.underflow['N']*other
			newSpectrum.underflow['eng'] = self.underflow['eng']*other
			return newSpectrum
		elif isinstance(other,ndarray) and other.ndim == 1 and other.size == self.length:
			return Spectrum(self.eng, other*self.dNdE, self.rs)
		else:
			print(type(other))
			raise TypeError("can only multiply scalars or ndarrays. Please use Spectrum.contract for matrix multiplication.")

	def __truediv__(self, other):
		return self*(1/other)

	def contract(self, mat):
		if isinstance(mat,ndarray) or isinstance(mat,list):
			return dot(mat,self.dNdE)
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
			return sum(dNdlogE)*logBinWidth + self.underflow['N']

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
			return dot(self.eng,dNdlogE)*logBinWidth + self.underflow['eng']

	def redshift(self, rsOut):

	 	prevBinEngBound = (self.rs/rsOut)*self.binBoundary

 		if self.rs < rsOut:
 			print("WARNING: blueshifting will cause underflow to be set to NaN.")
 			self.underflow['N'] = NaN
 			self.underflow['eng'] = NaN
 		self.underflow['N']   +=   self.totN(type='eng', low=self.binBoundary[0], upp=prevBinEngBound[0])
 		self.underflow['eng']  = (self.underflow['eng']+self.toteng(type='eng', low=self.binBoundary[0], upp=prevBinEngBound[0]))*rsOut/self.rs

	 	self.dNdE = [self.totN(type='eng', low=prevBinEngBound[i], upp=prevBinEngBound[i+1]) for i in arange(self.length)]/(self.binWidth*self.eng)
	 	self.rs   = rsOut

def sumspectrum(spectrumList):

	newSpectrum = Spectrum(self.eng, zeros(self.eng.size), self.rs[-1])
	for spec in spectrumList:
		newSpectrum += spec
	return newSpectrum
	

class Spectra:
	"""Structure for photon and electron spectra over many redshifts, with log-binning in energy.
	
	Parameters
	----------
	rs : array_like
		The redshifts of the spectra. Redshifts should be stored in reverse order.
	eng : array_like 
		Energy abscissa for the spectrum and spectrum stored as dN/dE. Must be log-spaced. 
	
	spectrumList : list of Spectrum
		One-dimensional list of Spectrums. 

	Attributes
	----------
	
	binWidth : float
		The *log* bin width. 
	binBoundary : ndarray
		The boundary of each energy bin. Has one more entry than ``eng.size``. 

	"""
	#__array_priority__ must be larger than 0, so that radd can work. Otherwise, ndarray + Spectrum works by iterating over the elements of ndarray first, which isn't what we want. 
	__array_priority__ = 1

	def __init__(self, rs, eng, spectrumList):
		
		if not utils.arrays_equal([spec.eng for spec in spectrumList]): 
			raise TypeError("all spectrum.eng must be equal.")


		if len(set(spec.length for spec in spectrumList)) > 1:
			raise TypeError("Spectrum in spectrumList are not of the same length.")

		if len(spectrumList) != (rs.size):
			print(len(spectrumList))
			print(rs.size)
			raise TypeError("spectrumList should have dimensions of rs.size")

		if not all(diff(eng) > 0): 
			raise TypeError("abscissa must be ordered in increasing energy.")

		if not all(diff(rs) < 0):
			raise TypeError("redshift must be in decreasing order.")
		
		self.eng             = spectrumList[0].eng
		self.rs              = rs
		self.spectrumList    = spectrumList
				
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
		if isinstance(other,Spectra):
			if not array_equal(self.eng, other.eng):
				raise TypeError("abscissae are different for the two spectra.")
			if not array_equal(self.rs, other.rs):
				raise TypeError("redshifts are different for the two spectra.")
			return Spectra(self.rs, self.eng, [spec1 + spec2 for spec1,spec2 in zip(self.spectrumList, other.spectrumList)])
		elif isinstance(other,ndarray) and other.ndim == 1 and other.size == self.eng.size: 
			return Spectra(self.rs, self.eng, [spec + other for spec in self.spectrumList])
		else: 
			raise TypeError("adding an object that is not of class Spectra or ndarray.")

	def __radd__(self, other):
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
		if isinstance(other,Spectra):
			if not array_equal(self.eng, other.eng):
				raise TypeError("abscissae are different for the two spectra.")
			if not array_equal(self.rs, other.rs):
				raise TypeError("redshifts are different for the two spectra.")
			return Spectra(self.rs, self.eng, [spec1 + spec2 for spec1,spec2 in zip(self.spectrumList, other.spectrumList)])
		elif isinstance(other,ndarray) and other.ndim == 1 and other.size == self.eng.size: 
			return Spectra(self.rs, self.eng, [spec + other for spec in self.spectrumList])
		else: 
			raise TypeError("adding an object that is not of class Spectra or ndarray.")

	def __mul__(self, other):
		# Multiplies spectra at each redshift by some array. 
		if (issubdtype(type(other),float) or issubdtype(type(other),int) or (isinstance(other,ndarray) and other.ndim == 1 and other.size == self.eng.size) 
			):
			return Spectra(self.rs, self.eng, [other*spec for spec in self.spectrumList])
		elif isinstance(other,ndarray):
			if other.shape == (self.rs,self.eng):
				return Spectra(self.rs, self.eng, [other[i,:]*spectrumList[i] for i in arange(self.rs.size)])
		elif isinstance(other,Spectra):
			if self.rs != other.rs or self.eng != other.rs:
				raise TypeError("the two spectra do not have the same abscissae.")
			return Spectra(self.rs, self.eng, [spec1*spec2 for spec1, spec2 in zip(self.spectrumList, other.spectrumList)])
		else:
			raise TypeError("can only multiply scalars or ndarrays. Please use Spectrum.contract for matrix multiplication.")

	def __rmul__(self, other):
		# Multiplies spectra at each redshift by some array. 
		if (issubdtype(type(other),float) or issubdtype(type(other),integer) or (isinstance(other,ndarray) and other.ndim == 1 and other.size == self.eng.size) 
			):
			return Spectra(self.rs, self.eng, [spec*other for spec in self.spectrumList])
		elif isinstance(other,ndarray):
			if other.shape == (self.rs,self.eng):
				return Spectra(self.rs, self.eng, [other[i,:]*spectrumList[i] for i in arange(self.rs.size)])
		else:
			raise TypeError("can only multiply scalars or ndarrays. Please use Spectrum.contract for matrix multiplication.")

	def sumbyengweight(self, mat):
		if isinstance(mat,ndarray) or isinstance(mat,list):
			return array([spec.contract(mat) for spec in self.spectrumList])
		else:
			raise TypeError("can only contract lists or ndarrays.")

	def sumbyrsweight(self,weight):
		if isinstance(weight,ndarray) and weight.ndim == 1:
			newSpectrum = weight[-1]*self.spectrumList[-1]
			for i in arange(len(self.spectrumList)-1):
				self.spectrumList[i].rs = self.rs[-1]
				newSpectrum += weight[i]*self.spectrumList[i]
			return newSpectrum

	def append(self,spec):
		if not array_equal(self.eng, spec.eng):
			raise TypeError("new spectrum does not have the same energy abscissa.")
		if self.rs[-1] <= spec.rs: 
			raise TypeError("addspectrum currently only supports appending spectra at the end, which must have a lower redshift than the last spectrum.")
		self.spectrumList.append(spec)
		self.rs = append(self.rs,spec.rs)

		


