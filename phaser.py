""" 2015-04-08 10:27 EST

The phaser module provides an implementation of the phase estimation algorithm
of "Estimating the phase of synchronized oscillators"; 
   S. Revzen & J. M. Guckenheimer; Phys. Rev. E; 2008, v. 78, pp. 051907
   doi: 10.1103/PhysRevE.78.051907

Phaser takes in multidimensional data from multiple experiments and fits the
parameters of the phase estimator, which may then be used on new data or the
training data. The output of Phaser is a phase estimate for each time sample
in the data. This phase estimate has several desirable properties, such as:
(1) d/dt Phase is approximately constant
(2) Phase estimates are robust to measurement errors in any one variable
(3) Phase estimates are robust to systematic changes in the measurement error

The top-level class of this module is Phaser. 
An example is found in test_sinnp.cos(); it requires matplotlib

COPYRIGHT

(c) Shai Revzen (2008), Simon Wilshin (2015)
test_sincos (c) Jimmy Sastra (2011)

This code is released under the GNU Public License version 3.0, with the
additional restriction that any use in research must appropriately cite 
the document doi: 10.1103/PhysRevE.78.051907 (above)
"""

import numpy as np
# from phaserutil import *
from scipy import signal
from scipy import stats
import warnings

from copy import deepcopy

class FourierSeries(object):
  def take(self,cols):
    """Get a FourierSeries that only has the selected columns"""
    other = self.copy()
    other.coef = other.coef[:,cols]
    other.m = other.m[cols]
    return other
    
  def val( self, phi ):
    """Evaluate fourier series at all the phases phi
       
       Returns rows corresponding to phi.flatten()
    """
    phi = np.asarray(phi).flatten()
    phi.shape = (len(phi.flat),1)
    th = phi * self.om
    return np.dot(np.exp(1j*th),self.coef)+self.m
  
  def residuals(self,phi):
    '''
    
    '''
    pass
    
  
  def integrate( self, z0=0 ):
    """Integrate fourier series, and set the mean values to z0
    """
    self.m[:] = np.asarray(z0)
    self.coef = -1j * self.coef / self.om.T
    return self
  
  def getDim( self ):
    """Get dimension of output"""
    return len(self.m)
  
  def getOrder( self ):
    """Get order of Fourier series"""
    return self.coef.shape[0]/2
    
  def extend( self, other ):
    """Extend a fourier series with additional output columns from
       another fourier series of the same order 
       
       If fs1 is order k with c1 output colums, and
       fs2 is order k with c2 output columns then the following holds:
       
       fs3 = fs1.copy().append(fs2)
       assert allclose( fs3.val(x)[:c1], fs1.val(x) )
       assert allclose( fs3.val(x)[c1:], fs2.val(x) )
    """
    assert len(other.om) == len(self.om), "Must have same order"
    self.m = np.hstack((self.m,other.m))
    self.coef = np.hstack((self.coef,other.coef))
    
  def diff( self ):
    """Differentiate the fourier series"""
    self.m[:] = 0
    self.coef = 1j * self.coef * self.om.T
    return self
  
  def copy( self ):
    """Return copy of the current fourier series"""
    return deepcopy( self )
  
  def fit( self, order, ph, data ):
    """Fit a fourier series to data at phases phi
       
       data is a row or two-dimensional array, with data points in columns
    """
    
    phi = np.reshape( np.mod(ph + np.pi,2*np.pi) - np.pi, (1,len(ph.flat)) )
    if phi.shape[1] != data.shape[1]:
      raise IndexError(
        "There are %d phase values for %d data points" 
            % (phi.shape[1],data.shape[1]))
    # Sort data by phase
    idx = np.argsort(phi).flatten()
    dat = np.c_[data.take(idx,axis=-1),data[:,idx[0]]]
    phi = np.concatenate( (phi.take(idx),[phi.flat[idx[0]]+2*np.pi]) )
    
    # Compute means values and subtract them
    #self.m = mean(dat,1).T
    # mean needs to be computed by trap integration also
    dphi = np.diff(phi)
    self.m = np.sum((dat[:,:-1] + dat[:,1:]) * .5 * dphi[np.newaxis,:], axis = 1) / (max(phi) - min(phi))
    #PDB.set_trace()
    dat = (dat.T - self.m).T
    # Allow 0th order (mean value) models
    order = max(0,order)
    self.order = order
    if order<1:
      order = 0
      self.coef = None
      return
    # Compute frequency vector
    om = np.zeros( 2*order )
    om[::2] = np.arange(1,order+1)
    om[1::2] = -om[::2]
    self.om = np.reshape(om,(1,order*2))
    # Compute measure for integral
    #if any(dphi<=0):
      #raise UserWarning,"Duplicate phase values in data"
    # Apply trapezoidal rule for data points (and add 2 np.pi factor needed later)      
    zd = (dat[:,1:]+dat[:,:-1])/(2.0*2*np.pi) * dphi
    # Compute phase values for integrals
    th = self.om.T * (phi[1:]-dphi/2)
    # Coefficients are integrals
    self.coef = np.dot(np.exp(-1j*th),zd.T)
    return self

  def fromAlien( self, other ):
    self.order = int(other.order)
    self.m = other.m.flatten()
    if other.coef.shape[0] == self.order * 2:
      self.coef = other.coef
    else:
      self.coef = other.coef.T
    self.om = other.om
    self.om.shape = (1,len(self.om.flat))
    return self
  
  def filter( self, coef ):
    """Filter the signal by multiplication in the frequency domain
       Assuming an order N fourier series of dimension D,
       coef can be of shape:
        N -- multiply all D coefficients of order k by 
            coef[k] and conj(coef[k]), according to their symmetry
        2N -- multiply all D coefficients of order k by 
            coef[2k] and coef[2k+1]
        1xD -- multiply each coordinate by the corresponding coef 
        NxD -- same as N, but per coordinate
        2NxD -- the obvious...
    """
    coef = np.asarray(coef)
    if coef.shape == (1,self.coef.shape[1]):
      c = coef
    elif coef.shape[0] == self.coef.shape[0]/2:
      if coef.ndim == 1:
        c = empty( (self.coef.shape[0],1), dtype=self.coef.dtype )
        c[::2,0] = coef
        c[1::2,0] = np.conj(coef)
      elif coef.ndim == 2:
        assert coef.shape[1]==self.coef.shape[1],"Same dimension"
        c = empty_like(self.coef)
        c[::2,:] = coef
        c[1::2,:] = np.conj(coef)
      else:
        raise ValueError("coef.ndim must be 1 or 2")
    self.coef *= c
    return self
        
  def bigSum( fts, wgt = None ):
    """[STATIC] Compute a weighted sum of FourierSeries models. 
       All models must have the same dimension and order.
       
       INPUT:
         fts -- sequence of N models
         wgt -- sequence N scalar coefficients, or None for averaging  
    
       OUTPUT:
         a new FourierSeries object
    """
    N = len( fts )
    if wgt is None:
      wgt = np.ones(N)/float(N)
    else:
      wgt = np.asarray(wgt)
      assert wgt.size==len(fts)
    
    fm = FourierSeries()
    fm.coef = np.zeros_like(fts[0].coef)
    fm.m = np.zeros_like(fts[0].m)
    for fs,w in zip(fts,wgt):
      # fm.coef += w * fs.coef
      # fm.m += w * fs.m
      np.add(fm.coef, w*fs.coef, out=fm.coef, casting='unsafe')
      np.add(fm.m, w*fs.m, out=fm.m, casting='unsafe')
    fm.order = fts[0].order
    fm.om = fts[0].om
    
    return fm
  bigSum = staticmethod( bigSum )

class ZScore( object ):
  """
  Class for finding z scores of given measurements with given or computed
  covarance matrix.
  
  This class implements equation (7) of [Revzen08]
  
  Properties:
    y0 -- Dx1 -- measurement mean
    M -- DxD -- measurement covariance matrix
    S -- DxD -- scoring matrix
  """
  
  def __init__( self, y = None, M = None ):
    """Computes the mean and scoring matrix of measurements
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
      M -- DxD (optional) -- measurement error covariance for y
        -- If M is missing, it is assumed to be diagonal with variances
    -- given by 1/2 variance of the second order differences of y
    """
    
    # if M given --> use fromCovAndMean
    # elif we got y --> use fromData
    # else --> create empty object with None in members 
    if M is not None:
      self.fromCovAndMean( stats.nanmean(y, 1), M)
    elif y is not None:
      self.fromData( y )
    else:
      self.y0 = None
      self.M = None
      self.S = None
    
  
  def fromCovAndMean( self, y0, M ):
    """
    Compute scoring matrix based on square root of M through svd
    INPUT:
      y0 -- Dx1 -- mean of data
      M -- DxD -- measurement error covariance of data
    """
    self.y0 = y0
    self.M = M
    (D, V) = linalg.eig( M )
    self.S = np.dot( V.transpose(), np.diag( 1/np.sqrt( D ) ) )
  
  def fromData( self, y ):
    """
    Compute scoring matrix based on estimated covariance matrix of y
    Estimated covariance matrix is geiven by 1/2 variance of the second order
    differences of y
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    """
    self.y0 = stats.nanmean( y, 1 )
    self.M = np.diag( stats.nanstd( np.diff( y, n=2, axis=1 ), axis=1 ) )
    self.S = np.diag( 1/np.sqrt( np.diag( self.M ) ) )
  
  def __call__( self, y ):
    """
    Callable wrapper for the class
    Calls self.zScore internally
    """
    return self.zScore( y )
  
  def zScore( self, y ):
    """Computes the z score of measurement y using stored mean and scoring
    matrix
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    OUTPUT:
      zscores for y -- DxN
    """
    return np.dot( self.S, y - self.y0.reshape( len( self.y0 ), 1 ) )


def _default_psf(x):
  """Default Poincare section function
     by rights, this should be inside the Phaser class, but np.pickle
     would barf on Phaser objects if they contained functions that
     aren't defined in the module top-level.
  """
  return signal.lfilter( 
    np.array([0.02008336556421, 0.04016673112842,0.02008336556421] ), 
    np.array([1.00000000000000,-1.56101807580072,0.64135153805756] ),
  x[0,:] )
  
class Phaser( object ):
  """
  Concrete class implementing a Phaser phase estimator
  
  Instance attributes:
    sc -- ZScore object for converting y to z-scores
    P_k -- list of D FourierSeries objects -- series correction for correcting proto-phases
    prj -- D x 1 complex -- projector on combined proto-phase
    P -- FourierSeries object -- series correction for combined phase
    psf -- callable -- callback to psecfun
  """

  def __init__( self, y = None, C = None, ordP = None, psecfunc = _default_psf, protophfun = signal.hilbert ):
    """
    Initilizing/training a phaser object
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
      ordP -- 1x1 (optional) -- Orders of series to use in series correction
      psecfunc -- 1x1 (optional) -- Poincare section function
      protophfun -- function -- protophase function with call signature and return identical to numpy hilbert transform
    """

    self.protophfun = protophfun
    self.psf = psecfunc
    
    # if y given -> calls self.phaserTrain
    if y is not None:
      self.phaserTrain( y, C, ordP )
     

  def __call__( self, dat ):
    """
    Callable wrapper for the class. Calls phaserEval internally
    """
    return self.phaserEval( dat )
  
    
  def phaserEval( self, dat ):
    """
    Computes the phase of testing data
    INPUT:
      dat -- DxN -- Testing data whose phase is to be determined
    OUTPUT:
      Returns the complex phase of input data
    """
    
    # compute z score
    z = self.sc.zScore( dat )
    
    # compute Poincare section
    p0 = self.psf( dat )
    
    # compute protophase using Hilbert transform
    zeta = self.mangle * self.protophfun( z )
    z0, ido0 = Phaser.sliceN( zeta, p0 )
    
    # Compute phase offsets for proto-phases
    ofs = np.exp(-1j * np.angle(stats.nanmean(z0, axis = 1)).T)
    
    # series correction for each dimision using self.P_k
    th = Phaser.angleUp( zeta * ofs[:,np.newaxis] ) 
    
    # evaluable weights based on sample length
    p = 1j * np.zeros( th.shape )
    for k in range( th.shape[0] ):
      p[k,:] = self.P_k[k].val( th[k,:] ).T + th[k,:]
    
    rho = stats.nanmean( abs( zeta ), 1 ).reshape(( zeta.shape[0], 1 ))
    # compute phase projected onto first principal components using self.prj
    ph = Phaser.angleUp( np.dot( self.prj.T, np.vstack( [np.cos( p ) * rho, np.sin( p ) * rho] ) ))
    
    # return series correction of combined phase using self.P
    phi = np.real( ph + self.P.val( ph ).T )
    pOfs2 = (p0[ido0+1] * np.exp(1j * phi.T[ido0+1]) - p0[ido0] * np.exp(1j * phi.T[ido0] )) / (p0[ido0+1] - p0[ido0])
    return phi - np.angle(np.sum(pOfs2))
  
  def phaserTrain( self, y, C = None, ordP = None ):
    """
    Trains the phaser object with given data.
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
    """
    
    # if given one sample -> treat it as an ensemble with one element
    if y.__class__ is np.ndarray:
      y = [y]
    # Copy the list container
    y = [yi for yi in y]
    # check dimension agreement in ensemble
    if len( set( [ ele.shape[0] for ele in y ] ) ) is not 1:
      raise Exception
    D = y[0].shape[0]
    
    # train ZScore object based on the entire ensemble
    self.sc = ZScore( np.hstack( y ), C )
    
    # initializing proto phase variable
    zetas = []
    cycl = np.zeros( len( y ))
    svm = 1j*np.zeros( (D, len( y )) )
    svv = np.zeros( (D, len( y )) )
    
    # compute protophases for each sample in the ensemble
    for k in range( len( y ) ):
      # hilbert transform the sample's z score
      zetas.append( self.protophfun( self.sc.zScore( y[k] ) ) )
      # trim beginning and end cycles, and check for cycle freq and quantity
      cycl[k], zetas[k], y[k] = Phaser.trimCycle( zetas[k], y[k] )

      # Computing the Poincare section
      sk = self.psf( y[k] )
      (sv, idx) = Phaser.sliceN( zetas[k], sk )
      if idx.shape[-1] == 0:
        raise Exception( 'newPhaser:emptySection', 'Poincare section is empty -- bailing out' )
      
      svm[:,k] = stats.nanmean( sv, 1 )
      svv[:,k] = np.var( sv, 1 ) * sv.shape[1] / (sv.shape[1] - 1)

    
    # computing phase offset based on psecfunc
    self.mangle, ofs = Phaser.computeOffset( svm, svv )
    
    # correcting phase offset for proto phase and compute weights
    wgt = np.zeros( len( y ) )
    rho_i = np.zeros(( len( y ), y[0].shape[0] ))
    for k in range( len( y ) ):
      zetas[k] = self.mangle * np.exp( -1j * ofs[k] ) * zetas[k]
      wgt[k] = zetas[k].shape[0]
      rho_i[k,:] = stats.nanmean( abs( zetas[k] ), 1 )
    
    # compute normalized weight for each dimension using weights from all samples
    wgt = wgt.reshape(( 1, len( y )))
    rho = ( np.dot( wgt, rho_i ) / np.sum( wgt ) ).T
    # if ordP is None -> use high enough order to reach Nyquist/2
    # 
    # if ordP is None:
    #   ordP = np.ceil( max( cycl ) / 4 )
    # This works well if you have one nice long segment, but for lots of little 
    # chunks it uses too few coefficients. Here we sum the coefficients and 
    # Estimate 10%
    if ordP is None:
        ordP = np.floor(np.sum(cycl)/10)
   
    # correct protophase using seriesCorrection
    self.P_k = Phaser.seriesCorrection( zetas, ordP )
    
    
    # loop over all samples of the ensemble
    q = []
    for k in range( len( zetas ) ):
      # compute protophase angle
      th = Phaser.angleUp( zetas[k] )
      
      phi_k = 1j * np.ones( th.shape )
      
      # loop over all dimensions
      for ki in range( th.shape[0] ):
        # compute corrected phase based on protophase
        phi_k[ki,:] = self.P_k[ki].val( th[ki,:] ).T + th[ki,:]
      
      # computer vectorized phase
      q.append( np.vstack( [np.cos( phi_k ) * rho, np.sin( phi_k ) * rho] ) )
    
    # project phase vectors using first two principal components
    W = np.hstack( q[:] )
    W = W - stats.nanmean( W, 1 )[:,np.newaxis]
    pc = np.linalg.svd( W, False )[0]
    self.prj = np.reshape( pc[:,0] + 1j * pc[:,1], ( pc.shape[0], 1 ) )
    
    # Series correction of combined phase
    qz = []
    for k in range( len( q ) ):
      qz.append( np.dot( self.prj.T, q[k] ) )
    
    # store object members for the phase estimator
    self.P = Phaser.seriesCorrection( qz, ordP )[0]
  
  def computeOffset( svm, svv ):
    """
    """
    # convert variances into weights
    svv = svv / np.sum( svv, axis=1 ).reshape( svv.shape[0], 1 )
    # compute variance weighted average of phasors on cross section to give the phase offset of each protophase
    mangle = np.sum( svm * svv, axis=1)
    if any( abs( mangle ) ) < .1:
      b = find( abs( mangle ) < .1 )
      raise Exception( 'computeOffset:badmeasureOfs', len( b ) + ' measurement(s), including ' + b[0] + ' are too noisy on Poincare section' )
    
    # compute phase offsets for trials
    mangle = np.conj( mangle ) / abs( mangle )
    mangle = mangle.reshape(( len( mangle ), 1))
    svm = mangle * svm
    ofs = stats.nanmean( svm, 0 )
    if any( abs( ofs ) < .1 ):
      b = find( abs( ofs ) < .1 )
      raise Exception( 'computeOffset:badTrialOfs', len( b ) + ' trial(s), including ' + b[0] + ' are too noisy on Poincare section' )
    
    return mangle, np.angle( ofs )
  
  computeOffset = staticmethod( computeOffset )
  
  def sliceN( x, s, h = None ):
    """
    Slices a D-dimensional time series at a surface
    INPUT:
      x -- DxN -- data with colums being points in the time series
      s -- N, array -- values of function that is zero and increasing on surface
      h -- 1x1 (optional) -- threshold for transitions, transitions>h are ignored
    OUPUT:
      slc -- DxM -- positions at estimated value of s==0
      idx -- M -- indices into columns of x indicating the last point before crossing the surface
    """
    
    # checking for dimension agreement
    if x.shape[1] != s.shape[0]:
      raise Exception( 'sliceN:mismatch', 'Slice series must have matching columns with data' )
    
    # idx = find(( s[1:] > 0 ) & ( s[0:-1] <= 0 ))
    idx = np.where(np.logical_and(( s[1:] > 0 ),( s[0:-1] <= 0 )))
    idx = idx[0][idx[0] < x.shape[1]]

    if h is not None:
      idx = idx( abs( s[idx] ) < h & abs( s[idx+1] ) < h );
    
    N = x.shape[0]
    
    if len( idx ) is 0:
      return np.zeros(( N, 0 )), idx
    
    wBfr = abs( s[idx] )
    wBfr = wBfr.reshape((1, len( wBfr )))
    wAfr = abs( s[idx+1] )
    wAfr = wAfr.reshape((1, len( wAfr )))
    slc = ( x[:,idx]*wAfr + x[:,idx+1]*wBfr ) / ( wBfr + wAfr )
    
    return slc, idx
  
  sliceN = staticmethod( sliceN )
  
  def angleUp( zeta ):
    """
    Convert complex data to increasing phase angles
    INPUT:
      zeta -- DxN complex
    OUPUT:
      returns DxN phase angle of zeta
    """
    # unwind angles
    th = np.unwrap( np.angle ( zeta ) )
    # reverse decreasing sequences
    bad = th[:,0] > th[:,-1]
    if any( bad ):
      th[bad,:] = -th[bad,:]
    return th
  
  angleUp = staticmethod( angleUp )
  
  def trimCycle( zeta, y ):
    """
    """
    # compute wrapped angle for hilbert transform
    ph = Phaser.angleUp( zeta )

    # estimate nCyc in each dimension
    nCyc = abs( ph[:,-1] - ph[:,0] ) / 2 / np.pi
    cycl = np.ceil( zeta.shape[1] / max( nCyc ) )
    # if nCyc < 7 -> warning
    # elif range(nCyc) > 2 -> warning
    # else truncate beginning and ending cycles
    if any( nCyc < 7 ):
      warnings.warn( "PhaserForSample:tooShort" )
    elif max( nCyc ) - min( nCyc ) > 2:
      warnings.warn( "PhaserForSample:nCycMismatch" )
    elif np.isinf(cycl):
      warnings.warn( "PhaserForSample:InfiniteCycles" )
    else:
      zeta = zeta[:,cycl:-cycl]
      y = y[:,cycl:-cycl]
    
    return cycl, zeta, y
  
  trimCycle = staticmethod( trimCycle )
  
  def seriesCorrection( zetas, ordP ):
    """
    Fourier series correction for data zetas up to order ordP
    INPUT:
      zetas -- [DxN_1, DxN_2, ...] -- list of D dimensional data to be corrected using Fourier series
      ordP -- 1x1 -- Number of Fourier modes to be used
    OUPUT:
      Returns a list of FourierSeries object fitted to zetas
    """
    
    # initialize proto phase series 2D list
    proto = []
    
    # loop over all samples of the ensemble
    wgt = np.zeros( len( zetas ) )
    for k in range( len( zetas ) ):
      proto.append([])
      # compute protophase angle (theta)
      zeta = zetas[k]
      N = zeta.shape[1]
      theta = Phaser.angleUp( zeta )
      
      # generate time variable
      # Liberate this section of code
      t = np.linspace( 0, 1, N )      
      # compute d_theta
      dTheta = np.diff( theta, 1 )
      # compute d_t
      dt = np.diff( t )
      # mid-sampling of protophase angle
      th = ( theta[:,1:] + theta[:,:-1] ) / 2.0
      
      # loop over all dimensions
      for ki in range( zeta.shape[0] ):
        # evaluate Fourier series for (d_theta/d_t)(theta)
        # normalize Fourier coefficients to a mean of 1
        fdThdt = FourierSeries().fit( ordP * 2, th[ki,:].reshape(( 1, th.shape[1])), dTheta[ki,:].reshape(( 1, dTheta.shape[1])) / dt )
        fdThdt.coef = fdThdt.coef / fdThdt.m
        fdThdt.m = np.array([1])
        
        # evaluate Fourier series for (d_t/d_theta)(theta) based on Fourier
        # approx of (d_theta/d_t)
        # normalize Fourier coefficients to a mean of 1
        fdtdTh = FourierSeries().fit( ordP, th[ki,:].reshape(( 1, th.shape[1])), 1 / fdThdt.val( th[ki,:].reshape(( 1, th.shape[1] )) ).T )
        fdtdTh.coef = fdtdTh.coef / fdtdTh.m
        fdtdTh.m = np.array([1])
        
        # evaluate corrected phsae phi(theta) series as symbolic integration of 
        # (d_t/d_theta), this is off by a constant
        # Liberate to this
        proto[k].append(fdtdTh.integrate())
      
      # compute sample weight based on sample length
      wgt[k] = zeta.shape[0]
      
    wgt = wgt / np.sum( wgt )
    
    # return phase estimation as weighted average of phase estimation of all samples
    proto_k = []
    for ki in range( zetas[0].shape[0] ):
      proto_k.append( FourierSeries.bigSum( [p[ki] for p in proto], wgt ))
      
    return proto_k
    
  seriesCorrection = staticmethod( seriesCorrection )
  
def test_sincos():
  """
  Simple test/demo of Phaser, recovering a sine and cosine
  
  Demo courtesy of Jimmy Sastra, U. Penn 2011
  """
  import numpy as np
  import matplotlib.pyplot as plt
  from numpy import random

  # create separate trials and store times and data
  dats=[]
  t0 = []
  period = 55 # i
  phaseNoise = 0.05/np.sqrt(period)
  snr = 20
  N = 10
  print(N,"trials with:")
  print("\tperiod %.2g"%period,"(samples)\n\tSNR %.2g"%snr,"\n\tphase noise %.2g"%phaseNoise,"(radian/cycle)")
  print("\tlength = [", end=' ')
  for li in range(N):
    l = random.randint(400,2000) # length of trial
    dt = np.pi*2.0/period + random.randn(l)*phaseNoise # create noisy time steps
    t = np.cumsum(dt)+random.rand()*2*np.pi # starting phase is random
    raw = np.asarray([np.sin(t),np.cos(t)]) # signal
    raw = raw + random.randn(*raw.shape)/snr # SNR=20 noise
    t0.append(t)
    dats.append( raw - stats.nanmean(raw,axis=1)[:,np.newaxis] )
    print(l, end=' ')
  print("]")
 
  phr = Phaser( dats, psecfunc = lambda x : np.dot([1,-1],x)) 
  phi = [ phr.phaserEval( d ) for d in dats ] # extract phaseNoise
  reg = np.array([np.linspace(0,1,t0[0].size),np.ones(t0[0].size)]).T
  tt = np.dot( reg, np.linalg.lstsq(reg,t0[0])[0] )
  plt.plot(((tt-np.pi/4) % (2*np.pi))/np.pi-1, dats[0].T,'.')
  plt.plot( (phi[0].T % (2*np.pi))/np.pi-1, dats[0].T,'x') # plot data versus phase
  
  plt.legend(['sin(t)','cos(t)','sin(phi)','cos(phi)'])
  plt.axis([-1,1,-1.2,1.2])
  plt.show()

if __name__=="__main__":
  test_sincos()
