import logging
from numpy import where,array,Inf,arange,diff,exp,sqrt,logical_not,any,isnan,concatenate
from numpy.ma import getmaskarray,masked_where,getdata
from scipy.spatial import KDTree
try:
 try:
  from netCDF4 import default_fillvals as fv
 except:
  from netCDF4 import _default_fillvals as fv
except:
  raise ImportError('Could not import netCDF4 default fill-values')

class interpolationGrid:

  """Class for interpolation from irregularly distributed data on any
  output position. It can use two different approaches to interpolate
  or a combination of the two:

     - search for the N closest points
     - search for all points within a certain radius

  Usage:

     - Create an instance for the interpolation grid defined by an arra of
       size (KxDim) containing the coordinates of your input data an array of
       size (MxDim) with the coordinates of the desired output.
     - Call the interpolation grid with the data you want to interpolate.

  Attributes:

     shape (tuple of integers): number of input points
     mask(boolean array): mask of positions to exclude (True) from
        interpolation (e.g. land points in ocean fields)
     scaling (float array): 1-D array with scaling weights for each dimension
        in order to differentiate the impact of distances across dimensions
     locations (float array): 2D array with sequence of point coordinates with
        positions of original grid [Numper of points X Number of dimensions]
     ips (list of pointIterpolators): list of interpolators for each output grid
        point
  """

  def __init__(self,locations,newlocations,scaling=None,Mask=None,Range=1.,NoP=None):

      """Defines grid of input data and scaling factors for the various
      dimensions and computes look-up table of point interpolators for newlocations
      [K locations x Dimensions].

      Args:

         locations(float array): point coordinates of input grid. The geometry
            needs to be reshaped in 2D [No of data points M x No of dimensions].
         newlocations(float array): point coordinates of output grid. The
            geometry needs to be reshaped in 2D [No of data points M x No of
            dimensions].
         scaling (float array): 1D [Dimensions] array that provides multipliers
            to use to weight distances across dimensions.
         Mask: mask of positions to exclude (True) from
            interpolation (e.g. land points in ocean fields). This will
            effectively remove the mask before interpolation, so the mask will
            not be interoplated to the output grid. Set to None if you want mask
            interpolation.
         Range: distance of points in input grid to points in output grid to be
            included in interpolation
         NoP: Number of points of inputgrids to be included in interpolation for
            each output grid point

      Set Range and NoP to desired values to interpolate on output grid points
      using the NoP closest points or within the radius of Range.
      If both are set, the closest NoP are looked up and then Range is
      applied as multiple of the furthest of those. If NoP is not set, scaling
      should be used to normalise the distances appropriately to Range. If Range
      is set to None, only the NoP closest points will be used.
"""

      self.shape=(locations.shape[0],)
      if Mask==None:
          self.Mask=None
      else:
          self.Mask=logical_not(Mask)
          locations=locations[self.Mask,:]
      self.scaling = scaling!=None and scaling or None
      if scaling==None:
          self.locations = locations
      else:
          self.locations = locations*scaling
      if self.scaling==None:
          newLocations = newlocations
      else:
          newLocations = newlocations*self.scaling
      ips=[]
      for p in newLocations:
          if any(getmaskarray(p)):
              ips.append(None)
          else:
              ip=pointInterpolator(self,p,Range=Range,NoP=NoP)
              ips.append(ip)
      self.ips=array(ips)

  def __call__(self,idata,treshold=0.5,fillValue=fv['f4']):

      """Interpolate data on output grid. Data must be flattened and of the same
      size as first dimension of input grid.

      Args:
        idata (float array): flattened input data
        treshold (float): threshold to identify masked points on output grid.
        fillValue (float): valeu applied to masked points on output grid

      Returns:
        data interpolated on output grid (float array).

      Masks of the input data are interpolated from (True) to 0 (False). Any
      point where the interpolated mask results > threshold will be masked on the
      output grid. If the Mask attribute of the instance is defined, the mask
      won't be interpolated, but data will be extrapolated from the input grid
      to all output gridpoints.
      """

      if self.Mask!=None:
          idata=idata[self.Mask]
      data=array([i!=None and i(idata,treshold=treshold) or fillValue  for i in self.ips])
      return masked_where(data==fillValue,data)

class pointInterpolator:

  """Class for interpolator instances that include weights for all points of the
  input grid,

  Attributes:
     positions (integer array): positions in input grid to use for interpolation
     weights (float array): weights for each point of input grid used for
        interpolation
  """

  def __init__(self,iGrid,location,Range=1.,NoP=None):

      """Determines interpolation positions and weights from
      interpolationGrid instance containg the coordinates of the original
      data. Weights and positions are set to False, if no valid position
      is found.

      Arguments:
         iGrid (interpolationGrid): input grid for interpolation
         location (float array): coordinates of point to interpolate to
         Range: distance of points in input grid to points in output grid to be
            included in interpolation
         NoP: Number of points of inputgrids to be included in interpolation for
            each output grid point

      Set Range and NoP to desired values to interpolate on output grid points
      using the NoP closest points or within the radius of Range.
      If both are set, the closest NoP are looked up and then Range is
      applied as multiple of the furthest of those. If NoP is not set, scaling
      should be used to normalise the distances appropriately to Range. If Range
      is set to None, only the NoP closest points will be used.
      """

      ns=[]
      ws=[]
      dz=_distance(iGrid.locations,location)
      if NoP!=None:
        n,w,d=nearestN(dz,NoP=NoP)
        ns.extend(n)
        ws.extend(w)
      if Range!=None:
        if NoP!=None:
          Range= any(d)==0. and Range or sqrt(d[-1])*Range
        n,w=nearestR(dz,Range=Range)
        if any(w):
          ns.extend(n)
          ws.extend(w)
      if len(ws)>0:
        self.positions=array(ns)
        ws=array(ws)
        ws/=ws.sum()
        self.weights=ws
      else:
        self.weights=False
        self.positions=False

  def __call__(self,data,treshold=0.5):

      """Interpolate data on output grid. Data must be flattened and of the same
      size as first dimension of input grid.

      Args:
        data (float array): flattened input data
        treshold (float): threshold to identify masked points on output grid.

      Returns:
        data interpolated on output point (float, None if masked).
      """

      if any(getmaskarray(data)):
          return self._maskedInterpolate(data,treshold)
      else:
         return self._interpolate(data)

  def _interpolate(self,data):
      """Interpolates data on point.
      Args:
        data (float array): input data, needs to be of shape given by instance
            attribute, i. e. of the length of the original coordinate array

      Returns:
        data interpolated to attribute position, None if can't interpolate.
      """

      if any(self.weights):
        return (array(data)[self.positions]*self.weights).sum()
      else:
        return None

  def _maskedInterpolate(self,data,maskTreshold):

      """Interpolates data on point.
      Args:
        data (float array): input data, needs to be of shape given by instance
            attribute, i. e. of the length of the original coordinate array
        threshold(float):  threshold to use for masking using interpolated
        input data mask (0,1)

      Returns:
        data interpolated to attribute position, None if masked.
      """

      mask=logical_not(getmaskarray(data))
      if any(self.weights):
        weights=self.weights*mask[self.positions]
        wsum=weights.sum()
        if wsum<maskTreshold:
           return None
        else:
           weights=weights/wsum
           return (array(data)[self.positions]*weights).sum()
      else:
        return None

_distance = lambda P,p: ((getdata(P) - getdata(p))**2).sum(1)

def nearestN(dz,NoP=3):

      """ Find the NoP positions of minimal distances for intepolation.

      Args:
         dz (float array): 1D-array with distances [No of grid points]
         NoP: Number of points with minimal distances do keep.

      Returns:
        list with positions in input array, weights for interpolation and
        distances for each selected point (list of 1D arrays)
      """

      npos=[]
      dzs=[]
      dz = where(getmaskarray(dz),10.e30,getdata(dz))
      for n in xrange(NoP):
          dzm=dz.min()
          pos=dz.argmin()
          npos.append(pos)
          dzs.append(dzm)
          dz[pos]=10.e30
      dzs=array(dzs)
      if NoP==1:
        weights=array([exp(-exp1)])
      else:
        weights=where(dzs<=0,1.,exp(-dzs/dzs.max()*exp1))
      if weights.sum()<=0: logging.warning("Broken weights: ",npos,weights,dzs)
      return array(npos),weights,dzs

def nearestR(dz,Range=2):

      """Find all points within the Range distance.

      Args:
         dz (float array): 1D-array with distances [No of grid points]
         Range: Maximal distance of inclusion.

      Returns:
        list with positions in input array and weights for interpolation and
        distance for each selected point (list of 1D arrays)
      """

      dz = where(getmaskarray(dz),10.e30,getdata(dz))
      shape=dz.shape[0]
      m=masked_where(dz>Range**2,arange(shape)).compressed()
      if m.sum()>0:
        dz=dz[m]
        weight=where(dz<=0,1.,exp(-dz/dz.max()*exp1))
        npos=arange(shape)[m]
      else:
          npos=False
          weight=False
      #if any(weight==Inf): weight=where(weight==Inf,1,0)
      return npos,weight

class KDinterpolationGrid(KDTree):

  """Defines grid of input data and scaling factors for the various
  dimensions. The geometry needs to be reshaped in 2D
  [No of data points M x No of Dimensions]. Scaling is 1D [Dimensions] and
  provides multipliers to use to weight distance across dimensions.
  Computes look-up table of point interpolators for newlocations
  [K locations x Dimensions].
  Set Range and NoP to desired values to interpolate using the NoP
  closest points or within the radius of Range.
  If both are set, the closest NoP are looked up and then Range is
  applied on the furthest of those. If NoP is not set, scaling should
  be used to normalise the distances appropriately to Range. If Range
  is set to None, only the NoP closest points will be used.
  Returns list of interpolators for locations.

  Attributes:

     shape (tuple of integers): number of input points
     Mask(boolean array): mask of positions to exclude (True) from
        interpolation (e.g. land points in ocean fields)
     scaling (float array): 1-D array with scaling weights for each dimension
        in order to differentiate the impact of distances across dimensions
     locations (float array): 2D array with sequence of point coordinates with
        positions of original grid [Numper of points X Number of dimensions]
     ips (list of pointIterpolators): list of interpolators for each output grid
        point

  Set Range and NoP to desired values to interpolate on output grid points
  using the NoP closest points or within the radius of Range.
  If both are set, the closest NoP are looked up and then Range is
  applied as multiple of the furthest of those. If NoP is not set, scaling
  should be used to normalise the distances appropriately to Range. If Range
  is set to None, only the NoP closest points will be used.
  """

  def __init__(self,locations,newlocations,scaling=None,Mask=None,Range=1.,NoP=None):

      """Defines grid of input data and scaling factors for the various
      dimensions and computes look-up table of point interpolators for newlocations
      [K locations x Dimensions].

      Args:

         locations(float array): point coordinates of input grid. The geometry
            needs to be reshaped in 2D [No of data points M x No of dimensions].
         newlocations(float array): point coordinates of output grid. The
            geometry needs to be reshaped in 2D [No of data points M x No of
            dimensions].
         scaling (float array): 1D [Dimensions] array that provides multipliers
            to use to weight distances across dimensions.
         Mask: mask of positions to exclude (True) from
            interpolation (e.g. land points in ocean fields). This will
            effectively remove the mask before interpolation, so the mask will
            not be interoplated to the output grid. Set to None if you want mask
            interpolation.
         Range: distance of points in input grid to points in output grid to be
            included in interpolation
         NoP: Number of points of inputgrids to be included in interpolation for
            each output grid point
      """

      self.shape=(locations.shape[0],)
      self.Mask=logical_not(Mask)
      if self.Mask!=None:
          locations=locations[self.Mask,:]
      self.scaling = scaling!=None and scaling or None
      if scaling==None:
          self.locations = locations
      else:
          self.locations = locations*scaling
      if self.scaling==None:
          newLocations = newlocations
      else:
          newLocations = newlocations*self.scaling
      KDTree.__init__(self,self.locations)
      ips=[]
      for p in newLocations:
          if any(getmaskarray(p)):
             ips.append(None)
      else:
          ip=KDpointInterpolator(self,p,Range=Range,NoP=NoP)
          ips.append(ip)
      self.ips=array(ips)

  def __call__(self,data):

      """Interpolate data on output grid. Data must be flattened and of the same
      size as first dimension of input grid.

      Args:
        data (float array): flattened input data

      Returns:
        data interpolated on output point (float, None if interpolation fails).
      """

      if any(self.weights):
        return (array(data)[self.positions]*self.weights).sum()
      else:
        return None


class KDpointInterpolator:

  def __init__(self,iGrid,location,Range=1.,NoP=None):

      """Determines interpolation positions and weights from
      interpolationGrid instance containg the coordinates of the original
      data. Weights and positions are set to False, if no valid position
      is found.

      Arguments:
         iGrid (KDInterpolationGrid): input grid for interpolation
         location (float array): coordinates of point to interpolate to
         Range: distance of points in input grid to points in output grid to be
            included in interpolation
         NoP: Number of points of inputgrids to be included in interpolation for
            each output grid point

      Set Range and NoP to desired values to interpolate on output grid points
      using the NoP closest points or within the radius of Range.
      If both are set, the closest NoP are looked up and then Range is
      applied as multiple of the furthest of those. If NoP is not set, scaling
      should be used to normalise the distances appropriately to Range. If Range
      is set to None, only the NoP closest points will be used.
      """

      if NoP!=None:
        d,n=iGrid.query(location,NoP)
      if Range!=None:
        if NoP!=None:
          Range= any(d)==0. and Range or d[-1]*Range
        dr,nr=iGrid.query(location,1000,distance_upper_bound=Range)
        if dr==Inf:
            dr=False
            nr=False
        self.positions=concatenate((n,array(nr)))
        if self.posistions.shape[0]==0:
            ws=array([exp(-exp1)])
        else:
            ws=concatenate(exp(-d**2/d.max()**2*exp1),exp(-dr**2/d.max()**2*exp1))
      else:
        self.positions=array(n)
        logging.info(self.positions)
        if self.positions.shape[0]==0:
            ws=array([exp(-exp1)])
        else:
            ws=exp(-d**2/d.max()**2*exp1)
      ws/=ws.sum()
      self.weights=ws


  def __call__(self,data):

      """Interpolate data on output grid. Data must be flattened and of the same
      size as first dimension of input grid.

      Args:
        data (float array): flattened input data

      Returns:
        data interpolated on output grid (float array).
      """

      return (array(data)[self.positions]*self.weights).sum()


exp1=exp(1)
