from scipy.spatial import cKDTree as KDT
from numpy import array,exp,where,ones,pi
from pyproj import Proj
from pdb import set_trace

#Gaussian yielding 1 at zero distance and most distant point at 3 standard deviations.
_distanceFunction = lambda w: exp(-(3.*w/w.max())**2)

class KDInterpolator(KDT):

    """Distance weighted nearest neighbour Interpolator for arbitrarily
    distributed in and output data. Should be used with unmasked 1D input arrays.

    Attributes:
     scales (float array): 1-D array with scaling weights for each dimension
        in order to differentiate the impact of distances across dimensions
    """

    def __init__(self,coords,scales,*opts,**ks):

        """Creates Intrepolator instance from coordinates and dimension scales.

        Args:
           coords (float array-like): array of coordinate tuples [# of points,
              # of dimensions]
           scales (float array-like): tuple of dimension weights [# of
              dimensions]
        """

        KDT.__init__(self,coords*scales,*opts,**ks)
        self.scales=scales

    def __call__(self,inData,outcoords,k=5,fill_value=None,*opts,**ks):

        """Interpolates input data on output grid.

        Args:
           inData (float array):
           outcoords (float array-like): array of output coordinate tuples
              [# of points, # of dimensions]
           k (integer): number of nearest points to consider
           fill_value (float): input data of this value will be ignored for
              interpolation, used for invalid points in interpolation output
           **opts: positional arguments passed to scipy.spatial.cKDTree.query
              function
           *ks: keyword arguments passed to scipy.spatial.cKDTree.query
              function

        Returns:
           interpolated data (float array).
        """

        data=[]
        for p in outcoords:
            w,c=self.query(p*self.scales,k=k,*opts,**ks)
            indata=inData[c]
            #convert distances to weights
            if w.sum()!=0.:
                w=_distanceFunction(w)
            else:
                w=ones(w.shape)
            if fill_value!=None:
               mask=where(indata==fill_value,0,1)
               if not mask.sum():
                  data.append(fill_value)
               else:
                  w*=mask
                  w/=w.sum()
                  data.append((indata*w).sum())
            else:
               w/=w.sum()
               data.append((indata*w).sum())
        return array(data)

def KDMask(incoord,scales,inMask,outcoord,lonAxis=None,latAxis=None,crit=.5):

    """Computes interpolated mask using KDInterpolator or KDGeographic.

    Args:
        incoord (float array-like): array of coordinate tuples [# of points,
           # of dimensions]
        scales (float array-like): tuple of dimension weights [3 of
           dimensions]
        inData (float array):
        outcoords (float array-like): array of output coordinate tuples
           [# of points, # of dimensions]
        crit (float): mask treshold for interpolated value.

    Returns:
        interpolated mask value for each output grid point (float array).

    """
    if lonAxis==None or latAxis==None:
        kd=KDInterpolator(incoord,scales)
    else:
        kd=KDGeographic(incoord,scales,lonAxis,latAxis)
    Mask=kd(1.*inMask,outcoord)
    return where(Mask>=.5,True,False)

class KDGeographic:

    """Lon,lat based interpolation projected on UTM grids to get
    more precise geographic interpolations. KDGeographic.interpolator
    contains a list of KDinterpolators.

    Attributes:
       interpolator (list of KDInterpolators): interpolators for each UTM zone
       Proj (list of pyproj.Proj) projection instances for each UTM UTMzone
       lonAxis: position of longitude dimension in input coordinates
       latAxis: position of latitude dimension in input coordinates
    """

    def __init__(self,coords,scales,lonAxis,latAxis,*opts,**ks):
        """Collects interpolators for input coordinates.

        Args:
           coords (float array-like): array of coordinate tuples [# of points,
              # of dimensions]
           scales (float array-like): tuple of dimension weights [# of
              dimensions]
           lonAxis: position of longitude dimension in input coordinates
           latAxis: position of latitude dimension in input coordinates
           **opts: positional arguments passed to scipy.spatial.cKDTree.query
              function
           *ks: keyword arguments passed to scipy.spatial.cKDTree.query
              function
        """

        self.interpolator=[]
        self.Proj=[]
        self.lonAxis=lonAxis
        self.latAxis=latAxis
        for i in range(1,61):
            #set up projection for each UTM zone:
            self.Proj.append(Proj(proj='utm',zone=i))
            x,y=self.Proj[-1](coords[:,lonAxis],coords[:,latAxis])
            crdsxy=coords.copy()
            crdsxy[:,lonAxis]=x/111120. #conv back to degree scale for appropriate scaling
            crdsxy[:,latAxis]=y/111120.
            self.interpolator.append(KDInterpolator(crdsxy,scales,*opts,**ks))
        #Universal Polar Stereographic (North)
        self.Proj.append(Proj(proj='ups'))
        x,y=self.Proj[-1](coords[:,lonAxis],coords[:,latAxis])
        crdsxy=coords.copy()
        crdsxy[:,lonAxis]=x/111120. #conv back to degree scale for appropriate scaling
        crdsxy[:,latAxis]=y/111120.
        Mask=zeros(len(crdsxy))
        nearest=( p*(1.e36,(0,0)) )
        for p in crdsxy
        self.interpolator.append(KDInterpolator(crdsxy,scales,*opts,**ks))
        #Universal Polar Stereographic (North)
        self.Proj.append(Proj(proj='ups',south=True))
        x,y=self.Proj[-1](coords[:,lonAxis],coords[:,latAxis])
        crdsxy=coords.copy()
        crdsxy[:,lonAxis]=x/111120. #conv back to degree scale for appropriate scaling
        crdsxy[:,latAxis]=y/111120.
        self.interpolator.append(KDInterpolator(crdsxy,scales,*opts,**ks))

    def __call__(self,inData,outcoords,k=5,*opts,**ks):

        """Interpolates input data on output grid.

        Args:
           inData (float array):
           outcoords (float array-like): array of output coordinate tuples
              [# of points, # of dimensions]
           k (integer): number of nearest points to consider
           **opts: positional arguments passed to scipy.spatial.cKDTree.query
              function
           *ks: keyword arguments passed to scipy.spatial.cKDTree.query
              function

        Returns:
           interpolated data (float array).
        """

        data=[]
        for p in outcoords:
            #retrieve ID of right projection:
            if p[self.latAxis]>84.:
                utmID=-2
            elif p[self.latAxis]<-80.:
                utmID=-1
            else:
                utmID=_UTMzone(p[self.lonAxis])
            #interpolate:
            x,y=self.Proj[utmID](p[self.lonAxis],p[self.latAxis])
            pxy=array(p).copy()
            pxy[self.lonAxis]=x/111120. #conv back to degree scale for appropriate scaling
            pxy[self.latAxis]=y/111120.
            d=self.interpolator[utmID](inData,array([pxy,]),k=k,*opts,**ks)
            data.append(d.squeeze())
        return array(data)

_UTMzone=lambda lon:int((lon+180)%360)//6 #UTMzone index (0-59)
