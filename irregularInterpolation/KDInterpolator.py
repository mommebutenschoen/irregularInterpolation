from scipy.spatial import cKDTree as KDT
from numpy import array,exp,where,ones,pi
from pyproj import Proj

#Gaussian yielding 1 at zero distance and most distant point at 3 standard deviations.
distanceFunction = lambda w: exp(-(3.*w/w.max())**2)

class KDInterpolator(KDT):
    """Interpolator for arbitrarily distributed in and output data. Should be used with unmasked 1D input arrays."""

    def __init__(self,coords,scales,*opts,**ks):
        """coords: 1D array of coordinate tuples
           scales: tuple of dimension weights"""
        KDT.__init__(self,coords*scales,*opts,**ks)
        self.scales=scales

    def __call__(self,inData,outcoords,k=5,*opts,**ks):
        data=[]
        for p in outcoords:
            w,c=self.query(p*self.scales,k=k,*opts,**ks)
            if w.sum()!=0.:
                w=distanceFunction(w)
                w/=w.sum()
            else:
                w=ones(w.shape)
                w/=w.sum()
            data.append((inData[c]*w).sum())
        return array(data)

def KDMask(incoord,scales,inMask,outcoord,lonAxis=None,latAxis=None,crit=.5):
    """Computing interpolated Mask from unmasked input coordinates,
    dimension weights, input mask and ouput coordinates."""
    if lonAxis==None or latAxis==None:
        kd=KDInterpolator(incoord,scales)
    else:
        kd=KDGeographic(incoord,scales,lonAxis,latAxis)
    Mask=kd(1.*inMask,outcoord)
    return where(Mask>=.5,True,False)

class KDGeographic:
    """Lon,lat based interpolation projected on UTM grids to get
    more precise geographic interpolations. KDGeogrphic.interpolator
    contains a list of KDinterpolators."""
    def __init__(self,coords,scales,lonAxis,latAxis,*opts,**ks):
        self.interpolator=[]
        self.Proj=[]
        self.lonAxis=lonAxis
        self.latAxis=latAxis
        for i in xrange(1,61):
            #set up projection for each UTM zone:
            self.Proj.append(Proj(proj='utm',zone=i))
            x,y=self.Proj[-1](coords[:,lonAxis],coords[:,latAxis])
            crdsxy=coords.copy()
            crdsxy[:,lonAxis]=x/111120. #conv back to degree scale for appropriate scaling
            crdsxy[:,latAxis]=y/111120.
            self.interpolator.append(KDInterpolator(crdsxy,scales,*opts,**ks))
    def __call__(self,inData,outcoords,k=5,*opts,**ks):
        data=[]
        for p in outcoords:
            #retrieve ID of right projection:
            utmID=UTMzone(p[self.lonAxis])
            #interpolate:
            x,y=self.Proj[utmID](p[self.lonAxis],p[self.latAxis])
            pxy=array(p).copy()
            pxy[self.lonAxis]=x/111120. #conv back to degree scale for appropriate scaling
            pxy[self.latAxis]=y/111120.
            d=self.interpolator[utmID](inData,array([pxy,]),k=k,*opts,**ks)
            data.append(d.squeeze())
        return array(data)

UTMzone=lambda lon:int((lon+180)%360)/6 #UTMzone index (0-59)
