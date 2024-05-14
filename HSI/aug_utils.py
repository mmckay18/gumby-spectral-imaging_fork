import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

class CosineSimAug(object):
    """ applies augmentation with probability p (default=0.3)
    aug is applied randomly to pixels with prob spec_p (default=0.5)
    force_val: if True, it will set degree to max_degree, rather than a random int
    """
    def __init__(self, max_deg:int=6, p:float=0.3, spec_p:float=0.5, force_val=False):
        self.rng = np.random.default_rng()
        self.max_deg = max_deg
        self.p = p
        self.spec_p = spec_p
        self.force_val = force_val
    
    def __call__(self, u):
        # only add augmentation with prob=p
        if self.rng.random(1) < self.p:
            # use a fixed value or some random integer degree
            if self.force_val:
                theta = self.max_deg * np.pi/180
            else:
                #rng.integers is not inclusive of second value
                theta = self.rng.integers(1,self.max_deg+1) * np.pi/180
            
            spec_mask = self.rng.random(u.shape[0:-1]) <= self.spec_p
            # vector to build basis from
            b = self.rng.random(u.shape)
            #creating the other part of basis, h, which is orthogonal to the given u
            h = b - (((u*b).sum(-1))/((u*u).sum(-1)))[...,np.newaxis]*u
            #need to have unit vectors so finding length of u and h
            norm_u = np.sqrt((u*u).sum(-1))
            norm_h = np.sqrt((h*h).sum(-1))
            #build the actual unit vectors of the orthogonal basis to be orthonormal basis
            unit_u = u / norm_u[...,np.newaxis]
            unit_h = h / norm_h[...,np.newaxis]
            #build the new vector that will be theta away from u
            v = np.cos(theta)*unit_u + np.sin(theta)*unit_h
            #scaling it back up as likely don't want just the unit vector in this direction, multiply by norm of given u
            output = norm_u[...,np.newaxis]*v
            # fixes nan division
            output[np.isnan(output)] = u[np.isnan(output)]
            # fills original data back in 
            output[~spec_mask] = u[~spec_mask]
        else:
            output = u
        return output

class WhiteGaussianNoiseAug(object):
    """ applies white guassian noise augmentation with probability p (default=0.3)
    noise is applied randomly to spaxels with prob spec_p (default=0.5)
    gaussian noise scale is percent (default 15%)
    force_val=True will fix the AWG to max_perc
    """
    def __init__(self, max_perc:float=0.15, p:float=0.3, spec_p:float=0.5, force_val=False):
        self.rng = np.random.default_rng()
        self.max_perc = max_perc
        self.p = p
        self.spec_p = spec_p
        self.force_val = force_val
    
    def __call__(self, u):
        if self.rng.random(1) < self.p:
            # select random percentage, unless using fixed value
            if self.force_val:
                perc = self.max_perc
            else:
                perc = self.rng.uniform(low=0.0, high=self.max_perc)
            
            spec_mask = self.rng.random(u.shape[0:-1]) <= self.spec_p
            # noise patch
            noise = self.rng.normal(0, u.std(), size=u.shape) * perc
            output = u + noise
            # fills original data back in 
            output[~spec_mask] = u[~spec_mask]
        else:
            output = u
        return output
    
class PeakShiftAug(object):
    """applies augmentation of shifting random single peak left or right with probability p (default=0.3)
    aug is applied randomly to pixels with prob spec_p (default=0.5)
    """
    def __init__(self, prominence:float=0.03, p:float=0.3, spec_p:float=0.5):
        self.rng = np.random.default_rng()
        self.prominence = prominence
        self.p = p
        self.spec_p = spec_p

    def __call__(self,u):
        if self.rng.random(1) < self.p:
            #do the augmentation
            #flatten so can go through one by one, vectorize difficult since diff number peaks for each spectrum
            u_flat = u.reshape(u.shape[0]*u.shape[1], u.shape[-1])
            #array to catch each augmentation
            temp = []
            for v in u_flat:
                #creating the new one, copy so don't overwrite original
                outputu = np.copy(v)
                #find the peaks using scipy
                peaks, properties = find_peaks(v, prominence = self.prominence)
                #check to see if found peaks, if did then peak swap, if not just append v
                if (len(peaks)>0):
                    #randomly pick which peak to shift
                    slot = self.rng.integers(0,len(peaks))
                    
                    #pick one of two numbers, determine whether shift left or right, picks 0 or 1, 2 not included in rng.integers
                    chance = self.rng.integers(0,2)
                    if chance == 0:
                        #moving the peak to the right one bin
                        outputu[peaks[slot]+1] = v[peaks[slot]]
                        #filling in the gap with interpolated to not create two peaks
                        outputu[peaks[slot]] = (v[peaks[slot]] + v[peaks[slot]-1])/2
                    else:
                        #moving the peak to the left one bin
                        outputu[peaks[slot]-1] = v[peaks[slot]]
                        #filling in the gap with interpolated to not create two peaks
                        outputu[peaks[slot]] = (v[peaks[slot]] + v[peaks[slot]+1])/2  
                    temp.append(outputu)
                else:
                    temp.append(v)
            output = np.array(temp).reshape(u.shape)
        else:
            #no augmentation so output input
            output = u
        return output
    
class TranslateUpAug(object):
    """applies augmention of shifting spectrum up with probability p (default=0.3)
    aug is applied randomly to pixels with prob spec_p (default=0.5)
    max_val is the maximum value for the data set such as reflectance would be 1
    """
    def __init__(self, max_val:float=1, p:float=0.3, spec_p:float=0.5):
        self.rng = np.random.default_rng()
        self.max_val = max_val
        self.p = p
        self.spec_p = spec_p

    def __call__(self,u):
        if self.rng.random(1) < self.p:
            #do the augmentation
            #creating the new one, copy so don't overwrite original
            output = np.copy(u)
            #getting max entry value in the spectrum
            max_spec = u.max()

            #checking if max entry in spectrum is greater than or equal to desired max value
            #if is then return original
            if max_spec >= self.max_val:
                output = u
            #if have room to translate up, then do following
            else:
                shift = self.rng.uniform(low=0,high=self.max_val-max_spec)
                for i in range(len(output)):
                    output[i] += shift
        else:
            #no augmentation
            output = u
        return output
    
class TranslateDownAug(object):
    """applies augmention of shifting spectrum down with probability p (default=0.3)
    aug is applied randomly to pixels with prob spec_p (default=0.5)
    min_val is the minimum value for the data set such as reflectance would be 0
    """
    def __init__(self, min_val:float=0, p:float=0.3, spec_p:float=0.5):
        self.rng = np.random.default_rng()
        self.min_val = min_val
        self.p = p
        self.spec_p = spec_p

    def __call__(self,u):
        if self.rng.random(1) < self.p:
            #do the augmentation
            #creating the new one, copy so don't overwrite original
            output = np.copy(u)
            #getting max entry value in the spectrum
            min_spec = u.min()

            #checking if min entry in spectrum is less than or equal to desired min value
            #if is then return original
            if min_spec <= self.min_val:
                output = u
            #if have room to translate down, then do following
            else:
                shift = self.rng.uniform(low=0,high=min_spec-self.min_val)
                for i in range(len(output)):
                    output[i] -= shift
        else:
            #no augmentation
            output = u
        return output
    
class CompressAug(object):
    """applies augmention of compressing random part of spectrum down with probability p (default=0.3)
    aug is applied randomly to pixels with prob spec_p (default=0.5)
    min_com and max_com are the min and max compression factors (0.8 to 0.99)
    good for LWIR since random part of spectrum compressed, not whole spectrum
    """
    def __init__(self, min_com:float=0.8, max_com:float=0.99, p:float=0.3, spec_p:float=0.5):
        self.rng = np.random.default_rng()
        self.min_com = min_com
        self.max_com = max_com
        self.p = p
        self.spec_p = spec_p 

    def __call__(self,u):
        if self.rng.random(1) < self.p:
            #flatten so can go through one by one, vectorize difficult since diff number peaks for each spectrum
            u_flat = u.reshape(u.shape[0]*u.shape[1], u.shape[-1])
            #array to catch each augmentation
            temp = []
            for v in u_flat:
                #creating the new one, copy so don't overwrite original
                spec = np.copy(v)
                #find peaks and troughs, start between peak and trough, end between trough and peak
                peaks, propertiesp = find_peaks(spec)
                trough, properties = find_peaks(-spec)
                if (len(peaks)>0):
                    #pick a random peak
                    min_peak = self.rng.integers(0,len(peaks)-4)
                    #find the trough to the right of that min peak
                    min_trough = np.searchsorted(trough, peaks[min_peak], side='right')
                    #pick randomly the bin from that peak to trough
                    min_bin = self.rng.integers(peaks[min_peak],trough[min_trough])
                    #pick random trough to the right of the min trough
                    max_trough = self.rng.integers(min_trough+1,len(trough)-2)
                    #find the peak to the right of that max trough
                    max_peak = np.searchsorted(peaks,trough[max_trough],side='right')
                    #pick randomly the bin from that trough to peak max
                    max_bin = self.rng.integers(trough[max_trough],peaks[max_peak])
                    #length of compression, need to add 1 if total number of bins
                    len_scale = max_bin-min_bin

                    #make numpy array of options for compressing factor, start, stop, step
                    #play around with data first to determine what ranges are appropriate for start, stop
                    scale_opt = np.arange(self.min_com,self.max_com,0.025)
                    scale = self.rng.choice(scale_opt)
                    
                    #make new spectrum, using spline interpolation
                    if ((min_bin >= 5) and (max_bin+5 <= len(v))):
                        #do the spline
                        #build compressed section
                        compressed = np.copy(v)
                        for i in range (min_bin,max_bin+1):
                            compressed[i] = v[i]*scale
                        #build the spline function x and y, which includes 5 more each side
                        interp_x = np.linspace(min_bin-5,max_bin+5,len_scale+11)
                        #should be 5 before of original, then min bin to max bin in middle, then 5 after
                        interp_y = np.concatenate([v[min_bin-5:min_bin],compressed[min_bin:max_bin+1],v[max_bin+1:max_bin+6]])
                        #print('len of interp_y: ', len(interp_y))
                        spline = CubicSpline(interp_x,interp_y,bc_type='natural')
                        smooth_compress = spline(np.linspace(min_bin,max_bin,len_scale+1))
                        spec = np.concatenate([v[:min_bin],smooth_compress,v[max_bin+1:]])
                        temp.append(spec)
                    else:
                        #do we want to do interpolation just on range or just return original
                        for i in range (min_bin,max_bin+1):
                            spec[i] = v[i]*scale
                        temp.append(spec)
                else:
                    temp.append(v)
            output = np.array(temp).reshape(u.shape)
        else:
            #no augmentation
            output = u
        return output 
    
class StretchAug(object):
    """applies augmention of stretching random part of spectrum down with probability p (default=0.3)
    aug is applied randomly to pixels with prob spec_p (default=0.5)
    min_stre and max_stre are the min and max compression factors (1.025 to 1.2)
    good for LWIR since random part of spectrum compressed, not whole spectrum
    """
    def __init__(self, min_stre:float=1.025, max_stre:float=1.2, p:float=0.3, spec_p:float=0.5):
        self.rng = np.random.default_rng()
        self.min_stre = min_stre
        self.max_stre = max_stre
        self.p = p
        self.spec_p = spec_p 

    def __call__(self,u):
        if self.rng.random(1) < self.p:
            #flatten so can go through one by one, vectorize difficult since diff number peaks for each spectrum
            u_flat = u.reshape(u.shape[0]*u.shape[1], u.shape[-1])
            #array to catch each augmentation
            temp = []
            for v in u_flat:
                #creating the new one, copy so don't overwrite original
                spec = np.copy(v)
                #find peaks and troughs, start between trough and peak, end between peak and trough
                trough, properties = find_peaks(-spec)
                peaks, propertiesp = find_peaks(spec)
                if (len(peaks)>0):
                    #pick a random trough
                    min_trough = self.rng.integers(0,len(trough)-4)
                    #find the peak to the right of that min trough
                    min_peak = np.searchsorted(peaks, trough[min_trough], side='right')
                    #pick randomly the bin from that trough to peak
                    min_bin = self.rng.integers(trough[min_trough],peaks[min_peak])
                    #pick random peak to the right of the min peak
                    max_peak = self.rng.integers(min_peak+1,len(peaks)-2)
                    #find the trough to the right of that max peak
                    max_trough = np.searchsorted(trough,peaks[max_peak],side='right')
                    #pick randomly the bin from that peak to trough max
                    max_bin = self.rng.integers(peaks[max_peak],trough[max_trough])
                    #length of compression, need to add 1 if total number of bins
                    len_scale = max_bin-min_bin

                    #make numpy array of options for stretching factor, start, stop, step
                    #play around with data first to determine what ranges are appropriate for start, stop
                    scale_opt = np.arange(self.min_stre,self.max_stre,0.025)
                    scale = self.rng.choice(scale_opt)
                    
                    #make new spectrum, using spline interpolation
                    if ((min_bin >= 5) and (max_bin+5 <= len(v))):
                        #do the spline
                        #build stretched section
                        stretched = np.copy(v)
                        for i in range (min_bin,max_bin+1):
                            stretched[i] = v[i]*scale
                        #build the spline function x and y, which includes 5 more each side
                        interp_x = np.linspace(min_bin-5,max_bin+5,len_scale+11)
                        #should be 5 before of original, then min bin to max bin in middle, then 5 after
                        interp_y = np.concatenate([v[min_bin-5:min_bin],stretched[min_bin:max_bin+1],v[max_bin+1:max_bin+6]])
                        #print('len of interp_y: ', len(interp_y))
                        spline = CubicSpline(interp_x,interp_y,bc_type='natural')
                        smooth_stretch = spline(np.linspace(min_bin,max_bin,len_scale+1))
                        spec = np.concatenate([v[:min_bin],smooth_stretch,v[max_bin+1:]])
                        temp.append(spec)
                    else:
                        #do we want to do interpolation just on range or just return original
                        for i in range (min_bin,max_bin+1):
                            spec[i] = v[i]*scale
                        temp.append(spec)
                else:
                    temp.append(v)
            output = np.array(temp).reshape(u.shape)
        else:
            #no augmentation
            output = u
        return output 

class SmoothGaussAug(object):
    """Smooths input spectra with 1D gaussian kernel of width sigma.
    applies augmention of with probability p (default=0.3)
    aug is applied randomly to pixels with prob spec_p (default=0.5)
    
    Inputs:
        sigma (default=2.0): standard deviation of 1d gaussian kernel
            default behavior uses radius=8 and truncates at 4 stdev
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
        if force_val=True, will use max_sigma.
        p (default=0.3): probability of applying augmentation to input
        spec_p (default=0.5)
    """
    def __init__(self, min_sigma:float=1.0, max_sigma:float=5.0, 
                 force_val:bool=False, p:float=0.3, spec_p:float=0.5):
        self.rng = np.random.default_rng()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.force_val = force_val
        self.p = p
        self.spec_p = spec_p

    def __call__(self,u):
        # only add augmentation with prob=p
        if self.rng.random(1) < self.p:
            # use a fixed value
            if self.force_val:
                sigma = self.max_sigma
            else:
                sigma = self.rng.uniform(low=self.min_sigma, high=self.max_sigma)
            output = gaussian_filter1d(
                u,
                sigma,
                axis=-1,
                mode='nearest',
            )
        else:
            #no augmentation so output input
            output = u
        return output


augmentation_fn_dict = {
    'cosine': CosineSimAug(max_deg=5, p=0.3, spec_p=1.0),
    'awg': WhiteGaussianNoiseAug(max_perc=0.15, p=0.3, spec_p=1.0),
    'peak':PeakShiftAug(prominence=0.03, p=1.0, spec_p=1.0),
    'trsltup':TranslateUpAug(max_val=1.1, p=0.3, spec_p=1.0),
    'trsltdown':TranslateDownAug(min_val=0, p=0.3, spec_p=1.0),
    'compress':CompressAug(min_com=0.8, max_com=0.99, p=0.3, spec_p=1.0),
    'stretch':StretchAug(min_stre=1.025, max_stre=1.2, p=0.3, spec_p=1.0),
    'smooth':SmoothGaussAug(min_sigma=1, max_sigma=5, p=0.3, spec_p=1.0),
    'all':[
        CosineSimAug(max_deg=5, p=0.3, spec_p=1.0),
        PeakShiftAug(prominence=0.03, p=1.0, spec_p=1.0),
        TranslateUpAug(max_val=1.1, p=0.3, spec_p=1.0),
        TranslateDownAug(min_val=0, p=0.3, spec_p=1.0),
        CompressAug(min_com=0.8, max_com=0.99, p=0.3, spec_p=1.0),
        StretchAug(min_stre=1.025, max_stre=1.2, p=0.3, spec_p=1.0),
        SmoothGaussAug(min_sigma=1, max_sigma=5, p=0.3, spec_p=1.0)
    ],
    'cossmo':[
        CosineSimAug(max_deg=5, p=0.3, spec_p=1.0), 
        SmoothGaussAug(min_sigma=1, max_sigma=5, p=0.3, spec_p=1.0)
    ],
    'strcom':[
        StretchAug(min_stre=1.025, max_stre=1.2, p=0.3, spec_p=1.0), 
        CompressAug(min_com=0.8, max_com=0.99, p=0.3, spec_p=1.0)
    ],
    'peatup':[
        PeakShiftAug(prominence=0.03, p=1.0, spec_p=1.0), 
        TranslateUpAug(max_val=1.1, p=0.3, spec_p=1.0)
    ],
}