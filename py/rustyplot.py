###
#
# Rusty plot driver
#
###
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from scipy.integrate import quad
import pumanoise as pn




def cosmology():
    return ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8)

def plotRusty (C,bfunc, nbarfunc, zmin, zmax,
               Ptfunc=None, toplot='Pn', kweight=1,
               kmin=5e-3, kmax=1.0, Nk=90, Nz=100,
               vmin=None, vmax=None,
               mu=0.5, plotlog=True):
    """ Evertying should be self-explanatory, except that
     *** WE WORK IN CCL UNITS, so nbar is in 1/Mpc^3 ***

    toplot can be Pn or SNR (Pn/(1+Pn)) or SNR2

    Ptfucn returns thermal noise, set to none for gals, takes (C,z,k)
    
    however kmin and kmax are in Mpc/h.

    """
    h=C['h']
    k_edges=np.logspace(np.log10(kmin*h), np.log10(kmax*h), Nk+1)
    z_edges=np.linspace(zmin,zmax,Nz+1)
    ks=np.sqrt(k_edges[:-1]*k_edges[1:]) ## log spacing
    zs=0.5*(z_edges[:-1]+z_edges[1:])
    hmap=np.zeros((Nk,Nz))
    kw=ks**kweight
    for zi,z in enumerate(zs):
        Pk=ccl.nonlin_matter_power(C,ks,1/(1+z))
        f=ccl.growth_rate(C,1/(1+z))
        bias=bfunc(z)
        nbar=nbarfunc(z)

        PkUse = (bias + f*mu**2)**2 * Pk
        Pnoise=1/nbar
        if Ptfunc is not None:
            Pnoise+=np.array([Ptfunc(C,z,kx) for kx in ks])
        Pn= PkUse/Pnoise
        SNR = Pn/(1+Pn)

        
        
        if (toplot=='Pn'):
            hmap[:,zi]=Pn if plotlog else Pn*kw
        elif (toplot=='SNR'):
            hmap[:,zi]=SNR
        elif (toplot=='SNR2'):
            hmap[:,zi]=SNR**2
        else:
            print ("Bad toplot!")
            stop()
            
    #plt.imshow (hmap, origin='lower', extent=(kmin,kmax, zmin,zmax),vmin=vmin,vmax=vmax)
    K,Z = np.meshgrid(ks,zs)
    if plotlog:
        plt.pcolor(K/h,Z,hmap.T,norm=LogNorm(),vmin=vmin,vmax=vmax)
    else:
        plt.pcolor(K/h,Z,hmap.T,vmin=vmin,vmax=vmax)
    plt.xscale('log')
    plt.colorbar()

def DESIParams(C):
    ### the following snippet stolen from PkSNR.py in unimap
    ## cut start
    h=C['h']
    z,_,_,_,_,_,V,nelg,nlrg,nqso,_,_=np.loadtxt('desi.dat',unpack=True)
    V*=1e9 ## to (Gpc/h)^3
    nelg*=0.1*14e3/V ## now in num/(Mpc/h)^3, 0.1 for dz=0.1
    nlrg*=0.1*14e3/V
    nqso*=0.1*14e3/V
    belg=0.84/ccl.growth_factor(C,1./(1.+z))
    blrg=1.7/ccl.growth_factor(C,1./(1.+z))
    bqso=1.2/ccl.growth_factor(C,1./(1.+z))
    ## --- cut end
    ### let's use ELGs
    biasfunc = interp1d(z,belg,bounds_error=False, fill_value='extrapolate')
    nbarfunc = interp1d(z,nelg*h**3,bounds_error=False, fill_value='extrapolate')
    Ptfunc=None
    zmin=0.1
    zmax=1.8
    return biasfunc, nbarfunc, zmin, zmax, Ptfunc


def LSSTSpecParams(C):
    biasfunc=lambda z:0.95/ccl.growth_factor(C,1/(1+z))
    ndens=49 ## per /arcmin^2, LSST SRD, page 47
    dndz=lambda z: z**2*np.exp(-(z/0.28)**0.94) ## LSST SRD, page 47
    arcminfsky = 1/ (4*np.pi/(np.pi/(180*60))**2)
    ## volume between z=3
    zmax=3
    V=4*np.pi**3/3 * ccl.comoving_radial_distance(C,1/(1+zmax))**3
    dVdz = lambda z: 3e3/C['h'] * 1/ccl.h_over_h0(C,1/(1+z)) * 4*np.pi*ccl.comoving_radial_distance(C,1/(1+z))**2
    norm = ndens/(quad(dndz, 0,zmax)[0]*arcminfsky)
    nbarofz = lambda z: norm*dndz(z)/dVdz(z)
    return biasfunc, nbarofz, 0,3,None

def PUMAParams(C):
    global puma
    puma=pn.PUMA(C)
    nbarofz = lambda z: 1/puma.PNoiseShot(z,1.0)
    ## assuming mu=0.5 and kpermin=0.05
    def PtFunc(C,z,k):
        noise = puma.PNoise(z,k*np.cos(0.5))/puma.Tb(z)**2
        kfg=0.01*0.07 # 0.01 h/Mpc in /Mpc
        if k*np.sqrt(1-0.5**2)<kfg:
            #noise=1e30
            noise+=np.exp((k-kfg)**2/1e-8)
        print (z,k,noise)
        return noise
    zmin=0.3
    zmax=6.0
    return puma.bias, nbarofz, zmin,zmax,PtFunc


