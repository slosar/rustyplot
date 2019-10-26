#!/usr/bin/env python
import rustyplot as rp
import matplotlib.pyplot as plt
import pyccl as ccl

C=rp.cosmology()
plt.figure(figsize=(10,10))
c=0
for q in ['Pn','SNR']:
    plt.subplot(2,3,1+c)
    rp.plotRusty(C,*rp.DESIParams(C),toplot=q)
    plt.title('DESI '+q)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('z')
    plt.subplot(2,3,2+c)
    rp.plotRusty(C,*rp.LSSTSpecParams(C),toplot=q)
    plt.title('LSSTSpec '+q)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('z')

    plt.subplot(2,3,3+c)
    rp.plotRusty(C,*rp.PUMAParams(C),toplot=q)
    plt.title('PUMA '+q)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('z')


    c+=3


plt.show()



