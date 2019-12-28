# -*- coding: utf-8 -*-
from imtools import *
import numpy as np
import numpy.fft as fft
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import *
from cmath import *

##Exercice 1: visualisation de fonctions de Sobolev

##1.a Sobolev 1d

##On suppose une décroissance des coefficients de Fourier en (1+k^2)**(-alpha/2).
##La fonction associée est H^m si et seulement si la série (1+k^2)**-(alpha-m) converge: 
##en 1d, cela donne alpha>m+1/2

## Sur l'intervalle [-N/2, N/2-1], les coefficients doivent avoir la décroissance requise: 
## on prolonge ensuite à [N/2, N-1] par périodicité. On bruite alors la phase de chaque 
## coefficient avec une distribution uniforme sur [0,2*pi] et on applique la fft inverse:
## ceci donne une fonction H^m mais à valeur complexe. Il suffit alors d'extraire la partie 
## réelle ou la partie imaginaire pour avoir une fonction H^m à valeur dans R. 
##Pour avoir directement un signal réel, il faudrait ne bruiter la phase que
## sur [1, N/2-1], passer au conjugué et symétriser pour définir le signal sur [-N/2+1,-1] 
## et enfin périodiser.


#N=2**8
#k=np.arange(N+1)
#alpha=0.7
#c1=(1./(1+k**2))**(alpha/2)
#c=np.hstack((c1,np.flipud(c1[1:N])))
#bruit=np.random.uniform(0,1,c.shape)
#a_bruit=numpy.exp(2*1j*numpy.pi*bruit)
#h1=np.imag(fft.ifft(c*a_bruit))
#plt.plot(h1)
#plt.show()

##1. b Sobolev 2d

##On suppose une décroissance des coefficients de Fourier en (1+k^2)**(-alpha/2).
##La fonction associée est H^m si et seulement si la série (1+k^2)**-(alpha-m) converge: 
##en 2d, cela donne alpha>m+1. En dimension d, la condition serait alpha>m+d/2.

#N=2**9
#C=np.zeros((2*N,2*N), dtype=np.complex)
#alpha=1.
#Bruit=np.random.uniform(0,1,(2*N,2*N))
##I=complex(0,1)
#A_bruit=numpy.exp(2*1j*pi*Bruit)
#for i in range(N+1):
#    for k in range(N+1):
#        C[i,k]=(1+i**2+k**2)**(-alpha/2)
#
#C[N+1:2*N-1,0:N]=np.flipud(C[1:N-1,0:N])
#C[:,N+1:2*N-1]=np.fliplr(C[:,1:N-1])
#        
#H1=np.real(fft.ifft2(C*A_bruit)) 
##on pourrait prendre la partie imaginaire, on aurait aussi 
##une représentation d'une fonction H^m. Sinon, il faut
##rajouter des contraintes sur les coefficients de Fourier.
#display_image(H1)

##Exercice 2: gradient, laplacien d'une image 

j=open_image('peppers.png')

def dxp(j): #dérivée à droite selon x
    N=j.shape[0]
    jr=np.append(j[:,1:N], j[:,0:1], axis=1)
    return jr-j
    
def dxm(j): #dérivée à gauche selon x
    N=j.shape[0]
    jl=np.append(j[:,N-1:N], j[:,0:N-1], axis=1)
    return j-jl 
    
def dyp(j): #dérivée à droite selon x
    N=j.shape[0]
    ju=np.append(j[1:N,:], j[0:1,:], axis=0)
    return ju-j
    
def dym(j): #dérivée à gauche selon y
    N=j.shape[0]
    jd=np.append(j[N-1:N,:], j[0:N-1,:], axis=0)
    return j-jd           
    
def lap(j): #laplacien linéaire
    return dym(dyp(j))+dxm(dxp(j))
    
## Max(i,0)
def MAX(j):
    return (j+np.abs(j))/2.
    
def lapROF(j,eps): #laplacien methode ROF
    G=(MAX(dxm(j))**2+MAX(-dxp(j))**2+MAX(dym(j))**2+MAX(-dyp(j))**2+eps)**(1/2.)#gradient
    return dym(dyp(j)*G**(-1))+dxm(dxp(j)*G**(-1))   
    
## Bruitage d'une image
def bruitage(i,sigma):
    N=i.shape[0]
    Bruit=np.random.normal(0,sigma,(N,N))
    return i+Bruit

#condition initiale
u0=bruitage(j,30)

#Equation de la chaleur  
#(Rq: j'ai rajouté un terme de relaxation pour garder 
#trace du fait qu'on veut rester assez proche de l'image originale)
dt=0.2
T=50.
Niter=np.floor(T/dt)

##Test 1: chaleur isotrope        
compteur=0
uc=u0
lam=0.01#parametre de relaxation

while compteur<Niter:
    uc=uc+dt*lap(uc)+lam*dt*(u0-uc)
    compteur=compteur+1
    
plt.subplot(2,2,1)
display_image(u0)
plt.title('image bruitee')
plt.subplot(2,2,2)    
display_image(uc)
plt.title('debruitee avec equation de la chaleur') 

#Test2: chaleur anisotrope modele ROF
compteur=0
uROF=u0
lam=0.01#parametre de relaxation
eps=0.5 #evite la division par zero dans le coefficient de diffusion
while compteur<Niter:
    uROF=uROF+dt*lapROF(uROF,eps)+lam*dt*(u0-uROF)
    compteur=compteur+1      
    
plt.subplot(2,2,4)
display_image(j)
plt.title('image originale')
plt.subplot(2,2,3)    
display_image(uROF)     
plt.title('debruitage avec modele ROF')
    
    









