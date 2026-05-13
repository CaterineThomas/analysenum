# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:48:42 2026

@author: tcatherine01
"""

import random
import matplotlib.pyplot as plt
import numpy as np


#1
def balayage_pas_constant (f, a, b, N):
    d = (b-a)/N
    x=a
    min=x
    minval=f(min)
    x+=d
    while x<=b :
        temp=f(x)
        if temp<minval :
            min=x
            minval=temp
        x+=d
    return min

def balayage_aleatoire (f, a, b, N):
    valeurs = []
    for k in range(1, N):
        valeurs.append(random.random()*(b-a))
    min=a
    minval=f(min)
    for v in valeurs :
        temp = f(v)
        if temp<=minval :
            min=v
            minval=temp
    return min

def f_test (x) :
    return x*x*x-3*x*x +2*x  + 5

print("Minimum de f d'après le balayage à pas constant est atteint en " + str(balayage_pas_constant(f_test, 0, 3, 1000)))

print("Minimum de f d'après le balayage aléatoire est atteint en " + str(balayage_aleatoire(f_test, 0, 3, 1000)))


#3
def plot_courbe_err (plot, bal, nmin, nmax, lab):
    #TODO:fix labelling
    expec=1+np.sqrt(3)/3
    err=[]
    for n in range(nmin, nmax):
        err.append(abs(expec-bal(f_test, 0, 3, n))/expec)
    plot.plot(err, label=lab)
    
plot_courbe_err(plt, balayage_pas_constant, 2, 500, 'balayage à pas constant')
plot_courbe_err(plt, balayage_aleatoire, 2, 500, 'balayage aléatoire')
plt.ylabel("erreur relative du résultat (log)")
plt.xlabel("nombre de valeurs calculées")
plt.yscale('log')
plt.grid()
plt.show()

max_f_test = balayage_pas_constant(lambda x : -f_test(x), 0, 3, 1000)
print("Maximum de f d'après le balayage à pas constant :" + str(f_test(max_f_test)) + " en " + str(max_f_test))

def gradient(df, a, b, u, eps):
    x=(b-a)/2
    d=u*df(x)
    while abs(d)>eps :
        x=x+d
        d=u*df(x)
    x=x+d
    return x

def df_test(x):
    return 3*x*x - 6*x + 2


print("Minimum de f d'après le gradient 1D est atteint en " + str(gradient(df_test, 0, 3, -0.001, 0.0001)))
