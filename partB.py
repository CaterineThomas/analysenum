# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:48:42 2026

@author: tcatherine01
"""

import random
import matplotlib.pyplot as plt
import numpy as np


#1
def min_pas_constant (f, a, b, N):
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

def min_pas_aleatoire (f, a, b, N):
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

print("Minimum de f d'après le balayage à pas constant est atteint en " + (lambda x : str(x) + " et vaut " +str(f_test(x)))(min_pas_constant(f_test, 0, 3, 100)))

print("Minimum de f d'après le balayage aléatoire est atteint en " + (lambda x : str(x) + " et vaut " +str(f_test(x)))(min_pas_aleatoire(f_test, 0, 3, 100)))


#3
def plot_courbe_err (plot, bal, nmin, nmax, lab):
    #TODO:fix labelling
    expec=1+np.sqrt(3)/3
    err=[]
    for n in range(nmin, nmax):
        err.append(abs(expec-bal(f_test, 0, 3, n))/expec)
    plot.plot(err, label=lab)
    return err
    
err_cst = plot_courbe_err(plt, min_pas_constant, 2, 500, 'balayage à pas constant')
err_al = plot_courbe_err(plt, min_pas_aleatoire, 2, 500, 'balayage aléatoire')
plt.ylabel("erreur relative du résultat (log)")
plt.xlabel("nombre de valeurs calculées")
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

plt.plot(np.subtract(err_cst, err_al))
plt.ylabel("Erreur du pas constant - pas aléatoire")
plt.xlabel("nombre de valeurs calculées")
plt.grid()
plt.legend()
plt.show()

plot_courbe_err(plt, min_pas_constant, 2, 8000, 'balayage aléatoire')
plt.ylabel("erreur relative du résultat (log)")
plt.xlabel("nombre de valeurs calculées")
plt.yscale('log')
plt.title("phénomène intéressant lorsque N devient très grand")
plt.grid()
plt.legend()
plt.show()

def max_test(N):
    max_f_test = min_pas_constant(lambda x : -f_test(x), 0, 3, N)
    print("Maximum de f d'après le balayage à pas constant avec "+str(N)+" intervalles :" + str(f_test(max_f_test)) + " en " + str(max_f_test))

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
max_test(1000)
