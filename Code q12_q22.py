# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:56:19 2026

@author: youss
"""


import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Fonctions communes : g, h et leurs gradients
# ============================================================

# g_{a,b}(x,y) = x²/a + y²/b
def g(a, b):
    return lambda x, y: (x**2)/a + (y**2)/b

# dérivée partielle de g par rapport à x
def dg_dx(a, b):
    return lambda x, y: 2*x/a

# dérivée partielle de g par rapport à y
def dg_dy(a, b):
    return lambda x, y: 2*y/b

# h(x,y) = cos(x) sin(y)
def h(x, y):
    return np.cos(x) * np.sin(y)

# dérivée partielle de h par rapport à x
def dh_dx(x, y):
    return -np.sin(x) * np.sin(y)

# dérivée partielle de h par rapport à y
def dh_dy(x, y):
    return np.cos(x) * np.cos(y)

# norme du gradient ||grad f(x,y)||
def norme_grad(x, y, df1, df2):
    return np.sqrt(df1(x, y)**2 + df2(x, y)**2)


# ============================================================
# Fonction graphique commune : courbes de niveau + itérations
# ============================================================

def afficher_contours_et_iterations(f, Xn, Yn, xmin, xmax, ymin, ymax, titre):
    x = np.linspace(xmin, xmax, 300)
    y = np.linspace(ymin, ymax, 300)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=25)
    plt.plot(Xn, Yn, marker='o', markersize=3, linewidth=1.2, label='Itérations')
    plt.scatter([Xn[0]], [Yn[0]], marker='s', s=70, label='Départ')
    plt.scatter([Xn[-1]], [Yn[-1]], marker='*', s=150, label='Arrivée')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titre)
    plt.legend()
    plt.grid(True)
    plt.show()



# Question 12 


def gradpc(eps, MaxIter, u, x0, y0, df1, df2):
    """
    Méthode du gradient à pas constant.

    Formule :
        (x_{n+1}, y_{n+1}) = (x_n, y_n) + u * grad f(x_n, y_n)

    Pour chercher un minimum : prendre souvent u < 0.
    Pour chercher un maximum : prendre souvent u > 0.

    Retourne :
        n : nombre d'itérations effectuées
        Xn, Yn : listes des points calculés
        converged : True si ||grad f|| < eps
    """
    Xn = [float(x0)]
    Yn = [float(y0)]
    n = 0

    while n < MaxIter and norme_grad(Xn[-1], Yn[-1], df1, df2) > eps:
        x_new = Xn[-1] + u * df1(Xn[-1], Yn[-1])
        y_new = Yn[-1] + u * df2(Xn[-1], Yn[-1])
        Xn.append(float(x_new))
        Yn.append(float(y_new))
        n += 1

    converged = norme_grad(Xn[-1], Yn[-1], df1, df2) <= eps
    return n, np.array(Xn), np.array(Yn), converged


# ============================================================
# Question 13



def question13():
    eps = 1e-5
    MaxIter = 120

   
    n_h, Xh, Yh, conv_h = gradpc(eps, MaxIter, 0.10, 0, 0, dh_dx, dh_dy)
    afficher_contours_et_iterations(
        h, Xh, Yh,
        xmin=-1, xmax=1, ymin=-0.5, ymax=2.0,
        titre='Q13 - Gradient à pas constant pour h, départ (0,0)'
    )
    print('Q13 - h :')
    print('Nombre d\'itérations =', n_h)
    print('Dernier point =', (Xh[-1], Yh[-1]))
    print('Convergence =', conv_h)
    print('Valeur h finale =', h(Xh[-1], Yh[-1]))

    
    f_g = g(2, 2/7)
    df1_g = dg_dx(2, 2/7)
    df2_g = dg_dy(2, 2/7)

    n_g, Xg, Yg, conv_g = gradpc(eps, MaxIter, -0.04, 7, 1.5, df1_g, df2_g)
    afficher_contours_et_iterations(
        f_g, Xg, Yg,
        xmin=-8, xmax=8, ymin=-2, ymax=2,
        titre='Q13 - Gradient à pas constant pour g_{2,2/7}, départ (7,1.5)'
    )
    print('\nQ13 - g_{2,2/7} :')
    print('Nombre d\'itérations =', n_g)
    print('Dernier point =', (Xg[-1], Yg[-1]))
    print('Convergence =', conv_g)
    print('Valeur g finale =', f_g(Xg[-1], Yg[-1]))

question13()


# ============================================================
# Question 14

def question14():
    eps = 1e-5
    MaxIter = 120
    x0, y0 = 7, 1.5

    f_g = g(1, 20)
    df1_g = dg_dx(1, 20)
    df2_g = dg_dy(1, 20)

    us = np.linspace(-0.99, -0.001, 400)
    erreurs = []
    convergences = []


    distance_initiale = np.sqrt(x0**2 + y0**2)

    for u in us:
        n, Xn, Yn, conv = gradpc(eps, MaxIter, u, x0, y0, df1_g, df2_g)
        distance_finale = np.sqrt(Xn[-1]**2 + Yn[-1]**2)
        erreur_relative = distance_finale / distance_initiale
        erreurs.append(erreur_relative)
        convergences.append(conv)

    erreurs = np.array(erreurs)
    convergences = np.array(convergences)

    plt.figure(figsize=(8, 6))
    plt.plot(us, erreurs)
    plt.yscale('log')
    plt.xlabel('pas u')
    plt.ylabel('erreur relative finale')
    plt.title('Q14 - Erreur relative finale pour g_{1,20}')
    plt.grid(True)
    plt.show()

    meilleur_indice = np.argmin(erreurs)
    print('Q14 :')
    print('Meilleur u testé =', us[meilleur_indice])
    print('Erreur relative minimale =', erreurs[meilleur_indice])
    print('Nombre de valeurs de u ayant convergé =', np.sum(convergences), '/', len(us))

question14()


# ============================================================
# Questions 15 et 16


def gradamax(eps, MaxIter, u, x0, y0, f, df1, df2, kmax=10000):
    """
    Gradient amélioré pour chercher un maximum.
    On avance dans la direction du gradient et on augmente k tant que f augmente.
    """
    Xn = [float(x0)]
    Yn = [float(y0)]
    n = 0

    while n < MaxIter and norme_grad(Xn[-1], Yn[-1], df1, df2) > eps:
        gx = df1(Xn[-1], Yn[-1])
        gy = df2(Xn[-1], Yn[-1])

        k = 1
        F1 = f(Xn[-1] + k*u*gx, Yn[-1] + k*u*gy)
        F2 = f(Xn[-1] + (k+1)*u*gx, Yn[-1] + (k+1)*u*gy)

        while F2 > F1 and k < kmax:
            k += 1
            F1 = F2
            F2 = f(Xn[-1] + (k+1)*u*gx, Yn[-1] + (k+1)*u*gy)

        Xn.append(float(Xn[-1] + k*u*gx))
        Yn.append(float(Yn[-1] + k*u*gy))
        n += 1

    converged = norme_grad(Xn[-1], Yn[-1], df1, df2) <= eps
    return n, np.array(Xn), np.array(Yn), converged


def gradamin(eps, MaxIter, u, x0, y0, f, df1, df2, kmax=10000):
    """
    Gradient amélioré pour chercher un minimum.
    Avec u < 0, on avance dans la direction opposée au gradient.
    On augmente k tant que f diminue.
    """
    Xn = [float(x0)]
    Yn = [float(y0)]
    n = 0

    while n < MaxIter and norme_grad(Xn[-1], Yn[-1], df1, df2) > eps:
        gx = df1(Xn[-1], Yn[-1])
        gy = df2(Xn[-1], Yn[-1])

        k = 1
        F1 = f(Xn[-1] + k*u*gx, Yn[-1] + k*u*gy)
        F2 = f(Xn[-1] + (k+1)*u*gx, Yn[-1] + (k+1)*u*gy)

        while F2 < F1 and k < kmax:
            k += 1
            F1 = F2
            F2 = f(Xn[-1] + (k+1)*u*gx, Yn[-1] + (k+1)*u*gy)

        Xn.append(float(Xn[-1] + k*u*gx))
        Yn.append(float(Yn[-1] + k*u*gy))
        n += 1

    converged = norme_grad(Xn[-1], Yn[-1], df1, df2) <= eps
    return n, np.array(Xn), np.array(Yn), converged


def question15_16():
    eps = 1e-5
    MaxIter = 120

    # Q15 : test de gradamax sur h depuis (0,0)
    n_max, Xmax, Ymax, conv_max = gradamax(eps, MaxIter, 0.01, 0, 0, h, dh_dx, dh_dy)
    afficher_contours_et_iterations(
        h, Xmax, Ymax,
        xmin=-1, xmax=1, ymin=-0.5, ymax=2.0,
        titre='Q15 - Gradient amélioré : recherche d’un maximum de h'
    )
    print('Q15 - gradamax sur h :')
    print('Nombre d\'itérations =', n_max)
    print('Dernier point =', (Xmax[-1], Ymax[-1]))
    print('Convergence =', conv_max)
    print('Valeur h finale =', h(Xmax[-1], Ymax[-1]))

    # Q16 : test de gradamin sur g_{1,20} depuis (7,1.5)
    f_g = g(1, 20)
    df1_g = dg_dx(1, 20)
    df2_g = dg_dy(1, 20)

    n_min, Xmin, Ymin, conv_min = gradamin(eps, MaxIter, -0.01, 7, 1.5, f_g, df1_g, df2_g)
    afficher_contours_et_iterations(
        f_g, Xmin, Ymin,
        xmin=-8, xmax=8, ymin=-3, ymax=3,
        titre='Q16 - Gradient amélioré : recherche du minimum de g_{1,20}'
    )
    print('\nQ16 - gradamin sur g_{1,20} :')
    print('Nombre d\'itérations =', n_min)
    print('Dernier point =', (Xmin[-1], Ymin[-1]))
    print('Convergence =', conv_min)
    print('Valeur g finale =', f_g(Xmin[-1], Ymin[-1]))

question15_16()


# ============================================================
# Question 17

def question17():
    eps = 1e-5
    MaxIter = 120
    x0, y0 = 7, 1.5

    f_g = g(1, 20)
    df1_g = dg_dx(1, 20)
    df2_g = dg_dy(1, 20)

    us = np.linspace(-0.999, -0.001, 300)
    iterations_pas_constant = []
    iterations_pas_ameliore = []

    for u in us:
        n1, X1, Y1, conv1 = gradpc(eps, MaxIter, u, x0, y0, df1_g, df2_g)
        n2, X2, Y2, conv2 = gradamin(eps, MaxIter, u, x0, y0, f_g, df1_g, df2_g)

        iterations_pas_constant.append(n1 if conv1 else np.nan)
        iterations_pas_ameliore.append(n2 if conv2 else np.nan)

    plt.figure(figsize=(8, 6))
    plt.plot(us, iterations_pas_constant, label='Pas constant')
    plt.plot(us, iterations_pas_ameliore, label='Pas amélioré')
    plt.xlabel('pas u')
    plt.ylabel('nombre d’itérations avant convergence')
    plt.title('Q17 - Comparaison des méthodes pour g_{1,20}')
    plt.grid(True)
    plt.legend()
    plt.show()

    print('Q17 :')
    print('Médiane des itérations - pas constant =', np.nanmedian(iterations_pas_constant))
    print('Médiane des itérations - pas amélioré =', np.nanmedian(iterations_pas_ameliore))

question17()




# ============================================================
# Question 19


def gradient_pas_optimal(A, b, y0=None, eps=1e-10, MaxIter=10000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if y0 is None:
        y = np.zeros(n)
    else:
        y = np.array(y0, dtype=float)

    Y = [y.copy()]

    for k in range(MaxIter):
        G = 2 * (A @ y - b)

        if np.linalg.norm(G) == 0:
            break

        rho = (np.linalg.norm(G)**2) / (2 * (G.T @ A @ G))
        y_new = y - rho * G
        Y.append(y_new.copy())

        if np.linalg.norm(y_new - y) < eps:
            y = y_new
            break

        y = y_new

    return k + 1, y, np.array(Y)


# ============================================================
# Question 20
# Tests pour n=2 et n=3, comparaison avec numpy.linalg.solve.


def question20():
    # Test n = 2
    A2 = np.array([[4, 1],
                   [1, 3]], dtype=float)
    b2 = np.array([1, 2], dtype=float)

    n2, y2, Y2 = gradient_pas_optimal(A2, b2, y0=[0, 0], eps=1e-10, MaxIter=10000)
    exact2 = np.linalg.solve(A2, b2)

    print('Q20 - Test n=2')
    print('Nombre d\'itérations =', n2)
    print('Solution par gradient =', y2)
    print('Solution exacte numpy =', exact2)
    print('Erreur =', np.linalg.norm(y2 - exact2))

    # Test n = 3
    A3 = np.array([[4, 1, 0],
                   [1, 3, 1],
                   [0, 1, 2]], dtype=float)
    b3 = np.array([1, 2, 3], dtype=float)

    n3, y3, Y3 = gradient_pas_optimal(A3, b3, y0=[0, 0, 0], eps=1e-10, MaxIter=10000)
    exact3 = np.linalg.solve(A3, b3)

    print('\nQ20 - Test n=3')
    print('Nombre d\'itérations =', n3)
    print('Solution par gradient =', y3)
    print('Solution exacte numpy =', exact3)
    print('Erreur =', np.linalg.norm(y3 - exact3))

question20()


# ============================================================
# Questions 21 et 22

def construire_systeme_chaleur(N, c, f_source, a_bord, b_bord):
    dx = 1 / (N + 1)
    xs = np.linspace(dx, 1 - dx, N)

    A = np.zeros((N, N))
    rhs = np.zeros(N)

    for i, x in enumerate(xs):
        ci = c(x) if callable(c) else c
        fi = f_source(x) if callable(f_source) else f_source

        A[i, i] = 2 / dx**2 + ci

        if i > 0:
            A[i, i-1] = -1 / dx**2
        if i < N - 1:
            A[i, i+1] = -1 / dx**2

        rhs[i] = fi

    rhs[0] += a_bord / dx**2
    rhs[-1] += b_bord / dx**2

    return A, rhs, xs


def resoudre_chaleur(N, c, f_source, a_bord, b_bord, eps=1e-10, MaxIter=100000):
    A, rhs, xs = construire_systeme_chaleur(N, c, f_source, a_bord, b_bord)
    n, T_interieur, Y = gradient_pas_optimal(A, rhs, y0=np.zeros(N), eps=eps, MaxIter=MaxIter)

    x_total = np.concatenate(([0], xs, [1]))
    T_total = np.concatenate(([a_bord], T_interieur, [b_bord]))

    return n, x_total, T_total


def question21():
    N = 60
    a_bord = 500
    b_bord = 350

    # Barre parfaitement calorifugée :
    n, x, T = resoudre_chaleur(N, c=0, f_source=0, a_bord=a_bord, b_bord=b_bord)

    # Solution exacte attendue : droite entre 500 K et 350 K
    T_exacte = a_bord + (b_bord - a_bord) * x

    plt.figure(figsize=(8, 6))
    plt.plot(x, T, marker='o', markersize=3, label='Solution numérique')
    plt.plot(x, T_exacte, linestyle='--', label='Solution exacte linéaire')
    plt.xlabel('x')
    plt.ylabel('Température T (K)')
    plt.title('Q21 - Barre calorifugée en régime permanent')
    plt.grid(True)
    plt.legend()
    plt.show()

    print('Q21 :')
    print('Nombre d\'itérations =', n)
    print('Température minimale =', np.min(T))
    print('Température maximale =', np.max(T))


def question22():
    N = 60
    a_bord = 500
    b_bord = 350
    Ta = 300
    D = np.sqrt(0.1)
    c_val = 1 / D**2
    f_val = c_val * Ta

    n, x, T = resoudre_chaleur(N, c=c_val, f_source=f_val, a_bord=a_bord, b_bord=b_bord)

    plt.figure(figsize=(8, 6))
    plt.plot(x, T, marker='o', markersize=3, label='Solution numérique')
    plt.axhline(Ta, linestyle='--', label='Température atmosphère Ta')
    plt.xlabel('x')
    plt.ylabel('Température T (K)')
    plt.title('Q22 - Barre avec échange conducto-convectif')
    plt.grid(True)
    plt.legend()
    plt.show()

    print('Q22 :')
    print('D =', D)
    print('c =', c_val)
    print('f =', f_val)
    print('Nombre d\'itérations =', n)
    print('Température minimale =', np.min(T))
    print('Température maximale =', np.max(T))

question21()
question22()
