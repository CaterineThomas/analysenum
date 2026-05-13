import numpy as np
import matplotlib.pyplot as plt

def g(a, b):
    return lambda x, y: (x*x)/a + (y*y)/b

def h(x, y):
    return np.cos(x)*np.sin(y)

def plot_3d(f, f_shortname, f_name, x_min, x_max, y_min, y_max, num_points):
    #l'argument num_points donne le nombre de points par axe, donc la racine du nombre total
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    fig.set_edgecolor('pink')
    plot_surface_on(fig, X, Y, Z, f_shortname).set_title('Représentation 3D de '+f_name)
    plt.show()

def plot_surface_on(fig, X, Y, Z, f_shortname):
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_zlabel(f_shortname)
    ax.set_xlabel('x')
    ax.set_ylabel('y')  
    return ax

plot_3d(g(2, 2/7), "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)
plot_3d(h, "h", "h = cos(x)sin(y)", -1*np.pi, 2*np.pi, -1*np.pi, 2*np.pi, 100)

def plot_cn(f, f_shortname, f_name, x_min, x_max,  n):
    #l'argument n donne le nombre de courbes de niveau désirées
    
    ax = fig.add_subplot()
    
    plt.figure(figsize=(7, 5))
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp) # Ajoute la légende des couleurs
    plt.title("Lignes de niveau (Contour Plot)")
    plt.show()


#courbes_niveau(h, "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)