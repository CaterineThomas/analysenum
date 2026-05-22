import numpy as np
import matplotlib.pyplot as plt
#import partB as B

def g(a, b):
    return lambda x, y: (x*x)/a + (y*y)/b

def h(x, y):
    return np.cos(x)*np.sin(y)




def plot_cn_on(fig,  f_shortname, f_name, X, Y, Z):
    #l'argument n donne le nombre de courbes de niveau désirées
    ax = fig.add_subplot(1,2,2)
    ax.contourf(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Lignes de niveau de '+ f_name)
    return ax

def plot_3d(f, f_shortname, f_name, x_min, x_max, y_min, y_max, num_points):
    #l'argument num_points donne le nombre de points par axe, donc la racine du nombre total
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    fig.set_edgecolor('pink')
    plot_surface_on(fig, X, Y, Z, f_shortname).set_title('Représentation 3D de '+f_name)
    plot_cn_on(fig, f_shortname, f_name, X, Y, Z)
    plt.show()

def plot_surface_on(fig, X, Y, Z, f_shortname):
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_zlabel(f_shortname)
    ax.set_xlabel('x')
    ax.set_ylabel('y')  
    return ax

#plot_3d(g(2, 2/7), "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)
#plot_3d(h, "h", "h = cos(x)sin(y)", -1*np.pi, 2*np.pi, -1*np.pi, 2*np.pi, 100)

def gradpc(eps, MaxIter, u, x0, y0, f):
    x, y = x0, y0
    grad = np.gradient(f)
    gradtemp=grad(x0, y0)
    i=0
    while (i<MaxIter and np.norm(gradtemp)<eps):
        x += u*gradtemp[0]
        y += u*gradtemp[1]
        gradtemp = grad(x, y)
        i += 1
    return (x, y) 
gradpc(0.01, 100, -0.1, 0,0, h)




#courbes_niveau(h, "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)