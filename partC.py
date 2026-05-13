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
    
    surf = plt.axes(projection='3d')
    surf.plot_surface(X, Y, Z)
    surf.set_title('Représentation 3D de '+f_name)
    surf.set_xlabel('x')
    surf.set_ylabel('y')  
    surf.set_zlabel(f_shortname)
    #surlignage des axes x et y
    surf.plot([x_min, x_max], [0, 0], [0, 0], color='black', linewidth=2)
    surf.plot([0, 0], [y_min, y_max], [0, 0], color='black', linewidth=2)
    plt.show()

plot_3d(g(2, 2/7), "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)
plot_3d(h, "h", "h = cos(x)sin(y)", -1*np.pi, 2*np.pi, -1*np.pi, 2*np.pi, 100)
