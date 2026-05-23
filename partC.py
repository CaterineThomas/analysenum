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

def plot_given_cn_on(axe, niveau, f, xmin, xmax, ymin, ymax, num_points):
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    ax.contour(X, Y,)
#plot_3d(g(2, 2/7), "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)
#plot_3d(h, "h", "h = cos(x)sin(y)", -1*np.pi, 2*np.pi, -1*np.pi, 2*np.pi, 100)

def gradpc(eps, MaxIter, u, x0, y0,f, df1, df2, rayon):
    Xn=[x0]
    Yn=[y0]
    gradtemp = [df1(x0,y0), df2(x0,y0)]
    i=0
    while (i<MaxIter and np.sqrt(gradtemp[0]**2+gradtemp[1]**2)>eps):
        Xn.append(Xn[i]+u*gradtemp[0])
        Yn.append(Yn[i] + u*gradtemp[1])
        i += 1
        gradtemp = [df1(Xn[i],Yn[i]), df2(Xn[i],Yn[i])]
    plt.scatter(Xn, Yn)
    """
    x = np.linspace(x0-rayon, x0+rayon, int(1/eps))
    y = np.linspace(y0-rayon, y0+rayon, int(1/eps))
    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)
    for k in range(0,i):
        plt.contour(X, Y, Z, levels=[f(Xn[k], Yn[k])])
    """
    plt.show()
    return (Xn[i], Yn[i])

gradpc(0.01, 100, -0.1, 1,1, h, lambda x,y: -np.sin(x)*np.sin(y), lambda x,y:np.cos(x)*np.cos(y), 6 )




#courbes_niveau(h, "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)