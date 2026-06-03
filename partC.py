import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
#import partB as B

def g(a, b):
    return lambda x, y: (x*x)/a + (y*y)/b

def h(x, y):
    return np.cos(x)*np.sin(y)




def plot_cn_onfig(fig,  f_shortname, f_name, X, Y, Z):
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

def plot_surface_onfig(fig, X, Y, Z, f_shortname):
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_zlabel(f_shortname)
    ax.set_xlabel('x')
    ax.set_ylabel('y')  
    return ax



def plot_iter_onax(axe, Xn, Yn):
    axe.scatter(Xn,Yn)
    plt.scatter([Xn[0]], [Yn[0]], marker='s', s=70, label='Départ')
    plt.scatter([Xn[-1]], [Yn[-1]], marker='*', s=150, label='Arrivée')

def gradpc(eps, MaxIter, u, x0, y0,f, df1, df2):
    Xn=[x0]
    Yn=[y0]
    gradtemp = [df1(x0,y0), df2(x0,y0)]
    Norms = [np.sqrt(gradtemp[0]**2+gradtemp[1]**2)]
    i=0
    while (i<MaxIter and Norms[i]>eps):
        Xn.append(Xn[i]+u*gradtemp[0])
        Yn.append(Yn[i] + u*gradtemp[1])
        i += 1
        gradtemp = [df1(Xn[i],Yn[i]), df2(Xn[i],Yn[i])]
        Norms.append(np.sqrt(gradtemp[0]**2+gradtemp[1]**2))
    minx = min(Xn)-10*eps
    maxx= max(Xn)+10*eps
    miny = min(Yn)-10*eps
    maxy = max(Yn)+10*eps
    x = np.linspace(minx, maxx, int(1/eps))
    y = np.linspace(miny, maxy, int(1/eps))
    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)
    
    norm = colors.Normalize(vmin=Z.min(), vmax=Z.max())
    cmap=cm.get_cmap('plasma')

    for k in range(0,i):
        this_Z_arr = [f(Xn[k], Yn[k])]
        plt.contour(X, Y, Z, levels=this_Z_arr, colors=cmap(norm(this_Z_arr)))
    
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = plt.gca())

    plt.plot(Xn, Yn, marker='o', markersize=500*eps, linewidth=1.2, label='Itérations')
    
    plt.scatter([Xn[0]], [Yn[0]], marker='s', s=70, label='Départ')
    plt.scatter([Xn[-1]], [Yn[-1]], marker='*', s=150, label='Arrivée')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)

    plt.show()
    return (Xn, Yn)

gradpc(0.01, 100, -0.1, 1,1, h, lambda x,y: -np.sin(x)*np.sin(y), lambda x,y:np.cos(x)*np.cos(y))




#courbes_niveau(h, "g", "g, avec a = 2 et b = 2/7", -5, 5, -5, 5, 100)