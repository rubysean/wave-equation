import numpy as np
import matplotlib.plot as plt

def I(x,y):
    
    return 0.2*np.exp(-((x-1)**2/0.1 + (y-1)**2/0.1))

def V(x,y):
    return 0
  
def celer(x,y):
  
    return 1
    
loop_exec = 1 # Processing loop execution flag

bound_cond = int(input('cond:'))  #Boundary cond 1 : Dirichlet, 2 : Neumann, 3 Mur

if bound_cond not in [1,2,3]:
    loop_exec = 0
    print("Please choose a correct boundary condition")

L_x = 5 #Range of the domain according to x [m]
dx = 0.05 #Infinitesimal distance in the x direction
N_x = int(L_x/dx) #Points number of the spatial mesh in the x direction
X = np.linspace(0,L_x,N_x+1) #Spatial array in the x direction

L_y = 5 #Range of the domain according to y [m]
dy = 0.05 #Infinitesimal distance in the x direction
N_y = int(L_y/dy) #Points number of the spatial mesh in the y direction
Y = np.linspace(0,L_y,N_y+1) #Spatial array in the y direction

L_t = 5 #Duration of simulation [s]
dt = dt = 0.3*min(dx, dy)   #Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
N_t = int(L_t/dt) #Points number of the temporal mesh
T = np.linspace(0,L_t,N_t+1) #Temporal array

c = np.zeros((N_x+1,N_y+1), float)
for i in range(0,N_x+1):
    for j in range(0,N_y+1):
        c[i,j] = celer(X[i],Y[j])

Cx2 = (dt/dx)**2
Cy2 = (dt/dy)**2 
CFL_1 = dt/dy*c[:,0]
CFL_2 = dt/dy*c[:,N_y]
CFL_3 = dt/dx*c[0,:]
CFL_4 =dt/dx*c[N_x,:]










