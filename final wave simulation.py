import numpy as np
import matplotlib.pyplot as plt

def I(x,y):
    return 0.2*np.exp(-((x-2.5)**2/0.1 + (y-2.5)**2/0.1))

def V(x,y):
    return 0
  
def celer(x,y):
    return 1
    

bound_cond = 1

L_x = 5 
dx = 0.05 
N_x = int(L_x/dx) 
X = np.linspace(0,L_x,N_x+1)
L_y = 5
dy = 0.05 
N_y = int(L_y/dy) 
Y = np.linspace(0,L_y,N_y+1) 

L_t = 5 
dt = dt = 0.3*min(dx, dy)   
N_t = int(L_t/dt) 
T = np.linspace(0,L_t,N_t+1)

c = np.zeros((N_x+1,N_y+1), float)
for i in range(0,N_x+1):
    for j in range(0,N_y+1):
        c[i,j] = celer(X[i],Y[j])

Cx2 = (dt/dx)**2
Cy2 = (dt/dy)**2 

if True:
    U = np.zeros((N_x+1,N_x+1,N_t+1),float) 

    u_nm1 = np.zeros((N_x+1,N_y+1),float)   #u_{i,j}^{n-1}
    u_n = np.zeros((N_x+1,N_y+1),float)     # u_{i,j}^{n}
    u_np1 = np.zeros((N_x+1,N_y+1),float)  # u_{i,j}^{n+1}
    V_init = np.zeros((N_x+1,N_y+1),float)
    q = np.zeros((N_x+1, N_y+1), float)

    #t = 0
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            q[i,j] = c[i,j]**2
    
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            u_n[i,j] = I(X[i],Y[j])
            
    for i in range(0, N_x+1):
        for j in range(0, N_y+1):
            V_init[i,j] = V(X[i],Y[j])
    
    U[:,:,0] = u_n.copy()

    # t = 1
    #without boundary cond
    u_np1[1:N_x,1:N_y] = 2*u_n[1:N_x,1:N_y] - (u_n[1:N_x,1:N_y] - 2*dt*V_init[1:N_x,1:N_y]) + Cx2*q[1:N_x, 1: N_y]*(u_n[2:N_x+1, 1: N_y] - 2*u_n[1:N_x, 1: N_y] + u_n[0:N_x-1, 1: N_y]) + Cy2*q[1:N_x, 1: N_y]*(u_n[1:N_x, 2: N_y+1] -2*u_n[1:N_x, 1: N_y] + u_n[1:N_x, 0: N_y-1]) 

    #boundary conditions
    if bound_cond == 1:
        #Dirichlet bound cond
        u_np1[0,:] = 0
        u_np1[-1,:] = 0
        u_np1[:,0] = 0
        u_np1[:,-1] = 0

    u_nm1 = u_n.copy()
    u_n = u_np1.copy()
    U[:,:,1] = u_n.copy()
    
    for n in range(2, N_t):
        
        #calculation at step j+1  
        #without boundary cond           
        u_np1[1:N_x,1:N_y] = 2*u_n[1:N_x,1:N_y] - u_nm1[1:N_x,1:N_y] + Cx2*q[1:N_x, 1: N_y]*(u_n[2:N_x+1, 1: N_y] - 2*u_n[1:N_x, 1: N_y] + u_n[0:N_x-1, 1: N_y]) + Cy2*q[1:N_x, 1: N_y]*(u_n[1:N_x, 2: N_y+1] -2*u_n[1:N_x, 1: N_y] + u_n[1:N_x, 0: N_y-1]) 
        
        if bound_cond == 1:
            #Dirichlet bound cond
            u_np1[0,:] = 0
            u_np1[-1,:] = 0
            u_np1[:,0] = 0
            u_np1[:,-1] = 0
            
        u_nm1 = u_n.copy()      
        u_n = u_np1.copy() 
        U[:,:,n] = u_n.copy()
    











