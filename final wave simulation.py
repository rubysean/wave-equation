import numpy as np
import matplotlib.pyplot as plt

def I(x,y):   
    return 0.2*np.exp(-((x-2.5)**2/0.1 + (y-2.5)**2/0.1))  # gpt: 가우시안 함수?? 를 쓰면 모양이 예쁘게 나온다

def V(x,y):   # 초기 속도
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

L_t = 10
dt = 0.015  
N_t = int(L_t/dt) 
T = np.linspace(0,L_t,N_t+1)

c = np.zeros((N_x+1,N_y+1), float)
for i in range(0,N_x+1):
    for j in range(0,N_y+1):
        c[i,j] = celer(X[i],Y[j])

Cx2 = (dt/dx)**2
Cy2 = (dt/dy)**2 

###########################################################
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
u_np1[1:N_x,1:N_y] = u_n[1:N_x,1:N_y] + dt*V_init[1:N_x,1:N_y] + 0.5 * Cx2*q[1:N_x, 1: N_y]*(u_n[2:N_x+1, 1: N_y] - 2*u_n[1:N_x, 1: N_y] + u_n[0:N_x-1, 1: N_y]) + Cy2*q[1:N_x, 1: N_y]*(u_n[1:N_x, 2: N_y+1] -2*u_n[1:N_x, 1: N_y] + u_n[1:N_x, 0: N_y-1]) 

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
    #bound conditions
    if bound_cond == 1:
        #Dirichlet bound cond
        u_np1[0,:] = 0
        u_np1[-1,:] = 0
        u_np1[:,0] = 0
        u_np1[:,-1] = 0
        
    u_nm1 = u_n.copy()      
    u_n = u_np1.copy() 
    U[:,:,n] = u_n.copy()
#########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def anim_2D(X, Y, L, pas_d_images, myzlim = (-0.15, 0.15)):
    fig = plt.figure(figsize = (8, 8), facecolor = "white")     # 스크린 크기
    ax = fig.add_subplot(111, projection='3d')   # 화면 위치 조정 및 3d 생성
    SX,SY = np.meshgrid(X,Y)  # 3차원 공간을 만들기 위한 격자 구조 생성  
    surf = ax.plot_surface(SX, SY, L[:,:,0],cmap = plt.cm.viridis)      # 표면 생성
    ax.set_zlim(myzlim[0], myzlim[1])   # z 축  고정
    
    def update_surf(num):  # num 은 funcanimation이 자동적으로 전달
        ax.clear()
        surf = ax.plot_surface(SX, SY, L[:,:,pas_d_images*num],cmap = plt.cm.viridis)    # cmap 컬러
        ax.set_zlim(myzlim[0], myzlim[1])  # 위에거 전부 삭제 되서 다시 설정
        plt.tight_layout()  # 화면 조정
        return surf
        
    anim = animation.FuncAnimation(fig, update_surf, frames = L.shape[2]//pas_d_images, interval = 50)  # animation 호출

    return anim


anim = anim_2D(X,Y,U,5)
plt.show()












