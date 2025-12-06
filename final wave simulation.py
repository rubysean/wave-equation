import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons


class WaveSimulation:

    def __init__(self, c = 1.0, cond =2):

        
        self.c1 = c

        self.bound_cond = cond  # 1: Dirichlet, 2: Neumanm
        self.init_type = 'gaussian'  # 초기 조건 타입

        def celer(x,y):
            return self.c1

        
        self.L_x = 5.0
        self.dx = 0.05 
        self.N_x = int(self.L_x/self.dx) 
        self.X = np.linspace(0,self.L_x,self.N_x+1)

        self.L_y = 5.0
        self.dy = 0.05
        self.N_y = int(self.L_y/self.dy)
        self.Y = np.linspace(0,self.L_y,self.N_y+1)

        
        self.c = np.zeros((self.N_x+1,self.N_y+1), float)
        for i in range(0,self.N_x+1):
            for j in range(0,self.N_y+1):
                self.c[i,j] = celer(self.X[i],self.Y[j])


        self.L_t = 10.0
        self.dt = min(self.dx,self.dy)/np.max(self.c) * 0.5   # CFL 조건을 만족하도록 설정함. 0.5는 보정 계수
        self.N_t = int(self.L_t/self.dt)
        self.T = np.linspace(0,self.L_t,self.N_t+1)
    
        
      
        self.anim = None



        
    def V(self,x,y):

        return 0 

    
    def initial_condition(self, x, y):


        if self.init_type == 'gaussian':
            # 중심 가우시안
            return 0.2*np.exp(-((x-2.5)**2/0.1 + (y-2.5)**2/0.1))
        
       
        elif self.init_type == 'double':
            # 이중 소스
            g1 = 0.15*np.exp(-((x-1.5)**2/0.1 + (y-2.5)**2/0.1))
            g2 = 0.15*np.exp(-((x-3.5)**2/0.1 + (y-2.5)**2/0.1))
            return g1 + g2




    def loop_condition(self):
        self.Cx2 = (self.dt/self.dx)**2
        self.Cy2 = (self.dt/self.dy)**2 

        ###########################################################
        self.U = np.zeros((self.N_x+1,self.N_x+1,self.N_t+1),float) 

        self.u_nm1 = np.zeros((self.N_x+1,self.N_y+1),float)   #u_{i,j}^{n-1}
        self.u_n = np.zeros((self.N_x+1,self.N_y+1),float)     # u_{i,j}^{n}
        self.u_np1 = np.zeros((self.N_x+1,self.N_y+1),float)  # u_{i,j}^{n+1}
        self.V_init = np.zeros((self.N_x+1,self.N_y+1),float)
        self.q = np.zeros((self.N_x+1, self.N_y+1), float)

        #t = 0
        for i in range(0, self.N_x+1):
            for j in range(0, self.N_y+1):
                self.q[i,j] = self.c[i,j]**2

        for i in range(0, self.N_x+1):
            for j in range(0, self.N_y+1):
                self.u_n[i,j] = self.initial_condition(self.X[i],self.Y[j])
                
        for i in range(0, self.N_x+1):
            for j in range(0, self.N_y+1):
                self.V_init[i,j] = self.V(self.X[i],self.Y[j])

        self.U[:,:,0] = self.u_n.copy()



    def loop(self):


        self.loop_condition()
        self.u_np1[1:self.N_x,1:self.N_y] = self.u_n[1:self.N_x,1:self.N_y] + self.dt*self.V_init[1:self.N_x,1:self.N_y] + 0.5 * self.Cx2*self.q[1:self.N_x, 1: self.N_y]*(self.u_n[2:self.N_x+1, 1: self.N_y] - 2*self.u_n[1:self.N_x, 1: self.N_y] + self.u_n[0:self.N_x-1, 1: self.N_y]) + self.Cy2*self.q[1:self.N_x, 1: self.N_y]*(self.u_n[1:self.N_x, 2: self.N_y+1] -2*self.u_n[1:self.N_x, 1:self.N_y] + self.u_n[1:self.N_x, 0: self.N_y-1]) 
        

        if self.bound_cond == 1:
            #Dirichlet bound cond
            self.u_np1[0,:] = 0
            self.u_np1[-1,:] = 0
            self.u_np1[:,0] = 0
            self.u_np1[:,-1] = 0

        elif self.bound_cond == 2:
            #Neumann bound cond
            self.u_np1[0,:] = self.u_np1[1,:]
            self.u_np1[-1,:] = self.u_np1[-2,:]
            self.u_np1[:,0] = self.u_np1[:,1]
            self.u_np1[:,-1] =self. u_np1[:,-2]


        self.u_nm1 = self.u_n.copy()
        self.u_n = self.u_np1.copy()
        self.U[:,:,1] = self.u_np1.copy()


        for n in range(2, self.N_t):

                   
            self.u_np1[1:self.N_x,1:self.N_y] = 2*self.u_n[1:self.N_x,1:self.N_y] - self.u_nm1[1:self.N_x,1:self.N_y] + self.Cx2*self.q[1:self.N_x, 1: self.N_y]*(self.u_n[2:self.N_x+1, 1: self.N_y] - 2*self.u_n[1:self.N_x, 1: self.N_y] + self.u_n[0:self.N_x-1, 1: self.N_y]) + self.Cy2*self.q[1:self.N_x, 1: self.N_y]*(self.u_n[1:self.N_x, 2: self.N_y+1] -2*self.u_n[1:self.N_x, 1:self.N_y] + self.u_n[1:self.N_x, 0: self.N_y-1]) 
            #bound conditions
            if self.bound_cond == 1:
                #Dirichlet bound cond
                self.u_np1[0,:] = 0
                self.u_np1[-1,:] = 0
                self.u_np1[:,0] = 0
                self.u_np1[:,-1] = 0

            elif self.bound_cond == 2:
                #Neumann bound cond
                self.u_np1[0,:] = self.u_np1[1,:]
                self.u_np1[-1,:] = self.u_np1[-2,:]
                self.u_np1[:,0] = self.u_np1[:,1]
                self.u_np1[:,-1] =self. u_np1[:,-2]

            self.u_nm1 = self.u_n.copy()
            self.u_n = self.u_np1.copy()
            self.U[:,:,n] = self.u_n.copy()


    def anim_2D(self):

        
        SX,SY = np.meshgrid(self.X,self.Y)  # 3차원 공간을 만들기 위한 격자 구조 생성  
        self.ax.set_zlim(-0.15, 0.15)   # z 축  고정
        
        def update_surf(num):  # num 은 funcanimation이 자동적으로 전달
            self.ax.clear()
            surf = self.ax.plot_surface(SX, SY, self.U[:,:,5*num],cmap = plt.cm.viridis)    # cmap 컬러
            self.ax.set_zlim(-0.15, 0.15)  # 위에거 전부 삭제 되서 다시 설정
            return surf
        
         
        if self.anim is not None:
            self.anim.event_source.stop()
            
    
        self.anim = animation.FuncAnimation(self.fig, update_surf, frames = self.U.shape[2]//5, interval = 50)  # animation 호출
        plt.draw()




    def gui(self):
        
        self.fig = plt.figure(figsize = (10, 9), facecolor = "white")     # 스크린 크기
        self.ax = self.fig.add_subplot(111, projection='3d')   # 화면 위치 조정 및 3d 생성
        plt.subplots_adjust(bottom=0.30 )
        
        
        ax_boundary = plt.axes([0.1, 0.15, 0.15, 0.1],)
        ax_boundary.set_title('Boudnary', fontsize=10, fontweight='bold')
        self.radio_boundary = RadioButtons(ax_boundary, ('Dirichlet', 'Neumann'), active=1 if self.bound_cond == 2 else 0)
        self.radio_boundary.on_clicked(self.on_boundary_change)

        ax_run = plt.axes([0.7, 0.15, 0.1, 0.04])
        self.btn_run = Button(ax_run, 'Run Simulation', color='lightgoldenrodyellow', hovercolor='0.975')
        self.btn_run.on_clicked(self.on_run_click)

        ax_init_cond = plt.axes([0.3, 0.15, 0.15, 0.1],)
        ax_init_cond.set_title('Initial Condition', fontsize=10, fontweight='bold')     
        self.radio_init = RadioButtons(ax_init_cond, ('Gaussian', 'Double Source'), active=0 if self.init_type == 'gaussian' else 1)
        self.radio_init.on_clicked(self.on_init_change)


    def on_boundary_change(self, label):
        
        if 'Dirichlet' in label:
            self.bound_cond = 1
        
        
        elif 'Neumann' in label:
            self.bound_cond = 2
        
        plt.draw()


    def on_init_change(self, label):
        if 'Gaussian' in label:
            self.init_type = 'gaussian'
        
        elif 'Double' in label:
            self.init_type = 'double'
        
        plt.draw()



    def on_run_click(self, event):
        self.loop()
        self.anim_2D()


    def run(self):
        plt.show()

    
# 시뮬레이터 생성
wave = WaveSimulation(c=1.0, cond=2)
wave.gui()
wave.run()
