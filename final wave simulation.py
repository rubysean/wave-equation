import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons


class WaveSimulation:

    def __init__(self, cond =2):        

        self.bound_cond = cond  # 1: Dirichlet, 2: Neumanm
        self.init_type = 'gaussian'  # 초기 조건 타입
        self.init_celer = 'constant'  # 초기 파동 속도 타입    
        
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
                self.c[i,j] = self.celer(self.X[i],self.Y[j])

        self.L_t = 10.0
        self.dt = min(self.dx,self.dy)/np.max(self.c) * 0.5   # CFL 조건을 만족하도록 설정함. 0.5는 보정 계수
        self.N_t = int(self.L_t/self.dt)
        self.T = np.linspace(0,self.L_t,self.N_t+1)
        self.anim = None

    def celer(self,x,y):
            
        if self.init_celer == 'constant':

            return 1.0   # 일정한 파동 속도
        
        elif self.init_celer == 'variable':

            return x + y + 1  #(x,y)에 따라 변화하는 파동 속도       
   
    def update_wave_speed(self):
        """파동 속도 배열을 재계산"""
        self.c = np.zeros((self.N_x+1, self.N_y+1), float)
        for i in range(0, self.N_x+1):
            for j in range(0, self.N_y+1):
                self.c[i,j] = self.celer(self.X[i], self.Y[j])

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

        self.U = np.zeros((self.N_x+1,self.N_y+1,self.N_t+1),float) 

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
            self.u_np1[:,-1] = self. u_np1[:,-2]

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

        def update_surf(num):  # num 은 funcanimation이 자동적으로 전달

            self.ax.clear()
            surf = self.ax.plot_surface(SX, SY, self.U[:,:,5*num].T,cmap = plt.cm.viridis)    # cmap 컬러
            self.ax.set_title(f'Time = {5*num*self.dt:.3f} s')
            self.ax.set_zlim(-0.15, 0.15)  # 위에거 전부 삭제 되서 다시 설정
            self.ax.set_xlabel('X [m]')
            self.ax.set_ylabel('Y [m]')
            self.ax.set_zlabel('U [m]')        
        
            self.ax1.clear()
            cont = self.ax1.pcolormesh(SX, SY, self.U[:,:,5*num].T, cmap=plt.cm.viridis)            
            self.ax1.set_xlabel('X [m]')
            self.ax1.set_ylabel('Y [m]')
            self.ax1.set_title(f'Time = {5*num*self.dt:.3f} s')

            return surf, cont        

         # 컬러바 추가 (한 번만)
        if not hasattr(self, 'cbar'):
            dummy_mesh = self.ax1.pcolormesh(SX, SY, self.U[:,:,0].T, cmap=plt.cm.viridis,vmin=-0.15, vmax=0.15)
            self.cbar = self.fig.colorbar(dummy_mesh, ax=self.ax1, label='U [m]', fraction=0.046, pad=0.04)

        if self.anim is not None:
            self.anim.event_source.stop() 
        
        self.anim = animation.FuncAnimation(self.fig, update_surf, frames = self.U.shape[2]//5, interval = 50)  # animation 호출

        plt.draw()

    def gui(self):
        
        #3D
        self.fig = plt.figure(figsize = (16, 10), facecolor = "white")    
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax.set_zlim(-0.15, 0.15)

        #label
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('U [m]')

        #2D
        self.ax1 = self.fig.add_subplot(122)

        #label
        self.ax1.set_xlabel('X [m]')
        self.ax1.set_ylabel('Y [m]')

        #간격 조정
        plt.subplots_adjust(bottom=0.30)        

        # boundary 조건 
        ax_boundary = plt.axes([0.1, 0.15, 0.15, 0.1],)
        ax_boundary.set_title('Boudnary', fontsize=10, fontweight='bold')
        self.radio_boundary = RadioButtons(ax_boundary, ('Dirichlet', 'Neumann'), active=1 if self.bound_cond == 2 else 0)
        self.radio_boundary.on_clicked(self.on_boundary_change)

        # 속도 조건 
        ax_celer = plt.axes([0.5, 0.15, 0.15, 0.1],)
        ax_celer.set_title('Wave Speed', fontsize=10, fontweight='bold')
        self.radio_celer = RadioButtons(ax_celer, ('Constant', 'Variable'), active=0 if self.init_celer == 'constant' else 1)
        self.radio_celer.on_clicked(self.on_celer_change)

        # 시뮬레이션 실행 
        ax_run = plt.axes([0.7, 0.15, 0.12, 0.06])
        self.btn_run = Button(ax_run, 'Run Simulation', color="#EA6077", hovercolor="#6789E8")
        self.btn_run.on_clicked(self.on_run_click)

        # 초기 함수 조건 
        ax_init_cond = plt.axes([0.3, 0.15, 0.15, 0.1],)
        ax_init_cond.set_title('Initial Condition', fontsize=10, fontweight='bold')     
        self.radio_init = RadioButtons(ax_init_cond, ('Gaussian', 'Double Source'), active=0 if self.init_type == 'gaussian' else 1)
        self.radio_init.on_clicked(self.on_init_change)

        # 제목 
        self.fig.suptitle('2D Wave Equation Simulator', fontsize=16, fontweight='bold')

    def on_boundary_change(self, label):
        
        if 'Dirichlet' in label:
            self.bound_cond = 1        
        
        elif 'Neumann' in label:
            self.bound_cond = 2

    def on_init_change(self, label):
        
        if 'Gaussian' in label:
            self.init_type = 'gaussian'
        
        elif 'Double' in label:
            self.init_type = 'double'
    
    def on_celer_change(self, label):

        if 'Constant' in label:
            self.init_celer ='constant'
            
        elif 'Variable' in label:
            self.init_celer ='variable'

    def on_run_click(self, event):
        
        self.update_wave_speed()
        self.dt = min(self.dx, self.dy)/np.max(self.c) * 0.5
        self.N_t = int(self.L_t/self.dt)

        self.loop()
        self.anim_2D()
        
    def run(self):
        plt.show()
    
# 시뮬레이터 생성
wave = WaveSimulation(cond=2)
wave.gui()
wave.run()

