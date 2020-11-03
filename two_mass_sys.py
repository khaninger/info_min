class two_mass_sys(nonlin_sys):
    def __init__(self, N, params = None, trj_decay = 0.75, dt = 0.02):
        super().__init__(self, N, params, trj_decay, dt)
        self.N = N
        self.n_x = 5
        self.n_u = 1
        self.n_y = 2
       
        self.modes = ('free_space', 'contact')
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.75])

        self.V = np.array([1e-3, 1e-2])
        self.W = 2.0*np.array([0, 5e-3, 0, 1e-2, 0])
        self.W0 = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
                
        if params: 
            self.phi = params
        else:
            self.phi = OrderedDict()
            self.phi['k1'] = 100
            self.phi['k2'] = 50
        
        self.deriv = derivatives(self)
        
        self.Sigma_hist = None
        self.x_trj, self.u_trj = None, None
        self.rollout()

    def reset(self):
        x_traj_new = np.zeros((self.N+1, self.n_x))
        x_traj_new[0,:] = self.x0 + np.multiply(np.sqrt(self.W0), np.random.randn(self.n_x))
        u_traj_new = np.zeros((self.N, self.n_u))
        return x_traj_new, u_traj_new

    def rollout(self):
        self.u_trj = np.random.randn(self.N, self.n_u)*0.001
        self.x_trj, _ = self.reset()
        for i in range(self.N):
            self.x_trj[i+1,:] = self.dyn(self.x_trj[i,:],self.u_trj[i], noise = True)   

    def dyn(self, x, u, phi = None, mode = None, noise = False):
        # x = [position, velocity, position, velocity, wall_pos], u = [force], mode = free_space || contact
        m = sym if x.dtype == object else np # check for autodiff            
        dt = self.dt
        m1, b1 = 0.2, 9.8
        m2, b2 = 0.11, 5.6 

        if phi is None:
            k1 = self.phi['k1']
            k2 = self.phi['k2']
        else:
            k1 = phi[0]
            k2 = phi[1]

        if not mode:
            mode = self.mode_check(x)

        if mode == 'free_space':
            x_next = x+np.array([dt*x[1], dt/m1*(u[0]-b1*x[1]-k1*(x[0]-x[2])), dt*x[3], dt/m2*(-b2*x[3]+k1*(x[0]-x[2])), 0.0], dtype=object)
        elif mode == 'contact':
            x_next = x+np.array([dt*x[1], dt/m1*(u[0]-b1*x[1]-k1*(x[0]-x[2])), dt*x[3], dt/m2*(-b2*x[3]+k1*(x[0]-x[2])-k2*(x[2]-x[4])), 0.0], dtype=object)
            
        if noise:
            noise = np.multiply(np.sqrt(self.W), np.random.randn(self.n_x))
            x_next += noise
        return x_next 

    def mode_check(self,x):
        if x[2] > x[-1]:
            return 'contact'
        else:
            return 'free_space'

    def cost_stage(self, x, u):
        c = 100/self.phi['k2']**2*(3-self.phi['k2']*(x[2]-x[-1]))**2 + 0.2*x[1]**2 + 0.2*x[3]**2+1e-1*u**2
        #c = 1*(x[2]-0.5)**2 + 0.1*x[1]**2 + 0.1*x[3]**2+1e-7*u**2
        return c 

    def cost_final(self, x):
        c = 100/self.phi['k2']**2*(3-self.phi['k2']*(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        #c = (x[2]-0.5)**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        return 1*c     

    def obs(self, x, mode = None, noise = False, phi = None):
        m = sym if x.dtype == object else np # check for autodiff 
        if phi is None:
            k1 = self.phi['k1']
            k2 = self.phi['k2']
        else:
            k1 = phi[0]
            k2 = phi[1]           
        if not mode:
            mode = self.mode_check(x)
        if mode == 'free_space': 
            y = [x[0], k1*(x[2]-x[0])]
        elif mode == 'contact':
            y = [x[0], k1*(x[2]-x[0])]
        if noise:
            y += np.multiply(np.sqrt(self.V), np.random.randn(2))
        return y

    def cost(self, x_trj = None, u_trj = None):
        cost_trj = 0.0
        if x_trj is None:
            for i in range(self.u_trj.shape[0]):
                cost_trj += self.cost_stage(self.x_trj[i,:], self.u_trj[i,:])
            cost_trj += self.cost_final(self.x_trj[-1,:])  
        else:
            for i in range(u_trj.shape[0]):
                cost_trj += self.cost_stage(x_trj[i,:], u_trj[i,:])
            cost_trj += self.cost_final(x_trj[-1,:])  
        return cost_trj

    def soft_update_trj(self, x_trj_new, u_trj_new, Sigma):
        self.x_trj = self.decay*self.x_trj + (1-self.decay)*x_trj_new
        self.u_trj = self.decay*self.u_trj + (1-self.decay)*u_trj_new
        if self.Sigma_hist is not None:
            self.Sigma_hist = self.decay*self.Sigma_hist+ (1-self.decay)*Sigma
        else:
            self.Sigma_hist = Sigma
