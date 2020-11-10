from info_optimizer import derivatives
import numpy as np
from pydrake.all import (Variable, SymbolicVectorSystem, VectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource)
import pydrake.symbolic as sym
class two_mass_sys():
    def __init__(self, N, params = None, dt = 0.02, x_w_cov = 1e-4):
        self.N = N
        self.n_x = 5
        self.n_u = 1
        self.n_y = 2
        self.dt = dt
       
        self.modes = ('free_space', 'contact')
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.75])

        self.V = np.array([5e-3, 5e-2])
        self.W = 1.0*np.array([1e-7, 1e-2, 1e-7, 4e-2, 0])
        self.W0 = np.array([3e-4, 3e-4, 3e-4, 3e-4, x_w_cov])
            
        self.sym_derivs = True # If system should use symbolic derivatives; if false, autodiff
        self.custom_sim = True # If rollouts should be gathered with sys.dyn() calls
        
        if params: 
            self.phi = params
        else:
            self.phi = OrderedDict()
            self.phi['k1'] = 100
            self.phi['k2'] = 50
        
        self.deriv = derivatives(self)
        
    def reset(self):
        x_traj_new = np.zeros((self.N+1, self.n_x))
        x_traj_new[0,:] = self.x0 + np.multiply(np.sqrt(self.W0), np.random.randn(self.n_x))
        u_traj_new = np.zeros((self.N, self.n_u))
        return x_traj_new, u_traj_new

    def dyn(self, x, u, phi = None, mode = None, noise = False):
        # Produce the next step given x and u, optionally with noise
        # TODO: change to just taking context
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
        # Helper function to determine what discrete mode the system is in
        if x[2] > x[-1]:
            return 'contact'
        else:
            return 'free_space'

    def cost_stage(self, x, u):
        # Cost at an individual time step
        contact_force = self.smax(self.phi['k2']*(x[2]-x[-1]))
        c = 0.05*(3-contact_force)**2 + 0.05*(x[2]-x[-1])**2 + 0.1*x[1]**2 + 0.1*x[3]**2+1e-8*u**2
        return c 

    def cost_final(self, x):
        contact_force = self.smax(self.phi['k2']*(x[2]-x[-1]))
        c = 0.05*(3-contact_force)**2 +  0.05*(x[2]-x[-1])**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        return 1.0*c     

    def smax(self, x, beta = 5.0): #Softmax function
        m = sym if x is object else np # check for autodiff            
        return 0.5*(m.sqrt(x**2+beta**2)+x)
    
    def obs(self, x, mode = None, noise = False, phi = None):
        # Return the observations; depending on mode and if input is symbolic var
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
        # Evaluates the cost for a trajectory
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
        
    def update_params(self, params):
        self.phi = params
        self.deriv = derivatives(self)
