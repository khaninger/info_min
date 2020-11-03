class derivatives():
    def __init__(self, sys):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(sys.n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(sys.n_u)])
        x = self.x_sym
        u = self.u_sym
        self.sys = sys
        
        
        l = sys.cost_stage(x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)
        
        l_final = sys.cost_final(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)
        
        self.f_x = {}
        self.f_u = {}
        self.g_x = {}
        for mode in sys.modes:
            f = sys.dyn(x, u, mode)
            self.f_x[mode] = sym.Jacobian(f, x)
            self.f_u[mode] = sym.Jacobian(f, u)
            
            y = sys.obs(x, mode)
            self.g_x[mode] = sym.Jacobian(y, x)
    
    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        
        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)
        
        mode = self.sys.mode_check(x)
        
        f_x = sym.Evaluate(self.f_x[mode], env)
        f_u = sym.Evaluate(self.f_u[mode], env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u
    
    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        
        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        
        return l_final_x, l_final_xx

    def filter(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}      
          
        mode = self.sys.mode_check(x)
        
        f_x = sym.Evaluate(self.f_x[mode], env)
        g_x = sym.Evaluate(self.g_x[mode], env)

        return f_x, g_x

class nonlin_sys():
    def __init__(self):
        self.n_x = 3
        self.n_u = 1
        self.modes = ('free_space', 'contact')
        
        self.V = [0.0001, 0.001]
        self.W = [0.0, 0.001, 0.0]
        self.W0 = [0.0, 0.001, 0.03]
        
        self.x0 = np.array([0.0, 0.0, 0.75])
        
        self.k = 40
        
    def reset(self):
        return self.x0 + np.multiply(np.sqrt(self.W0), np.random.randn(self.n_x))

    def dyn(self, x, u, mode = None, noise = False):
        # x = [position, velocity, wall_pos], u = [force], mode = free_space || contact
        m = sym if x.dtype == object else np # check for autodiff
        m, b, k, dt = 0.2, 10, self.k, .02 

        if not mode:
            mode = self.mode_check(x)

        if mode == 'free_space':
            x_next = x+np.array([dt*x[1], dt/m*(u[0]-b*x[1]), 0.0], dtype=object)
        elif mode == 'contact':
            x_next = x+np.array([dt*x[1], dt/m*(u[0]-b*x[1]-k*(x[0]-x[2])), 0.0], dtype=object)
            
        if noise:
            x_next += np.multiply(np.sqrt(self.W), np.random.randn(self.n_x))
        return x_next 

    def mode_check(self,x):
        if x[0] > x[-1]:
            return 'contact'
        else:
            return 'free_space'

    def cost_stage(self, x, u):
        c = 1*(x[0]-x[-1])**2 + 0.1*x[1]**2 + 1e-7*u**2
        return c 

    def cost_final(self, x):
        c = (x[0]-x[-1])**2 + 0.1*x[1]**2 
        return 100*c     

    def rollout(self, x0, u_trj):
        x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
        x_trj[0,:] = self.reset()
        for i in range(u_trj.shape[0]):
            x_trj[i+1,:] = self.dyn(x_trj[i,:],u_trj[i], noise = True)    
        return x_trj

    def obs(self, x, mode = None, noise = False):
        if not mode:
            mode = self.mode_check(x)
        if mode == 'free_space': 
            y= [x[0], 0]
        elif mode == 'contact':
            y = [x[0], self.k*(x[2]-x[0])]
        if noise:
            y += np.multiply(np.sqrt(self.V), np.random.randn(2))
        return y

    def cost_trj(self, x_trj, u_trj):
        cost_trj = 0.0
        for i in range(u_trj.shape[0]):
            cost_trj += self.cost_stage(x_trj[i,:], u_trj[i,:])
        cost_trj += self.cost_final(x_trj[-1,:])   
        return cost_trj

class iLQR():
    def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_x = l_x + V_x.T.dot(f_x)
        Q_u = l_u + V_x.T.dot(f_u)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
        Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
        
    def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
        V_x = Q_x + Q_u.dot(K)
        V_xx = Q_xx + K.T.dot(Q_ux) + Q_ux.T.dot(K) + K.T.dot(Q_uu).dot(K)
        return V_x, V_xx

    def gains(Q_uu, Q_u, Q_ux):
        Q_uu_inv = np.linalg.inv(Q_uu)
        k = -Q_uu_inv.dot(Q_u)
        K = -Q_uu_inv.dot(Q_ux)
        return k, K 
        
    def expected_cost_reduction(Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

    def forward_pass(x_trj, u_trj, k_trj, K_trj, sys):
        x_trj_new = np.zeros(x_trj.shape)
        x_hat = np.zeros(x_trj.shape)
        x_hat[0,:] = sys.x0
        x_hat_pred = np.zeros(x_trj.shape)
        P = np.zeros((x_trj.shape[0], x_trj.shape[1], x_trj.shape[1]))
        P[0,:,:] = np.diag(sys.W0)
        x_trj_new[0,:] = sys.reset()#x_trj[0,:]
        u_trj_new = np.zeros(u_trj.shape)
        for n in range(u_trj.shape[0]):
            u_trj_new[n,:] = u_trj[n,:]+k_trj[n,:]+K_trj[n,:].dot(x_trj_new[n,:]-x_trj[n,:])
            x_trj_new[n+1,:] = sys.dyn(x_trj_new[n,:],u_trj_new[n,:], noise=True)
            y = sys.obs(x_trj_new[n+1,:], noise=True)
            x_hat[n+1,:], P[n+1,:,:] = iLQR.filter_step(x_hat[n,:], u_trj_new[n,:], P[n,:,:], y, sys, deriv)
        return x_trj_new, u_trj_new, x_hat, P
    
    def backward_pass(x_trj, u_trj, regu, deriv):
        k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
        K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
        expected_cost_redu = 0
        V_x, V_xx = deriv.final(x_trj[-1])
        for n in range(u_trj.shape[0]-1, -1, -1):        
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = deriv.stage(x_trj[n],u_trj[n])
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = iLQR.Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
            # We add regularization to ensure that Q_uu is invertible and nicely conditioned
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
            k, K = iLQR.gains(Q_uu_regu, Q_u, Q_ux)
            k_trj[n,:] = k
            K_trj[n,:,:] = K
            V_x, V_xx = iLQR.V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            expected_cost_redu += iLQR.expected_cost_reduction(Q_u, Q_uu, k)
        return k_trj, K_trj, expected_cost_redu
        
    def filter_step(x_prev, u_prev, P_prev, y,  sys, deriv):
        x_pred = sys.dyn(x_prev, u_prev)
        f_x, g_x = deriv.filter(x_pred)
        
        P_pred = f_x.dot(P_prev).dot(f_x.T)+np.diag(sys.W)
        residual = y - sys.obs(x_pred)
        residual_cov = g_x.dot(P_pred).dot(g_x.T)+np.diag(sys.V)
        obs_gain = P_pred.dot(g_x.T).dot(np.linalg.inv(residual_cov))
        x_corr = x_pred + obs_gain.dot(residual)
        P_corr = (np.identity(sys.n_x) - obs_gain.dot(g_x)).dot(P_pred)
        return x_corr, P_corr
            
import matplotlib.pyplot as plt
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)
import pydrake.symbolic as sym
import numpy as np
    
def run_ilqr(N, sys, max_iter=100, regu_init=1.0):
    # First forward rollout
    u_trj = np.random.randn(N-1, sys.n_u)*0.001
    x_trj = sys.rollout(sys.x0, u_trj)
    total_cost = sys.cost_trj(x_trj, u_trj)
    regu = regu_init
    max_regu = 10.0
    min_regu = 1e-4
    
    # Setup traces
    cost_trace = [total_cost]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]
    
    plt.figure(figsize=(9.5,8))
    plt.grid(True)
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = iLQR.backward_pass(x_trj, u_trj, regu, deriv)
        x_trj_new, u_trj_new, x_hat, P = iLQR.forward_pass(x_trj, u_trj, k_trj, K_trj, sys)
        plt.cla()
        plt.grid(True)
        plt.plot(x_hat[:,0],color='r')
        #plt.plot(P[:,0,0], color='r')
        plt.plot(x_trj_new[:,0],color='k')
        plt.plot(x_trj_new[:,-1],color='k')
        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time
        # Evaluate new trajectory
        total_cost = sys.cost_trj(x_trj_new, u_trj_new)
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu+1e-8)
        # Accept or reject iteration
        if cost_redu > -1e-1:
            # Improvement! Accept new trajectories and lower regularization
            regu *= 0.8
        redu_ratio_trace.append(redu_ratio)
        cost_trace.append(total_cost)
        x_trj = x_trj_new
        u_trj = u_trj_new
        #else:
            # Reject new trajectories and increase regularization
        #   regu *= 1.1
        #    cost_trace.append(cost_trace[-1])
        #    redu_ratio_trace.append(0)
            #u_trj = u_trj + np.random.randn(N-1, sys.n_u)*0.001
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)

        # Early termination if expected improvement is small
        if expected_cost_redu <= 1e-6:
            break
            
    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace

# Setup problem and call iLQR
N = 200
max_iter=150
regu_init=100

test_sys = nonlin_sys()
deriv = derivatives(test_sys)
x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = run_ilqr(N, test_sys, max_iter, regu_init)

# Plot resulting trajecotry of car
plt.figure(figsize=(9.5,8))
plt.plot(x_trj[:,0], linewidth=2)
plt.plot(x_trj[0,-1], linewidth=1, color='r')
plt.plot(u_trj[:,0], linewidth=2)
plt.grid(True)
plt.draw()
plt.waitforbuttonpress(0)

plt.clf()
#plt.subplots(figsize=(10,6))
# Plot results
plt.subplot(2, 2, 1)
plt.plot(cost_trace)
plt.xlabel('# Iteration')
plt.ylabel('Total cost')
plt.title('Cost trace')

plt.subplot(2, 2, 2)
delta_opt = (np.array(cost_trace) - cost_trace[-1])
plt.plot(delta_opt)
plt.yscale('log')
plt.xlabel('# Iteration')
plt.ylabel('Optimality gap')
plt.title('Convergence plot')

plt.subplot(2, 2, 3)
plt.plot(redu_ratio_trace)
plt.title('Ratio of actual reduction and expected reduction')
plt.ylabel('Reduction ratio')
plt.xlabel('# Iteration')

plt.subplot(2, 2, 4)
plt.plot(regu_trace)
plt.title('Regularization trace')
plt.ylabel('Regularization')
plt.xlabel('# Iteration')
plt.tight_layout()
plt.draw()
plt.waitforbuttonpress(0)
