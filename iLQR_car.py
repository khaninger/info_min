


class derivatives():
    def __init__(self, discrete_dynamics, cost_stage, cost_final, n_x, n_u):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(n_u)])
        x = self.x_sym
        u = self.u_sym
        
        l = cost_stage(x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)
        
        l_final = cost_final(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)
        
        f = discrete_dynamics(x, u)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)
    
    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        
        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)
        
        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u
    
    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        
        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        
        return l_final_x, l_final_xx

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

import matplotlib.pyplot as plt
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)
import pydrake.symbolic as sym
import numpy as np

n_x = 5
n_u = 2
def car_continuous_dynamics(x, u):
    # x = [x position, y position, heading, speed, steering angle] 
    # u = [acceleration, steering velocity]
    m = sym if x.dtype == object else np # Check type for autodiff
    heading = x[2]
    v = x[3]
    steer = x[4]
    x_d = np.array([
        v*m.cos(heading),
        v*m.sin(heading),
        v*m.tan(steer),
        u[0],
        u[1]        
    ])
    return x_d
    
def discrete_dynamics(x, u):
    dt = 0.1
    x_next = x + dt*car_continuous_dynamics(x, u)
    return x_next

r = 2.0
v_target = 2.0
eps = 1e-6 # The derivative of sqrt(x) at x=0 is undefined. Avoid by subtle smoothing
def cost_stage(x, u):
    m = sym if x.dtype == object else np # Check type for autodiff
    c_circle = (m.sqrt(x[0]**2 + x[1]**2 + eps) - r)**2
    c_speed = (x[3]-v_target)**2
    c_control= (u[0]**2 + u[1]**2)*0.1
    return c_circle + c_speed + c_control

def cost_final(x):
    m = sym if x.dtype == object else np # Check type for autodiff
    c_circle = (m.sqrt(x[0]**2 + x[1]**2 + eps) - r)**2
    c_speed = (x[3]-v_target)**2
    return c_circle + c_speed


def rollout(x0, u_trj):
    x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
    x_trj[0,:] = x0
    for i in range(u_trj.shape[0]):
        x_trj[i+1,:] = discrete_dynamics(x_trj[i,:],u_trj[i])    
    return x_trj

def cost_trj(x_trj, u_trj):
    cost_trj = 0.0
    for i in range(u_trj.shape[0]):
        cost_trj += cost_stage(x_trj[i,:], u_trj[i,:])
    cost_trj += cost_final(x_trj[-1,:])   
    return cost_trj

def forward_pass(x_trj, u_trj, k_trj, K_trj):
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0,:] = x_trj[0,:]
    u_trj_new = np.zeros(u_trj.shape)
    for n in range(u_trj.shape[0]):
        u_trj_new[n,:] = u_trj[n,:]+k_trj[n,:]+K_trj[n,:].dot(x_trj_new[n,:]-x_trj[n,:])
        x_trj_new[n+1,:] = discrete_dynamics(x_trj_new[n,:],u_trj_new[n,:])
    return x_trj_new, u_trj_new
    
def backward_pass(x_trj, u_trj, regu, deriv):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0
    V_x, V_xx = deriv.final(x_trj[-1])
    for n in range(u_trj.shape[0]-1, -1, -1):        
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = deriv.stage(x_trj[n],u_trj[n])
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = iLQR.Q_terms( l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
        # We add regularization to ensure that Q_uu is invertible and nicely conditioned
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
        k, K = iLQR.gains(Q_uu_regu, Q_u, Q_ux)
        k_trj[n,:] = k
        K_trj[n,:,:] = K
        V_x, V_xx = iLQR.V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
        expected_cost_redu += iLQR.expected_cost_reduction(Q_u, Q_uu, k)
    return k_trj, K_trj, expected_cost_redu
    
def run_ilqr(x0, N, deriv, max_iter=50, regu_init=100, n_u = 1):
    # First forward rollout
    u_trj = np.random.randn(N-1, n_u)*0.0001
    x_trj = rollout(x0, u_trj)
    total_cost = cost_trj(x_trj, u_trj)
    regu = regu_init
    max_regu = 10000
    min_regu = 0.01
    
    # Setup traces
    cost_trace = [total_cost]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu, deriv)
        print(k_trj)
        print(K_trj)
        x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)
        # Evaluate new trajectory
        total_cost = cost_trj(x_trj_new, u_trj_new)
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu+1e-8)
        # Accept or reject iteration
        if cost_redu > 0:
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost)
            x_trj = x_trj_new
            u_trj = u_trj_new
            regu *= 0.7
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)

        # Early termination if expected improvement is small
        if expected_cost_redu <= 1e-6:
            break
            
    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace

# Setup problem and call iLQR
x0 = np.array([-3.0, 1.0, -0.2, 0.0, 0.0])
N = 50
max_iter=50
regu_init=100

deriv = derivatives(discrete_dynamics, cost_stage, cost_final, 5, 2)
x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = run_ilqr(x0, N, deriv, max_iter, regu_init, n_u = 2)

# Plot resulting trajecotry of car
plt.figure(figsize=(9.5,8))
plt.plot(x_trj[:,0], linewidth=2)
plt.plot(x_trj[:,1], linewidth=2)
plt.show()

plt.subplots(figsize=(10,6))
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
plt.show()
