'''
Drake-based implementation of iLQG, EKF, and information optimizer
@ Kevin Haninger,  October 2020 for all code besides derivatives, iLQG
The classes iLQG, derivatives based on Underactuated course notes by Russ Tedrake, 
    https://colab.research.google.com/github/RussTedrake/underactuated/blob/master/exercises/trajopt/ilqr_driving/ilqr_driving.ipynb
    Licensed under BSD 3-Clause, reproduced below
    Copyright 2018 Russ Tedrake and Robot Locomotion Group @ CSAIL
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.  Redistributions
    in binary form must reproduce the above copyright notice, this list of
    conditions and the following disclaimer in the documentation and/or
    other materials provided with the distribution.  Neither the name of
    the Massachusetts Institute of Technology nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    

'''

import matplotlib.pyplot as plt
from pydrake.all import (Variable, SymbolicVectorSystem, VectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource)
import pydrake.symbolic as sym
import numpy as np
from collections import OrderedDict    
from time import sleep
import copy

class derivatives():
    ''' 
    Class for derivatives of the system dynamics, cost (as required for iLQG)
    as well as 2nd derivative w.r.t. the design parameters phi.
    ''' 
    def __init__(self, sys):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(sys.n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(sys.n_u)])
        x = self.x_sym
        u = self.u_sym
        self.sys = sys
        if self.sys.sym_derivs: 
            self.phi_sym = np.array([sym.Variable(list(sys.phi.keys())[i]) for i in range(len(sys.phi))])
            phi = self.phi_sym
        
        l = sys.cost_stage(x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)
        
        l_final = sys.cost_final(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)
        
        if self.sys.sym_derivs:
            self.f_x = {}
            self.f_u = {}
            self.g_x = {}
            self.f_x_phi = {}
            self.g_x_phi = {}
            
            for mode in sys.modes:
                f = sys.dyn(x, u, phi = phi, mode = mode)
                self.f_x[mode] = sym.Jacobian(f, x)
                self.f_u[mode] = sym.Jacobian(f, u)
                
                g = sys.obs(x, phi = phi, mode = mode)
                self.g_x[mode] = sym.Jacobian(g, x)
                self.f_x_phi[mode] = sym.Jacobian(sym.Jacobian(f,x).ravel(), phi)
                self.g_x_phi[mode] = sym.Jacobian(sym.Jacobian(g,x).ravel(), phi)
    
    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})
        
        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)
        
        
        if self.sys.sym_derivs:
            mode = self.sys.mode_check(x)
            f_x = sym.Evaluate(self.f_x[mode], env)
            f_u = sym.Evaluate(self.f_u[mode], env)
        else:
            f_x, f_u, _ = self.sys.get_deriv(x, u)
        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u
    
    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})

        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        
        return l_final_x, l_final_xx

    def filter(self, x, u = None):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}      
        if u is not None: env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})

        if self.sys.sym_derivs:
            mode = self.sys.mode_check(x)        
            f_x = sym.Evaluate(self.f_x[mode], env)
            g_x = sym.Evaluate(self.g_x[mode], env)
        else:
            f_x, _, g_x = self.sys.get_deriv(x, u)
        return f_x, g_x

    def params(self, x, u = None):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}      
        if u is not None: env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})        
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})
        
        if self.sys.sym_derivs:
            mode = self.sys.mode_check(x) 
            f_x_phi = sym.Evaluate(self.f_x_phi[mode], env)
            g_x_phi = sym.Evaluate(self.g_x_phi[mode], env)
        else:
            f_x_phi = self.sys.get_param_deriv(x, u)
            g_x_phi = np.zeros((self.sys.n_y*self.sys.n_x,3))
        return f_x_phi, g_x_phi

class iLQR(VectorSystem):
    ''' 
    Currently; iLQR is written so it can be used as an independent system 
    (i.e. it can do it's own dynamics calls etc) or as a part of a drake
    diagram, right now that's switched by the simulator argument - if 
    there is none, it's standalone.
    '''
    def __init__(self, sys, max_regu = 10.0, min_regu = 1e-6, state_regu = 1e-4, trj_decay = 0.8, simulator = None):
        VectorSystem.__init__(self, sys.n_y, sys.n_u)           
        self.sys = sys
        self.ekf = EKF(sys)
        self.simulator = simulator
        self.max_regu = max_regu
        self.min_regu = min_regu
        self.state_regu = state_regu
        
        # FF, FB, and value of info trajectories
        self.k_trj = np.zeros([sys.N, sys.n_u])
        self.K_trj = np.zeros([sys.N, sys.n_u, sys.n_x])
        self.k_trj_hist = None #np.zeros([sys.N, sys.n_u])
        self.K_trj_hist = None #np.zeros([sys.N, sys.n_u, sys.n_x])
        self.Xi_trj = np.zeros([sys.N, sys.n_x, sys.n_x])
    
        # *_hist is the smoothed, historical state/input trajectories
        # u/x_trj are trajectories from last rollout
        self.u_trj_hist = None #np.zeros([sys.N, sys.n_u])  # smoothed trajectories
        self.x_trj_hist = None #np.zeros([sys.N+1, sys.n_x])
        self.u_trj = np.zeros([sys.N, sys.n_u])       # current trajectories
        self.x_trj = np.zeros([sys.N+1, sys.n_x])
        
        self.trj_decay = trj_decay
        self.n = 0
        print('iLQR Initialized')
        
    def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx, state_regu = 1e-8):
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
        Q_ux = l_ux + f_u.T.dot(V_xx+state_regu*np.identity(V_xx.shape[0])).dot(f_x) # Regularize, effectively penalize deviation from initial trajectory; per Tassa2012
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
        if np.sum(np.isnan(k))+np.sum(np.isnan(K)):
            print("Gains are NaN! Quu: {} \n Qu: {} \n Qux {}".format(Q_uu, Q_u, Q_ux))
        return k, K 
        
    def expected_cost_reduction(Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

    def soft_update_trj(self):
        self.x_trj_hist = self.trj_decay*self.x_trj_hist + (1-self.trj_decay)*self.x_trj
        self.u_trj_hist = self.trj_decay*self.u_trj_hist + (1-self.trj_decay)*self.u_trj
        
        self.k_trj_hist = copy.deepcopy(self.k_trj)
        self.K_trj_hist = copy.deepcopy(self.K_trj)

    def reset(self):
        # Reset all iLQR vars for a new rollout; also the EKF and simulator are stateful
        self.n = 0 
        self.x_trj, self.u_trj = self.sys.reset()
        self.ekf.filter_init(x_init = self.x_trj[0,:])
        if self.simulator is not None:
            self.simulator.get_mutable_context().SetTime(0.0)
            self.simulator.Initialize()
            sleep(3.5)
        #return x_trj_new, u_trj_new
     
    def random_rollout(self):
        self.reset()
        if self.simulator is not None: # Use simulator; x_trj and u_trj will be updated by DoCalcVectorOutput
            self.simulator.AdvanceTo((self.sys.N)*self.sys.dt)
        else:    
            for n in range(self.sys.N):
                self.u_trj[n,:] = 0.0001*np.random.rand(self.sys.n_u)
                self.x_trj[n+1,:] = self.sys.dyn(self.x_trj[n,:], self.u_trj[n,:], noise=True)
                y = self.sys.obs(self.x_trj[n+1,:], noise=True) #y_{n+1}
                self.ekf.filter_step(self.u_trj[n,:], y, n)
                
        self.x_trj_hist = copy.deepcopy(self.x_trj)
        self.u_trj_hist = copy.deepcopy(self.u_trj)
        return self.sys.cost(self.x_trj, self.u_trj)
        
    def forward_pass(self):
    # Executes a roll-out where system is updated according to the sys.dyn call
        self.reset()
        if self.simulator is not None: # Use simulator; x_trj and u_trj will be updated by DoCalcVectorOutput
            self.simulator.AdvanceTo((self.sys.N)*self.sys.dt)
            x_trj_new = self.x_trj
            u_trj_new = self.u_trj
        else:    
            for n in range(self.sys.N):
                self.u_trj[n,:] = self.u_trj_hist[n,:]+self.k_trj[n,:]+self.K_trj[n,:].dot(self.ekf.x_hat[n,:]-self.x_trj_hist[n,:])
                self.x_trj[n+1,:] = self.sys.dyn(self.x_trj[n,:], self.u_trj[n,:], noise=True)
                y = self.sys.obs(self.x_trj[n+1,:], noise=True) #y_{n+1}
                self.ekf.filter_step(self.u_trj[n,:], y, n)
        
        return self.x_trj, self.u_trj
            
    def DoCalcVectorOutput(self, context, u, x, output):
    # This function is for when iLQG is used as a DRAKE system in a diagram
        y_n = u      # Input to ctrl is ouptut of plant
        n = self.n
        if n is 0:   # First step of iLQR since a reset; init filter with first measurement
            self.ekf.filter_init(y = y_n)
        elif n >= self.sys.N: # Simulator is not a fixed step size, can take smaller steps -> more calls to iLQR
            #print('Too many steps for iLQR to track; n = {}'.format(n))
            output[:] = 0.001*np.random.randn(self.sys.n_u)
            return
        else:
            self.ekf.filter_step(self.u_trj[n-1,:], y_n, n-1) #now x_hat_n should be current
            
        if self.u_trj_hist is None or self.k_trj is None: # First call to iLQR since init, random traj
            u_n = 0.001*np.random.randn(self.sys.n_u)
        else: # Use the latest accepted traj and gains which are stored in sys.
            u_n = self.u_trj_hist[n,:]+self.k_trj[n,:]+self.K_trj[n,:].dot(self.ekf.x_hat[n,:]-self.x_trj_hist[n,:])
        
        self.x_trj[n,:] = self.ekf.x_hat[n,:]
        self.u_trj[n,:] = u_n
        output[:] = u_n
        self.n = n+1
        
    def backward_pass(self, regu = 0.1, Xi_trj = None):
        x_trj = self.x_trj_hist
        u_trj = self.u_trj_hist
        deriv = self.sys.deriv
        
        expected_cost_redu = 0.0
        change_in_gains = [0.0, 0.0]
        V_x, V_xx = deriv.final(x_trj[-1])
        
        for n in range(u_trj.shape[0]-1, -1, -1):
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = deriv.stage(x_trj[n,:],u_trj[n,:])
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = iLQR.Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx, state_regu = self.state_regu)
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu  #Regularization in Q_uu, effective additional control cost

            k, K = iLQR.gains(Q_uu_regu, Q_u, Q_ux)
            change_in_gains[0] += np.sum(abs(self.k_trj[n,:] - k))
            change_in_gains[1] += np.sum(abs(self.K_trj[n,:] - K))
            self.k_trj[n,:] = k
            self.K_trj[n,:,:] = K
            V_x, V_xx = iLQR.V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            self.Xi_trj[n,:,:] = K.T.dot(f_u.T.dot(V_xx).dot(f_u)+l_uu).dot(K)
                
            expected_cost_redu += iLQR.expected_cost_reduction(Q_u, Q_uu, k)
        return expected_cost_redu, change_in_gains
    
    def plot(self, x_trj_new, u_trj_new):
        plt.clf()
        #plt.figure(figsize=(8,5), dpi =80)
        #ax = plt.gca()

        if self.sys.sym_derivs:
            ind = [0, 2, 4]
        else:
            ind = [4, 2, self.sys.door_index]

        ax = plt.subplot(2,1,1)
        plt.grid(True)
        plt.plot(x_trj_new[:-1,ind[0]],'m', label = 'x_1 true')
        plt.plot(self.ekf.x_hat[:-1,ind[0]],'m-.', label = 'x_1 est')
        cov = np.sqrt(self.ekf.P[:-1,ind[0],ind[0]])
        ax.fill_between(range(self.sys.N), (self.ekf.x_hat[:-1,ind[0]]-cov), (self.ekf.x_hat[:-1,ind[0]]+cov), color='r', alpha=.25)

        plt.plot(x_trj_new[:-1,ind[1]],'b', label = 'x_2 true')
        plt.plot(self.ekf.x_hat[:-1,ind[1]],'b-.', label = 'x_2 est')
        cov = 2*np.sqrt(self.ekf.P[:-1,ind[1],ind[1]])
        ax.fill_between(range(self.sys.N), (self.ekf.x_hat[:-1,ind[1]]-cov), (self.ekf.x_hat[:-1,ind[1]]+cov), color='b', alpha=.25)

        plt.plot(x_trj_new[:-1,ind[2]],'k', label = 'door_hinge true')
        plt.plot(self.ekf.x_hat[:-1,ind[2]],'k-.', label = 'door_hinge est')
        cov = np.sqrt(self.ekf.P[:-1,ind[2],ind[2]])
        ax.fill_between(range(self.sys.N), (self.ekf.x_hat[:-1,ind[2]]-cov), (self.ekf.x_hat[:-1,ind[2]]+cov), color='k', alpha=.25)
        
        plt.xlabel(' Time step')
        plt.ylabel(' State value/estimate')
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.grid(True)
        plt.plot(np.log(np.sum(np.abs(self.k_trj),axis=1)),'r', linewidth=3, label = 'log sum abs of FF Traj')
        plt.plot(np.log(np.sum(np.abs(self.K_trj),axis=(1,2))),'k', linewidth=3, label = 'log sum abs of Feedback gains')
        plt.xlabel('Time step')
        plt.ylabel('log Gain value')
        plt.legend()
       
        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time    
    
    def fancy_plot(self, cost_trace, regu_trace, redu_ratio_trace, redu_trace, expected_redu_trace):

        plt.figure()
        # Plot results
        plt.subplot(2, 2, 1)
        plt.plot(cost_trace)
        plt.xlabel('# Iteration')
        plt.ylabel('Total cost')
        plt.title('Cost trace')

        plt.subplot(2, 2, 2)
        plt.plot(expected_redu_trace)
        plt.xlabel('# Iteration')
        plt.ylabel('Expected reduction')
        plt.title('Expected cost reduction')

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
        #plt.waitforbuttonpress(0)
        
    def run(self, regu=0.01, max_iter=50, do_plots = False, do_final_plot = False, do_fancy_plot = False, expected_cost_redu_thresh = 0.1, redu_thresh = 1e-5):  
        # Setup traces
        cost_trace = [self.random_rollout()]
        expected_cost_redu_trace = []
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regu]
        
        if do_plots or do_final_plot: 
            plt.figure(figsize=(9.5,8))
            plt.grid(True)
        
        # Iterate iLQR until: max_iter || expected_cost_redu <= threshold
        for it in range(max_iter):
            # Backward and forward pass
            expected_cost_redu, change_in_gains = self.backward_pass(regu)
            self.forward_pass()
            
            # Logging
            cost_trace.append(self.sys.cost(self.x_trj, self.u_trj))
            redu_trace.append(cost_trace[-2] - cost_trace[-1])
            redu_ratio_trace.append(redu_trace[-1] / abs(expected_cost_redu+1e-8))
            expected_cost_redu_trace.append(expected_cost_redu)

            #If there is a reduction; accept traj and reduce control regularization
            if redu_trace[-1] > redu_thresh:
                regu *= 0.8
                self.soft_update_trj()
            else: 
                regu *= 1.2
                #if self.k_trj_hist is not None:
                #    self.k_trj = self.k_trj_hist # Roll-back the ctrl gains!
                #    self.K_trj = self.K_trj_hist
            regu = min(max(regu, self.min_regu), self.max_regu)
            regu_trace.append(regu)
            if do_plots: self.plot(self.x_trj, self.u_trj)

            # Early termination if expected improvement is small
            if expected_cost_redu <= expected_cost_redu_thresh:
                print('Expected improvement small: stopping iLQR')
                break
        if do_final_plot: self.plot(self.x_trj, self.u_trj)
        if do_fancy_plot: self.fancy_plot(cost_trace, regu_trace, redu_ratio_trace, redu_trace, expected_cost_redu_trace)
        return self.x_trj, self.u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace    

class EKF():
    def __init__(self, sys, min_cov = 1e-10, max_cov = 1e10):
        self.sys = sys
        self.W = self.sys.W
        self.W0 = self.sys.W0
        self.V = self.sys.V
        self.x_hat = np.zeros((sys.N+1, sys.n_x))
        self.P = np.zeros((sys.N+1, sys.n_x, sys.n_x))
        
        self.min_cov = min_cov
        self.max_cov = max_cov
        
    def filter_init(self, y = None, x_init = None):
        if x_init is None:
            x_pred = self.sys.x0
        else:
            x_pred = x_init
        P_pred = np.diag(self.W0)
        
        # First step correction
        if y is None:
            y = self.sys.obs(x_pred, noise = True)
        residual = y-self.sys.obs(x_pred)
        _, g_x = self.sys.deriv.filter(x_pred)
        residual_cov = g_x.dot(P_pred).dot(g_x.T)+np.diag(self.V)
        obs_gain = P_pred.dot(g_x.T).dot(np.linalg.inv(residual_cov))

        x_corr = x_pred + obs_gain.dot(residual)
        P_corr = (np.identity(self.sys.n_x) - obs_gain.dot(g_x)).dot(P_pred)
        self.x_hat[0,:] = x_corr
        self.P[0,:,:] = P_corr

    def filter_step(self, u_prev, y, n):
    # Should be called with u_n, y_{n+1}, will update x_hat_{n+1} and P_{n+1}
        P_prev = self.P[n,:,:]
        x_prev = self.x_hat[n,:]
        x_pred = self.sys.dyn(x_prev, u_prev)
        f_x, _ = self.sys.deriv.filter(x_prev, u=u_prev)
        _, g_x = self.sys.deriv.filter(x_pred, u=u_prev)
        #f_x, g_x = self.sys.deriv.filter(x_prev, u=u_prev)
        P_pred = f_x.dot(P_prev).dot(f_x.T)+np.diag(self.W)
        residual = y - self.sys.obs(x_pred)
        residual_cov = g_x.dot(P_pred).dot(g_x.T)+np.diag(self.V)
        
        obs_gain = P_pred.dot(g_x.T).dot(np.linalg.inv(residual_cov))
        x_corr = x_pred + obs_gain.dot(residual)
        P_corr = (np.identity(self.sys.n_x) - obs_gain.dot(g_x)).dot(P_pred)
        
        if not np.allclose(P_corr, P_corr.T, rtol=1e-5, atol=1e-8):
            print('Updated belief covariacne is not symmetric! Probably numerical issue: ', str(P_corr))
        
        self.P[n+1,:,:] = P_corr
        self.x_hat[n+1,:] = x_corr

class info_optimizer():
    def __init__(self, iLQR, min_det = 1e-30):
        self.iLQR = iLQR
        self.sys = iLQR.sys
        self.ekf = iLQR.ekf
        
        self.min_det = min_det
        
        self.A_trj = np.zeros((self.sys.N, self.sys.n_x, self.sys.n_x))
        self.dA_trj = np.zeros((self.sys.N, self.sys.n_x, self.sys.n_x))
        self.C_trj = np.zeros((self.sys.N+1, self.sys.n_y, self.sys.n_x))
        self.dC_trj = np.zeros((self.sys.N+1, self.sys.n_y, self.sys.n_x))

        self.Sigma_trj = None
        self.Xi_trj = None
        
    def rollout_and_update_traj_gradients(self, par = None):
        x_trj, u_trj = self.iLQR.forward_pass() #rollout trajectory from random initial condition
    
        self.Sigma_trj = self.iLQR.ekf.P
        deriv = self.sys.deriv
        if par is not None:
            if isinstance(par, str):
                par_index = list(self.sys.phi).index(par)
            else:
                par_index = par

        for n in range(self.sys.N):
            f_x, g_x = deriv.filter(x_trj[n],u_trj[n])
            self.A_trj[n,:,:] = f_x
            self.C_trj[n,:,:] = g_x
            if par is not None:
                f_x_phi, g_x_phi = deriv.params(x_trj[n], u_trj[n])
                self.dA_trj[n,:,:] = np.reshape(f_x_phi[:,par_index],(self.sys.n_x, self.sys.n_x))
                self.dC_trj[n,:,:] = np.reshape(g_x_phi[:,par_index],(self.sys.n_y, self.sys.n_x))
        
        _, g_x = deriv.filter(x_trj[self.sys.N])
        self.C_trj[self.sys.N,:,:] = g_x
        _, g_x_phi = deriv.params(x_trj[self.sys.N])
        if par is not None: self.dC_trj[self.sys.N,:,:] = np.reshape(g_x_phi[:,par_index],(self.sys.n_y, self.sys.n_x))

        return x_trj, u_trj
    
    def plot_w_di(self):
        di_trj, perf_trj = self.performance_trj()
        x_trj_new, u_trj_new = self.iLQR.forward_pass()
        
        plt.figure(figsize=(5,7), dpi =150)

        if self.sys.door_index is not None:
            ind = [0, 2, 4]
        else:
            ind = [4, 2, self.sys.door_index]

        ax = plt.subplot(3,1,1)
        plt.grid(True)

        ''' Plot was a bit cluttered with additional states
        plt.plot(x_trj_new[:-1,ind[0]],'m', label = 'x_1 true')
        plt.plot(self.ekf.x_hat[:-1,ind[0]],'m-.', label = 'x_1 est')
        cov = np.sqrt(self.ekf.P[:-1,ind[0],ind[0]])
        ax.fill_between(range(self.sys.N), (self.ekf.x_hat[:-1,ind[0]]-cov), (self.ekf.x_hat[:-1,ind[0]]+cov), color='r', alpha=.25)

        plt.plot(x_trj_new[:-1,ind[1]],'b', label = 'x_2 true')
        plt.plot(self.ekf.x_hat[:-1,ind[1]],'b-.', label = 'x_2 est')
        cov = np.sqrt(self.ekf.P[:-1,ind[1],ind[1]])
        ax.fill_between(range(self.sys.N), (self.ekf.x_hat[:-1,ind[1]]-cov), (self.ekf.x_hat[:-1,ind[1]]+cov), color='b', alpha=.25)
        '''
        plt.plot(x_trj_new[1:-1,ind[2]],'k', label = 'door angle true')
        plt.plot(self.ekf.x_hat[1:-1,ind[2]],'k-.', label = 'door angle estimate')
        cov = np.sqrt(self.ekf.P[1:-1,ind[2],ind[2]])
        ax.fill_between(range(1,self.sys.N), (self.ekf.x_hat[1:-1,ind[2]]-cov), (self.ekf.x_hat[1:-1,ind[2]]+cov), color='k', alpha=.25, label = 'std in estimate')
        
        
        plt.xlabel(' Time step')
        plt.ylabel(' State value/estimate')
        plt.legend()
        
        plt.subplot(3,1,2)
        plt.grid(True)
        plt.plot(np.log(np.sum(np.abs(self.iLQR.k_trj),axis=1)),'r', linewidth=3, label = 'log sum abs of FF Traj')
        plt.plot(np.log(np.sum(np.abs(self.iLQR.K_trj),axis=(1,2))),'k', linewidth=3, label = 'log sum abs of Feedback gains')
        plt.xlabel('Time step')
        plt.ylabel('log Gain value')
        plt.legend()
        
        ax = plt.subplot(3,1,3)
        plt.grid(True)
        mean = np.mean(di_trj[:,1:], axis=0)
        cov = np.sqrt(np.std(di_trj[:,1:], axis=0))
        plt.plot(mean, 'r', label = 'Mutual info $I(x^t,y_t|y^{-1})$')
        ax.fill_between(range(1,self.sys.N), (mean-cov), (mean+cov), color='r', alpha=.25)
        plt.text(0.1, 0.55, 'Total DI: {:.2f}'.format(np.sum(mean)), fontsize=12, transform=ax.transAxes)


        mean = np.mean(perf_trj, axis=0)
        cov = np.sqrt(np.std(perf_trj, axis=0))
        plt.plot(mean, 'b', label = 'Per stage cost, $\ell(x_t,u_t)$')
        ax.fill_between(range(self.sys.N), (mean-cov), (mean+cov), color='b', alpha=.25)
        plt.xlabel('Time step')
        plt.ylabel('Cost/Information')
        plt.text(0.1, 0.45, 'Total Cost: {:.2f}'.format(np.sum(mean)), fontsize=12, transform=ax.transAxes)
        ax.legend()

        #plt.plot()
        plt.show()
        #plt.waitforbuttonpress(0) # this will wait for indefinite time
    
    def performance(self, num_iter = 1):
        # Calculate the directed info of the control trajectory currently saved in iLQR        
        di = np.ones(num_iter, dtype=np.double)
        perf = np.zeros(num_iter, dtype=np.double)

        for i in range(num_iter):
            x_trj, u_trj = self.rollout_and_update_traj_gradients()
            for n in range(self.sys.N):
                W = np.diag(self.sys.W)
                if n is 0:
                    W = np.diag(self.sys.W0)
                di_1 = np.linalg.det(self.A_trj[n,:,:].dot(self.Sigma_trj[n,:,:]).dot(self.A_trj[n,:,:].T)+W)
                di_2 = np.linalg.det(self.Sigma_trj[n,:,:])
                
                #Alternative statement by matrix inversion; has more numerical problems
                #di_alt_1 = np.linalg.det(np.linalg.inv(self.Sigma_trj[n,:,:]) + self.A_trj[n,:,:].T.dot(np.linalg.inv(W)).dot(self.A_trj[n,:,:]))
                #di_alt_2 = np.linalg.det(W)
                #print('DI-DI_alt: {}'.format(np.log(di_1/di_2)-np.log(di_alt_1*di_alt_2)))
                
                if abs(di_1) <self.min_det or abs(di_2) < self.min_det:
                    print('DI is below threshold; may have numerical issues')
                                
                di[i] += 0.5*np.log(di_1)-0.5*np.log(di_2)
            perf[i] = self.sys.cost(x_trj, u_trj)
        
        print('  Directed Info    {:>6.2f} +/- {:>3.2f}'.format( np.mean(di), np.std(di)))
        print('  Total Cost       {:>6.2f} +/- {:>3.2f}'.format( np.mean(perf), np.std(perf)))
        return np.mean(di), np.mean(perf)
        
    def performance_trj(self, num_iter = 1):
        # Calculate the directed info of the control trajectory currently saved in iLQR        
        di_trj = np.ones((num_iter, self.sys.N), dtype=np.double)
        perf_trj = np.zeros((num_iter, self.sys.N), dtype=np.double)

        for i in range(num_iter):
            x_trj, u_trj = self.rollout_and_update_traj_gradients()
            for n in range(self.sys.N):
                W = np.diag(self.sys.W)
                if n is 0:
                    W = np.diag(self.sys.W0)
                di_1 = np.linalg.det(self.A_trj[n,:,:].dot(self.Sigma_trj[n,:,:]).dot(self.A_trj[n,:,:].T)+W)
                di_2 = np.linalg.det(self.Sigma_trj[n,:,:])
                di_trj[i,n] = 0.5*np.log(di_1) - 0.5*np.log(di_2)
                perf_trj[i,n] = self.sys.cost_stage(x_trj[n,:], u_trj[n,:])
                
                #if abs(di_1) <self.min_det or abs(di_2) < self.min_det:
                    #print('DI is below threshold; may have numerical issues')
                    #di_trj[i,n] = 0.0
        return di_trj, perf_trj

    def gamma_grad(self,par):
        d_gamma_trj = np.zeros((self.sys.N+1, self.sys.n_x, self.sys.n_x))
        V_inv = np.linalg.inv(np.diag(self.sys.V))
        self.rollout_and_update_traj_gradients(par)
        d_gamma_trj[0,:,:] = self.dC_trj[0,:,:].T.dot(V_inv).dot(self.C_trj[0,:,:])+self.C_trj[0,:,:].T.dot(V_inv).dot(self.dC_trj[0,:,:])
        for t in range(1,self.sys.N+1): 
            Sigma = self.Sigma_trj[t-1,:,:]       
            C = self.C_trj[t,:,:]
            dC = self.dC_trj[t,:,:]
            A = self.A_trj[t-1,:,:]
            dA = self.dA_trj[t-1,:,:]
            
            AGammaA_inv = np.linalg.inv(A.dot(Sigma).dot(A.T)+np.diag(self.sys.W))
            N0 = dC.T.dot(V_inv).dot(C)+C.T.dot(V_inv).dot(dC) 
            N1 = AGammaA_inv.dot(dA.dot(Sigma).dot(A.T)+A.dot(Sigma).dot(dA.T)).dot(AGammaA_inv)
            N = AGammaA_inv.dot(A.dot(Sigma))
            d_gamma_trj[t,:,:] = N0-N1+N.dot(d_gamma_trj[t-1,:,:]).dot(N.T)
            
            if not np.allclose(d_gamma_trj[t,:,:], d_gamma_trj[t,:,:].T, rtol=1e-5, atol=1e-8):
                print('d_gamma_trj is not symmetric! Probably numerical issue: ', str(d_gamma_trj[t,:,:]))
                
        return d_gamma_trj
    
    def grad_directed_info_alt(self,par, d_gamma_trj = None):
        # Assumes the sys linearization and dC, dA are current. 
        di_grad = 0
        if d_gamma_trj is None: d_gamma_trj = self.gamma_grad(par)
        Sigma_trj = self.Sigma_trj
        
        for t in range(0,self.sys.N):
            C = self.C_trj[t,:,:]
            dC = self.dC_trj[t,:,:]
            A = self.A_trj[t,:,:]
            dA = self.dA_trj[t,:,:]
            Sigma = Sigma_trj[t,:,:]
            
            if t is 0:
                W = np.diag(self.sys.W0)
            else:
                W = np.diag(self.sys.W)
            
            ASAW_inv = np.linalg.inv(A.dot(Sigma).dot(A.T)+W)
            di_grad += 0.5*np.trace(ASAW_inv.dot(A.dot(Sigma).dot(d_gamma_trj[t,:,:]).dot(Sigma).dot(A.T)+dA.dot(Sigma).dot(A.T)+A.dot(Sigma).dot(dA.T)))
            di_grad += 0.5*np.trace(d_gamma_trj[t,:,:].dot(Sigma))
        return di_grad
    
    def grad_directed_info(self,par, d_gamma_trj = None):
        di_grad = 0
        if d_gamma_trj is None: d_gamma_trj = self.gamma_grad(par)
        Sigma_trj = self.Sigma_trj
        
        for t in range(0,self.sys.N):
            C = self.C_trj[t,:,:]
            dC = self.dC_trj[t,:,:]
            A = self.A_trj[t,:,:]
            dA = self.dA_trj[t,:,:]
            if t is 0:
                W = np.diag(self.sys.W0)
            else:
                W = np.diag(self.sys.W)
            W_inv_reg = np.linalg.inv(W+1e-8*np.identity(W.shape[0]))
            
            GAWA_inv = np.linalg.inv(np.linalg.inv(Sigma_trj[t,:,:])+A.T.dot(W_inv_reg).dot(A))
            di_grad += 0.5*np.trace(GAWA_inv.dot(d_gamma_trj[t,:,:]+dA.T.dot(W).dot(A)+A.T.dot(W).dot(dA)))
        return di_grad
     
    def grad_barrier(self,par,D, d_gamma_trj = None):
        Xi_trj = self.iLQR.Xi_trj
        Sigma_trj = self.Sigma_trj
        if d_gamma_trj is None: d_gamma_trj = self.gamma_grad(par)
        
        d_Sigma = np.zeros(d_gamma_trj.shape)
        for n in range(self.sys.N):
            d_Sigma[n,:,:] = self.Sigma_trj[n,:,:].dot(d_gamma_trj[n,:,:]).dot(self.Sigma_trj[n,:,:])
        
        d_barrier = 0.0
        for n in range(self.sys.N):
            d_barrier += np.trace(Xi_trj[n,:,:].dot(d_Sigma[n,:,:]))
            
        return -d_barrier/abs(D)
    
    def grad(self, D, num_iter = 1, beta = 1):
        grad = {par:np.zeros(num_iter) for par in self.sys.phi}
        bar = {par:np.zeros(num_iter) for par in self.sys.phi}
        grad_sum = 0.0 # To normalize the gradient
        for par in self.sys.phi:
            for n in range(num_iter):
                self.rollout_and_update_traj_gradients(par)
                d_gamma_trj = self.gamma_grad(par)
                bar[par][n] =  beta*self.grad_barrier(par, D, d_gamma_trj)
                grad[par][n] = self.grad_directed_info(par, d_gamma_trj)
            grad_sum += np.mean(grad[par])
            
        for par in self.sys.phi:
            grad[par] = grad[par]/grad_sum+bar[par]
            print('  Grad for {} is:  {:>5.2f} w/ bar: {:>5.2f}'.format(par, np.mean(grad[par]), np.mean(bar[par])))
        return grad
    


