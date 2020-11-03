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
class derivatives():
    def __init__(self, sys):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(sys.n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(sys.n_u)])
        x = self.x_sym
        u = self.u_sym
        self.sys = sys
        if sys is two_mass_sys: 
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
        
        if sys is two_mass_sys:
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
        
        
        if self.sys is two_mass_sys:
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

        if self.sys is two_mass_sys:
            mode = self.sys.mode_check(x)        
            f_x = sym.Evaluate(self.f_x[mode], env)
            g_x = sym.Evaluate(self.g_x[mode], env)
        else:
            f_x, _, g_x = self.sys.get_deriv(x, u)
        return f_x, g_x

    def params(self, x, u = None):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}      
        if u: env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})        
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})
        
        if self.sys is two_mass_sys:
            mode = self.sys.mode_check(x) 
            f_x_phi = sym.Evaluate(self.f_x_phi[mode], env)
            g_x_phi = sym.Evaluate(self.g_x_phi[mode], env)
        else:
            f_x_phi = np.zeros(self.sys.n_x, self.sys.n_x)
            g_x_phi = np.zeros(self.sys.n_y, self.sys.n_x)
        return f_x_phi, g_x_phi

class two_mass_sys():
    def __init__(self, N, params = None, trj_decay = 0.9, dt = 0.02, x_w_cov = 1e-4):
        self.N = N
        self.n_x = 5
        self.n_u = 1
        self.n_y = 2
        self.dt = dt
        self.decay = trj_decay
       
        self.modes = ('free_space', 'contact')
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.75])

        self.V = np.array([5e-3, 5e-2])
        self.W = 1.0*np.array([1e-7, 1e-2, 1e-7, 4e-2, 0])
        self.W0 = np.array([3e-4, 3e-4, 3e-4, 3e-4, x_w_cov])
                
        if params: 
            self.phi = params
        else:
            self.phi = OrderedDict()
            self.phi['k1'] = 100
            self.phi['k2'] = 50
        
        self.deriv = derivatives(self)
        
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
        contact_force = self.smax(self.phi['k2']*(x[2]-x[-1]))
        c = 0.05*(3-contact_force)**2 + 0.05*(x[2]-x[-1])**2+ 0.1*x[1]**2 + 0.1*x[3]**2+1e-8*u**2
        
        c = 1/self.phi['k2']**2*(3-self.phi['k2']*(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2+1e-8*u**2

        #c = 0.3*(0.01-(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2+1e-8*u**2
        return c 

    def cost_final(self, x):
        contact_force = self.smax(self.phi['k2']*(x[2]-x[-1]))
        c = 0.05*(3-contact_force)**2 + 0.05*(x[2]-x[-1])**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        #c = 1/self.phi['k2']**2*(3-self.phi['k2']*(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        #c = 0.003*(1-self.phi['k2']*(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        #c = 0.3*(0.01-(x[2]-x[-1]))**2 + 0.1*x[1]**2 + 0.1*x[3]**2
        return 1*c     

    def smax(self, x, beta = 1):
        m = sym if x is object else np # check for autodiff            
        return 0.5*(m.sqrt(x**2+beta**2)+x)
    
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
        
    def update_params(self, params):
        self.phi = params
        self.deriv = derivatives(self)

class iLQR():
    def __init__(self, sys, max_regu = 10.0, min_regu = 1e-6, state_regu = 1e-2):
        self.sys = sys
        self.ekf = EKF(sys)
        self.max_regu = max_regu
        self.min_regu = min_regu
        self.state_regu = state_regu
        
        self.k_trj = np.zeros([sys.N, sys.n_u])
        self.K_trj = np.zeros([sys.N, sys.n_u, sys.n_x])
        self.Xi_trj = np.zeros([sys.N, sys.n_x, sys.n_x])
        
    def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx, state_regu = 1e-8):
        #Q_x = l_x + V_x.T.dot(f_x)
        Q_x = l_x + f_x.T.dot(V_x)
        #Q_u = l_u + V_x.T.dot(f_u)
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

    def forward_pass(self):
        x_trj_new, u_trj_new = self.sys.reset()
        self.ekf.filter_init()
            
        for n in range(self.sys.N):
            u_trj_new[n,:] = self.sys.u_trj[n,:]+self.k_trj[n,:]+self.K_trj[n,:].dot(self.ekf.x_hat[n,:]-self.sys.x_trj[n,:])
            x_trj_new[n+1,:] = self.sys.dyn(x_trj_new[n,:], u_trj_new[n,:], noise=True)
            
            y = self.sys.obs(x_trj_new[n+1,:], noise=True)
            self.ekf.filter_step(u_trj_new[n,:], y)


        return x_trj_new, u_trj_new
    
    def backward_pass(self, regu = 0.1, Xi_trj = None):
        u_trj = self.sys.u_trj #smoothed references
        x_trj = self.sys.x_trj #smoothed refernces
        deriv = self.sys.deriv
        
        expected_cost_redu = 0.0
        change_in_gains = [0.0, 0.0]
        V_x, V_xx = deriv.final(x_trj[-1])
        
        for n in range(u_trj.shape[0]-1, -1, -1):
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = deriv.stage(x_trj[n],u_trj[n])
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
        plt.cla()
        #plt.figure(figsize=(8,5), dpi =80)
        ax = plt.gca()
        if self.sys is two_mass_sys:
            ind = [0, 2, 4]
        else:
            ind = [0, 4, 7]
        #plt.ylim(0,1.2)
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.plot(np.sum(np.abs(self.k_trj),axis=1),'r', linewidth=3, label = 'FF Traj')
        
        plt.plot(x_trj_new[:,ind[0]],'m', label = 'x_1 true')
        plt.plot(self.ekf.x_hat[:,ind[0]],'m-.', label = 'x_1 est')
        cov = np.sqrt(self.ekf.P[:,ind[0],ind[0]])
        ax.fill_between(range(self.sys.N+1), (self.ekf.x_hat[:,ind[0]]-cov), (self.ekf.x_hat[:,ind[0]]+cov), color='r', alpha=.25)

        plt.plot(x_trj_new[:,ind[1]],'b', label = 'x_2 true')
        plt.plot(self.ekf.x_hat[:,ind[1]],'b-.', label = 'x_2 est')
        cov = 2*np.sqrt(self.ekf.P[:,ind[1],ind[1]])
        ax.fill_between(range(self.sys.N+1), (self.ekf.x_hat[:,ind[1]]-cov), (self.ekf.x_hat[:,ind[1]]+cov), color='b', alpha=.25)

        plt.plot(x_trj_new[:,ind[2]],'k', label = 'x_w true')
        #plt.plot(self.ekf.x_hat[:,ind[2]],'k-.', label = 'x_w est')
        #cov = np.sqrt(self.ekf.P[:,ind[2],ind[2]])
        #ax.fill_between(range(self.sys.N+1), (self.ekf.x_hat[:,-1]-cov), (self.ekf.x_hat[:,-1]+cov), color='k', alpha=.25)
        plt.legend()
        '''
        plt.ylim(-8, 30)
        plt.plot(np.log(np.trace(self.ekf.P, axis1=1, axis2=2)), color='k', label = 'log Tr of Sigmat')
        #plt.plot(np.trace(self.sys.Sigma_hist, axis1=1, axis2=2), color='g')
        plt.plot(np.trace(self.K_trj, axis1 = 1, axis2=2), color='r', label = 'Tr of Kt')
        plt.plot(self.k_trj, color='m', label = 'kt')
        #plt.plot(self.K_trj[:,0,4], color='r')
        plt.legend()
        plt.grid(True)
        '''
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
        
    def run(self, regu=0.01, max_iter=50, do_plots = False, do_final_plot = False, do_fancy_plot = False, expected_cost_redu_thresh = 5):  
        # Setup traces
        cost_trace = [self.sys.cost()]
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
            x_trj_new, u_trj_new = self.forward_pass()

            # Evaluate new trajectory
            cost_trace.append(self.sys.cost(x_trj_new, u_trj_new))
            redu_trace.append(cost_trace[-2] - cost_trace[-1])
            redu_ratio_trace.append(redu_trace[-1] / abs(expected_cost_redu+1e-8))
            expected_cost_redu_trace.append(expected_cost_redu)
            
            #If there is a reduction; accept traj and reduce control regularization
            if redu_trace[-1] > 1e-5:
                regu *= 0.9
                self.sys.soft_update_trj(x_trj_new, u_trj_new, self.ekf.P)
            else: 
                regu *= 1.0
            regu = min(max(regu, self.min_regu), self.max_regu)
            regu_trace.append(regu)
            
            if do_plots: self.plot(x_trj_new, u_trj_new)

            # Early termination if expected improvement is small
            if expected_cost_redu <= expected_cost_redu_thresh:
                print('Expected improvement small: stopping iLQR')
                break
        if do_final_plot: self.plot(x_trj_new, u_trj_new)
        if do_fancy_plot: self.fancy_plot(cost_trace, regu_trace, redu_ratio_trace, redu_trace, expected_cost_redu_trace)
        return self.sys.x_trj, self.sys.u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace    

class EKF():
    def __init__(self, sys, min_cov = 1e-10, max_cov = 1e10):
        self.sys = sys
        self.W = self.sys.W
        self.W0 = self.sys.W0
        self.V = self.sys.V
        self.x_hat = np.zeros(sys.x_trj.shape)
        self.P = np.zeros((sys.x_trj.shape[0], sys.x_trj.shape[1], sys.x_trj.shape[1]))
        self.n = 0
        
        self.min_cov = min_cov
        self.max_cov = max_cov
        
    def filter_init(self):
        x_pred = self.sys.x0
        P_pred = np.diag(self.W0)
        
        # First step correction
        y = self.sys.obs(x_pred, noise = True)
        residual = y-self.sys.obs(x_pred)
        _, g_x = self.sys.deriv.filter(x_pred)
        residual_cov = g_x.dot(P_pred).dot(g_x.T)+np.diag(self.V)
        obs_gain = P_pred.dot(g_x.T).dot(np.linalg.inv(residual_cov))

        x_corr = x_pred + obs_gain.dot(residual)
        P_corr = (np.identity(self.sys.n_x) - obs_gain.dot(g_x)).dot(P_pred)
        self.x_hat[0,:] = x_corr
        self.P[0,:,:] = P_corr
        self.n = 0
    
    def filter_step(self, u_prev, y):
        P_prev = self.P[self.n,:,:]
        x_prev = self.x_hat[self.n,:]
        
        x_pred = self.sys.dyn(x_prev, u_prev)
        f_x, _ = self.sys.deriv.filter(x_prev, u=u_prev)
        _, g_x = self.sys.deriv.filter(x_pred, u=u_prev)
        P_pred = f_x.dot(P_prev).dot(f_x.T)+np.diag(self.W)
        residual = y - self.sys.obs(x_pred)
        residual_cov = g_x.dot(P_pred).dot(g_x.T)+np.diag(self.V)
        
        # Make sure residual covarance is not too poorly conditioned.
        # u, s, vh = np.linalg.svd(residual_cov, full_matrices=True)
        # np.clip(s, self.min_cov, self.max_cov)
        # residual_cov = np.dot(u * s, vh)
        
        obs_gain = P_pred.dot(g_x.T).dot(np.linalg.inv(residual_cov))
        x_corr = x_pred + obs_gain.dot(residual)
        P_corr = (np.identity(self.sys.n_x) - obs_gain.dot(g_x)).dot(P_pred)
        
        if not np.allclose(P_corr, P_corr.T, rtol=1e-5, atol=1e-8):
            print('Updated belief covariacne is not symmetric! Probably numerical issue: ', str(P_corr))
        
        self.n += 1
        self.P[self.n,:,:] = P_corr
        self.x_hat[self.n,:] = x_corr

class info_optimizer():
    def __init__(self, iLQR, min_det = 1e-30):
        self.iLQR = iLQR
        self.sys = iLQR.sys
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
            par_index = list(self.sys.phi).index(par)

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

    def gamma_grad(self,par):
        d_gamma_trj = np.zeros((self.sys.N+1, self.sys.n_x, self.sys.n_x))
        V_inv = np.linalg.inv(np.diag(self.sys.V))
               
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
    
    def grad_directed_info(self,par, d_gamma_trj = None):
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
            #if par is 'k2':
                #print('sig_trj', str(np.linalg.eigvals(Sigma_trj[t,:,:])))
                #print('AWA', str(np.linalg.eigvals(A.T.dot(W_inv_reg).dot(A))))
                #print('AT', str((A.T)))
                #print('Winv', str((W_inv_reg)))
                #print('AWA', str((A.T.dot(W_inv_reg).dot(A))))
               # print('gawa_inv', str(np.linalg.eigvals(GAWA_inv)))
                #print('gawa_inv*dgamma,    ', str(np.linalg.eigvals(GAWA_inv.dot(d_gamma_trj[t,:,:]))))
                #print(dA)
        return di_grad
    
    def grad_directed_info_old(self,par, d_gamma_trj = None):
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
            #if par is 'k2':
                #print('sig_trj', str(np.linalg.eigvals(Sigma_trj[t,:,:])))
                #print('AWA', str(np.linalg.eigvals(A.T.dot(W_inv_reg).dot(A))))
                #print('AT', str((A.T)))
                #print('Winv', str((W_inv_reg)))
                #print('AWA', str((A.T.dot(W_inv_reg).dot(A))))
               # print('gawa_inv', str(np.linalg.eigvals(GAWA_inv)))
                #print('gawa_inv*dgamma,    ', str(np.linalg.eigvals(GAWA_inv.dot(d_gamma_trj[t,:,:]))))
                #print(dA)
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
                grad[par][n] = self.grad_directed_info_old(par, d_gamma_trj)
            grad_sum += np.mean(grad[par])
            
        for par in self.sys.phi:
            grad[par] = grad[par]/grad_sum+bar[par]
            print('  Grad for {} is:  {:>5.2f} w/ bar: {:>5.2f}'.format(par, np.mean(grad[par]), np.mean(bar[par])))
        return grad
    
import matplotlib.pyplot as plt
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)
import pydrake.symbolic as sym
import numpy as np
from collections import OrderedDict    

