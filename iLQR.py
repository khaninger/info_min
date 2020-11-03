class derivatives():
    def __init__(self, sys):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(sys.n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(sys.n_u)])
        self.phi_sym = np.array([sym.Variable(list(sys.phi.keys())[i]) for i in range(len(sys.phi))])
        x = self.x_sym
        u = self.u_sym
        phi = self.phi_sym
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
        
        mode = self.sys.mode_check(x)
        
        f_x = sym.Evaluate(self.f_x[mode], env)
        f_u = sym.Evaluate(self.f_u[mode], env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u
    
    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})

        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        
        return l_final_x, l_final_xx

    def filter(self, x, u = None):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}      
        if u: env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})

        mode = self.sys.mode_check(x)        
        f_x = sym.Evaluate(self.f_x[mode], env)
        g_x = sym.Evaluate(self.g_x[mode], env)

        return f_x, g_x

    def params(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}      
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})        
        env.update({self.phi_sym[i]: list(self.sys.phi.values())[i] for i in range(len(self.sys.phi))})
        
        mode = self.sys.mode_check(x) 
        f_x_phi = sym.Evaluate(self.f_x_phi[mode], env)
        g_x_phi = sym.Evaluate(self.g_x_phi[mode], env)

        return f_x_phi, g_x_phi

class two_mass_sys():
    def __init__(self, N, params = None, trj_decay = 0.75, dt = 0.02):
        self.N = N
        self.n_x = 5
        self.n_u = 1
        self.n_y = 2
        self.dt = dt
        self.decay = trj_decay
       
        self.modes = ('free_space', 'contact')
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.75])

        self.V = np.array([1e-3, 1e-2])
        self.W = 1.0*np.array([0, 5e-3, 0, 1e-2, 0])
        self.W0 = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-5])
                
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

class iLQR():
    def __init__(self, sys, max_regu = 10.0, min_regu = 1e-4, state_regu = 1e-2, max_iter = 50):
        self.sys = sys
        self.ekf = EKF(sys)
        self.max_regu = max_regu
        self.min_regu = min_regu
        self.state_regu = state_regu
        
        self.k_trj = None
        self.K_trj = None
        self.Xi_trj = None
        
    def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx, state_regu = 1e-5):
        Q_x = l_x + V_x.T.dot(f_x)
        Q_u = l_u + V_x.T.dot(f_u)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
        Q_ux = l_ux + f_u.T.dot(V_xx+state_regu*np.identity(V_xx.shape[0])).dot(f_x) # Regularize, effectively penalize deviation from initial trajectory; per Tassa2012
        Q_uu = l_uu + f_u.T.dot(V_xx+state_regu*np.identity(V_xx.shape[0])).dot(f_u) # Same regularization
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

    def forward_pass(self):
        x_trj_new, u_trj_new = self.sys.reset()
        self.ekf.filter_init(x_trj_new)
            
        for n in range(self.sys.N):
            u_trj_new[n,:] = 0.7*self.sys.u_trj[n,:]+0.7*self.k_trj[n,:]+self.K_trj[n,:].dot(self.ekf.x_hat[n,:]-self.sys.x_trj[n,:])
            x_trj_new[n+1,:] = self.sys.dyn(x_trj_new[n,:], u_trj_new[n,:], noise=True)
            
            y = sys.obs(x_trj_new[n+1,:], noise=True)
            self.ekf.filter_step(u_trj_new[n,:], y)


        return x_trj_new, u_trj_new
    
    def backward_pass(self, regu = 1, Xi_trj = None):
        u_trj = self.sys.u_trj
        x_trj = self.sys.x_trj
        deriv = self.sys.deriv
        
        self.k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
        self.K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
        self.Xi_trj = np.zeros((self.sys.N, self.sys.n_x, self.sys.n_x))

        expected_cost_redu = 0
        V_x, V_xx = deriv.final(x_trj[-1])
        for n in range(u_trj.shape[0]-1, -1, -1):
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = deriv.stage(x_trj[n],u_trj[n])
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = iLQR.Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx, state_regu = self.state_regu)
            # We add regularization to ensure that Q_uu is invertible and nicely conditioned
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
            k, K = iLQR.gains(Q_uu_regu, Q_u, Q_ux)
            self.k_trj[n,:] = k
            self.K_trj[n,:,:] = K
            V_x, V_xx = iLQR.V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
            
            self.Xi_trj[n,:,:] = K.T.dot(f_u.T.dot(V_xx).dot(f_u)+l_uu).dot(K)
                
            expected_cost_redu += iLQR.expected_cost_reduction(Q_u, Q_uu, k)
        return expected_cost_redu
    
    def plot(self, x_trj_new, u_trj_new):
        plt.cla()
        ax = plt.gca()
        #plt.ylim(0,1.2)
        plt.grid(True)
        
        plt.plot(x_trj_new[:,0],'m')
        plt.plot(self.ekf.x_hat[:,0],'m-.')
        cov = np.sqrt(self.ekf.P[:,0,0])
        ax.fill_between(range(self.sys.N+1), (self.ekf.x_hat[:,0]-cov), (self.ekf.x_hat[:,0]+cov), color='r', alpha=.1)

        plt.plot(self.ekf.x_hat[:,2],'b-.')
        plt.plot(x_trj_new[:,2],'b')
        cov = np.sqrt(self.ekf.P[:,2,2])
        ax.fill_between(range(self.sys.N+1), (self.ekf.x_hat[:,2]-cov), (self.ekf.x_hat[:,2]+cov), color='b', alpha=.1)

        plt.plot(x_trj_new[:,4],'k')
        plt.plot(self.ekf.x_hat[:,4],'k-.')
        cov = np.sqrt(self.ekf.P[:,4,4])
        ax.fill_between(range(self.sys.N+1), (self.ekf.x_hat[:,-1]-cov), (self.ekf.x_hat[:,-1]+cov), color='k', alpha=.1)
        

        #plt.plot(np.trace(self.ekf.P, axis1=1, axis2=2), color='r')
        #plt.plot(np.trace(self.sys.Sigma_hist, axis1=1, axis2=2), color='g')
        #plt.plot(np.trace(self.K_trj, axis1=1, axis2=2), color='r')
        #plt.plot(self.K_trj[:,0,4], color='r')
        plt.draw()
        plt.waitforbuttonpress(0) # this will wait for indefinite time
    
    def fancy_plot(self, cost_trace, regu_trace, redu_ratio_trace, redu_trace):
        plt.figure(figsize=(9.5,8))
        plt.plot(x_trj[:,0], linewidth=2)
        plt.plot(x_trj[0,-1], linewidth=1, color='r')
        plt.plot(u_trj[:,0], linewidth=2)
        plt.grid(True)
        plt.draw()
        plt.waitforbuttonpress(0)

        plt.clf()
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
        
    def run(self, regu=1.0, max_iter=50, do_plots = False, do_final_plot = False):  
        # Setup traces
        cost_trace = [self.sys.cost()]
        expected_cost_redu_trace = []
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regu]
        
        if do_plots or do_final_plot: 
            plt.figure(figsize=(9.5,8))
            plt.grid(True)
        
        # Run main loop
        for it in range(max_iter):
            # Backward and forward pass
            expected_cost_redu = self.backward_pass(regu)
            x_trj_new, u_trj_new = self.forward_pass()
            
            # Evaluate new trajectory
            cost_trace.append(sys.cost(x_trj_new, u_trj_new))
            redu_trace.append(cost_trace[-1] - cost_trace[-1])
            redu_ratio_trace.append(redu_trace[-1] / abs(expected_cost_redu+1e-8))
            # Accept or reject iteration
            if redu_trace[-1] > -1e-1:
            # Improvement (in a somewhat relaxed sense)! Accept new trajectories and lower regularization
                regu *= 0.8
                self.sys.soft_update_trj(x_trj_new, u_trj_new, self.ekf.P)
            else: 
            # Reject new trajectories and increase regularization
                regu *= 1.05
                #u_trj = u_trj + np.random.randn(N-1, sys.n_u)*0.001
            regu = min(max(regu, self.min_regu), self.max_regu)
            regu_trace.append(regu)
            
            if do_plots: self.plot(x_trj_new, u_trj_new)

            # Early termination if expected improvement is small
            if expected_cost_redu <= 1e-2:
                break
        if do_final_plot: self.plot(x_trj_new, u_trj_new)
        return self.sys.x_trj, self.sys.u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace    

class EKF():
    def __init__(self, sys, min_cov = 1e-6, max_cov = 1e5):
        self.sys = sys
        self.W = self.sys.W
        self.W0 = self.sys.W0
        self.V = self.sys.V
        self.x_hat = np.zeros(sys.x_trj.shape)
        self.P = np.zeros((sys.x_trj.shape[0], sys.x_trj.shape[1], sys.x_trj.shape[1]))
        self.n = 0
        
        self.min_cov = min_cov
        self.max_cov = max_cov
        
    def filter_init(self, x_trj):
        x_pred = self.sys.x0
        P_pred = np.diag(self.W0)
        
        y = self.sys.obs(x_pred, noise = True)
        residual = y-self.sys.obs(x_pred)
        _, g_x = self.sys.deriv.filter(x_pred)
        residual_cov = g_x.dot(P_pred).dot(g_x.T)+np.diag(self.V)
        obs_gain = P_pred.dot(g_x.T).dot(np.linalg.inv(residual_cov))

        x_corr = x_pred + obs_gain.dot(residual)
        P_corr = (np.identity(sys.n_x) - obs_gain.dot(g_x)).dot(P_pred)
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
        u, s, vh = np.linalg.svd(residual_cov, full_matrices=True)
        np.clip(s, self.min_cov, self.max_cov)
        residual_cov = np.dot(u * s, vh)
        
        obs_gain = P_pred.dot(g_x.T).dot(np.linalg.inv(residual_cov))
        x_corr = x_pred + obs_gain.dot(residual)
        P_corr = (np.identity(sys.n_x) - obs_gain.dot(g_x)).dot(P_pred)
        self.n += 1
        self.P[self.n,:,:] = P_corr
        self.x_hat[self.n,:] = x_corr

class info_optimizer():
    def __init__(self, iLQR, min_det = 1e-27):
        self.iLQR = iLQR
        self.sys = iLQR.sys
        self.min_det = min_det
        
        self.C_trj = None
        self.dC_trj = None
        self.A_trj = None
        self.dA_trj = None
        self.Sigma_trj = None
        self.Xi_trj = None
        
    def rollout_and_update_traj_gradients(self, par):
        x_trj, u_trj = self.iLQR.forward_pass()
        
        self.Sigma_trj = self.iLQR.ekf.P
        deriv = self.sys.deriv
        
        self.A_trj = np.zeros((self.sys.N, self.sys.n_x, self.sys.n_x))
        self.dA_trj = np.zeros((self.sys.N, self.sys.n_x, self.sys.n_x))
        self.C_trj = np.zeros((self.sys.N, self.sys.n_y, self.sys.n_x))
        self.dC_trj = np.zeros((self.sys.N, self.sys.n_y, self.sys.n_x))

        if par: 
            par_index = list(self.sys.phi).index(par)
        else:
            par_index = 0
            print('Failed to give a parameter')
        for n in range(self.sys.N):
            f_x, g_x = deriv.filter(x_trj[n],u_trj[n])
            self.A_trj[n,:,:] = f_x
            self.C_trj[n,:,:] = g_x
            f_x_phi, g_x_phi = deriv.params(x_trj[n], u_trj[n])
            self.dA_trj[n,:,:] = np.reshape(f_x_phi[:,par_index],(self.sys.n_x, self.sys.n_x))
            self.dC_trj[n,:,:] = np.reshape(g_x_phi[:,par_index],(self.sys.n_y, self.sys.n_x))
       return x_trj, u_trj
       
    def performance(self, num_iter = 20):
    # Calculate the directed info of the control trajectory currently saved in iLQR        
        di = np.zeros(num_iter, dtype=np.double)
        perf = np.zeros(num_iter, dtype=np.double)
        for i in range(num_iter):
            x_trj, u_trj = self.rollout_and_update_traj_gradients('k1')
            #self.iLQR.backward_pass()
            for n in range(self.sys.N):
                W = np.diag(self.sys.W)
                if n is 0:
                    W = np.diag(self.sys.W0)
                di_1 = np.linalg.det(self.A_trj[n,:,:].dot(self.Sigma_trj[n,:,:]).dot(self.A_trj[n,:,:].T)+W)
                di_2 = np.linalg.det(self.Sigma_trj[n,:,:])
                
                #Alternative statement by matrix inversion; has more numerical problems
                #di_1 = np.linalg.det(np.linalg.inv(Sigma[n]) + A[n,:,:].T.dot(np.linalg.inv(W)).dot(A[n,:,:]))
                #di_2 = np.linalg.det(W)
                
                #print(A[n,:,:])
                #print('di1, {}, di2, {}'.format(di_1, di_2))
                
                di_1 = max(di_1, self.min_det)
                di_2 = max(di_2, self.min_det)
                
                #if s1<0 or s2<0:
                #print('Negative determinant at time step {}'.format(n))
                
                di[i] += 0.5*np.log(di_1)-0.5*np.log(di_2)
                perf[i] = self.sys.cost(x_trj, u_trj)
        
        print('Directed Info, mean: {}, std: {}'.format(np.mean(di), np.std(di)))
        print('Total Cost, mean: {}, std: {}'.format(np.mean(perf), np.std(perf)))
        return np.mean(di), np.mean(perf)

    def gamma_grad(self,par):
        d_gamma_trj = np.zeros((self.sys.N, self.sys.n_x, self.sys.n_x))
        d_sigma_trj = np.zeros((self.sys.N, self.sys.n_x, self.sys.n_x))
        V_inv = np.linalg.inv(np.diag(self.sys.V))
        Sigma_hist = self.sys.Sigma_hist 
               
        for t in range(1,self.sys.N): #d gamma / d phi at time step zero is 0 - noise is assumed to be independent of design parameters. 
            Sigma = self.Sigma_trj[t-1,:,:]
            Sigma_h  = Sigma_hist[t-1,:,:]+Sigma           
            C = self.C_trj[t-1,:,:]
            dC = self.dC_trj[t-1,:,:]
            A = self.A_trj[t-1,:,:]
            dA = self.dA_trj[t-1,:,:]
            
            AGammaA_inv = np.linalg.inv(A.dot(Sigma).dot(A)+np.diag(self.sys.W)+1e-6*np.identity(A.shape[0]))
            N0 = dC.T.dot(V_inv).dot(C)+C.T.dot(V_inv).dot(dC) 
            N1 = AGammaA_inv.dot(dA.dot(Sigma).dot(A.T)+A.dot(Sigma).dot(dA.T)).dot(AGammaA_inv)
            N = AGammaA_inv.dot(A.dot(Sigma))

            d_gamma_trj[t,:,:] = N0-N1+N.dot(d_gamma_trj[t-1,:,:]).dot(N.T) 
            
            #if not np.all(np.linalg.eigvals(N0-N1)>-1e-8):
            #    print('N0-N1 not positive definite, eigenvalues are  \n: {}'.format(np.linalg.eigvals(N0-N1)))
            #u, s, vh = np.linalg.svd(AGammaA_inv.dot(A.dot(Sigma)), full_matrices=True)
            #if not np.all(s<1.01):
            #    print('N_{} may be expansive, eigenvalues are: {}'.format(t,s))

            #d_gamma_trj[t,:,:] -= AGammaA_inv.dot(dA.dot(Sigma).dot(A.T)-A.dot(d_sigma_trj[t-1,:,:]).dot(A.T)+A.dot(Sigma).dot(dA.T)).dot(AGammaA_inv)
            #d_sigma_trj[t,:,:] = Sigma.dot(d_gamma_trj[t,:,:]).dot(Sigma)           
        return d_gamma_trj
        
    def grad_directed_info(self,par, d_gamma_trj = None):
        di_grad = 0
        if d_gamma_trj is None: d_gamma_trj = self.gamma_grad(par)
        Sigma_trj = self.Sigma_trj
        W = self.sys.W
        
        for t in range(0,self.sys.N):
            C = self.C_trj[t,:,:]
            dC = self.dC_trj[t,:,:]
            A = self.A_trj[t,:,:]
            dA = self.dA_trj[t,:,:]
            
            GAWA_inv = np.linalg.inv(np.linalg.inv(Sigma_trj[t,:,:])+A.T.dot(W).dot(A))
            di_grad += np.trace(GAWA_inv.dot(d_gamma_trj[t,:,:]+dA.T.dot(W).dot(A)+A.dot(W).dot(dA.T)))
        return di_grad
     
    def grad_barrier(self,par,D, d_gamma_trj = None):
        Xi_trj = self.iLQR.Xi_trj
        Sigma_trj = self.Sigma_trj
        if d_gamma_trj is None: d_gamma_trj = self.gamma_grad(par)
        
        d_Sigma = np.zeros(d_gamma_trj.shape)
        for n in range(self.sys.N):
            d_Sigma[n,:,:] = self.Sigma_trj[n,:,:].dot(d_gamma_trj[n,:,:]).dot(self.Sigma_trj[n,:,:])
        
        barrier = 0.0
        d_barrier = 0.0
        for n in range(self.sys.N):
            barrier += np.trace(Xi_trj[n,:,:].dot(Sigma_trj[n,:,:]))
            d_barrier += np.trace(Xi_trj[n,:,:].dot(d_Sigma[n,:,:]))
            
        if barrier >= D:
            print('Constraint not feasible, following attained: {}'.format(barrier))
            
        return d_barrier/(D-barrier)
    
    def grad(self, D, num_iter):
        grad = {par:np.zeros(num_iter) for par in self.sys.phi}
        for par in self.sys.phi:
            for n in range(num_iter):
                self.rollout_and_update_traj_gradients(par)
                d_gamma_trj = self.gamma_grad(par)
                grad[par][n] = self.grad_directed_info(par, d_gamma_trj) + self.grad_barrier(par, D, d_gamma_trj)
            print('Total gradient for {} is mean {} std {}'.format(par, np.mean(grad[par]), np.std(grad[par])))
            #plt.show()
        return grad
    
import matplotlib.pyplot as plt
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)
import pydrake.symbolic as sym
import numpy as np
from collections import OrderedDict    

