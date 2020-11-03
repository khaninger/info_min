''' 
Unit tests for the information optimization
@Kevin Haninger October 2020

''' 
import matplotlib.pyplot as plt
import copy

def d_sigma_test():
    # Verify that the gradient of precision matrix is approx equal the actual change.
    params = OrderedDict()
    params['k1'] = 10
    params['k2'] = 40
    sys = two_mass_sys(N=125, params = params, x_w_cov = 1e-10)
    iLQR_ctrl = iLQR(sys)
    inf_opt = info_optimizer(iLQR_ctrl)

    iLQR_ctrl.run()
    
    d_gamma_trj1 = {par:np.zeros((sys.N+1, sys.n_x, sys.n_x)) for par in params}
    d_gamma_trj2 = {par:np.zeros((sys.N+1, sys.n_x, sys.n_x)) for par in params}
    d_sigma_trj1 = {par:np.zeros((sys.N+1, sys.n_x, sys.n_x)) for par in params}
    d_sigma_trj2 = {par:np.zeros((sys.N+1, sys.n_x, sys.n_x)) for par in params}
 
    for par in params:
        x_trj1,_ = inf_opt.rollout_and_update_traj_gradients(par)
        d_gamma_trj1[par] = inf_opt.gamma_grad(par) 
        Sigma_trj1 = copy.deepcopy(inf_opt.Sigma_trj)
        for n in range(sys.N+1):
            d_sigma_trj1[par][n,:,:] = -Sigma_trj1[n,:,:].dot(d_gamma_trj1[par][n,:,:]).dot(Sigma_trj1[n,:,:])  

    

    # Change params, and verify change approx equal to grad
    d1 = 0.1
    d2 = -0.1
    params['k1'] += d1
    params['k2'] += d2
    sys.update_params
    
    for par in params:
        x_trj2, _ = inf_opt.rollout_and_update_traj_gradients(par)
        d_gamma_trj2[par] = inf_opt.gamma_grad(par)  
        Sigma_trj2 = inf_opt.Sigma_trj
        for n in range(sys.N+1):
            d_sigma_trj2[par][n,:,:] = -Sigma_trj2[n,:,:].dot(d_gamma_trj2[par][n,:,:]).dot(Sigma_trj2[n,:,:])      
    
    err = Sigma_trj1 + d1*d_sigma_trj1['k1'] + d2*d_sigma_trj2['k2'] - Sigma_trj2
    err_raw = Sigma_trj1 - Sigma_trj2
    
    # Verifying tr, for various combinations of d1/d2.  
    # Bump at contact; this is removed when the dynamics are deterministic (i.e. the two traj are same)
    # Gradient not accounting for how K1 K2 affect trajectory....
    #plt.plot(np.trace(err, axis1 = 1, axis2 = 2),'k')
    #plt.plot(np.trace(err_raw, axis1 = 1, axis2 = 2),'r')
    
    # Verifying individual indices are doing OK
    #ind = 2
    #plt.plot(err[:,ind,ind],'k')
    #plt.plot(err_raw[:,ind,ind],'r')
    #plt.plot(d_sigma_trj1['k1'][:,ind,ind],'b')
    #plt.plot(d_sigma_trj1['k2'][:,ind,ind],'g')
    
    # Verify that, eg increasing k2 or k1 dec uncertainty in wall position
    #ind = 4
    #plt.plot(Sigma_trj1[:,ind,ind],'k')
    #plt.plot(d_sigma_trj1['k1'][:,ind,ind], 'r')
    #plt.plot(d_sigma_trj1['k2'][:,ind,ind], 'g')
    
    plt.show()
    
def DI_grad_test():
    # Verify the directed info calc is correct
    
    params = OrderedDict()
    params['k1'] = 80
    params['k2'] = 80
    sys = two_mass_sys(N=125, params = params, x_w_cov = 1e-5)
    iLQR_ctrl = iLQR(sys)
    inf_opt = info_optimizer(iLQR_ctrl)

    iLQR_ctrl.run()
    
    ddi_1 = {par:0.0 for par in params}
    ddi_2 = {par:0.0 for par in params}
    
    
    di_1, perf_1 = inf_opt.performance(num_iter = 10)
    
    for par in params:
        x_trj1,_ = inf_opt.rollout_and_update_traj_gradients(par)
        ddi_1[par] = inf_opt.grad_directed_info_old(par) 

    # Change params, and verify change approx equal to grad
    d1 = 1.0
    d2 = 1.0
    params['k1'] += d1
    params['k2'] += d2
    sys.update_params
    
    di_2, perf_2 = inf_opt.performance(num_iter = 10)
    
    for par in params:
        x_trj2, _ = inf_opt.rollout_and_update_traj_gradients(par)
        ddi_2[par] = inf_opt.grad_directed_info_old(par) 
        
    print('dDI_dk1: {}'.format(ddi_1['k1']))
    print('dDI_dk2: {}'.format(ddi_1['k2']))
    step_size = np.sqrt(d1**2 + d2**2)
    print('Error: {}'.format((di_1-di_2+d1*ddi_1['k1']+d2*ddi_1['k2'])/step_size))
        
    

from info_optimizer import *
#d_sigma_test()
DI_grad_test()
