from info_optimizer import *
import numpy as np
def grad_descent_test(max_iter = 100, D = 40.0, k10=50, k20=20, x_w_cov = 1e-4, beta = 5e4):
    # Setup problem and call iLQR
    params = OrderedDict()
    params['k1'] = k10
    params['k2'] = k20
    sys = two_mass_sys(N = 125, params = params, dt = 0.02, x_w_cov = x_w_cov)
    iLQR_ctrl = iLQR(sys)
    inf_opt = info_optimizer(iLQR_ctrl)

    threshold = -0.5
    total_improvement = threshold + 1.0
    DI_imp = threshold + 1.0
    iters = 0
    iLQR_ctrl.run()
    DI_temp, perf_temp = inf_opt.performance()
    DI_hist = np.array((DI_temp))
    perf_hist = np.array((DI_hist))
    param_hist = {par : np.empty(0) for par in params}
    while total_improvement > threshold or iters < 5:
        print("\nIn the {}th parameter optimization iteration".format(iters))
        iLQR_ctrl.run(max_iter = 15, do_plots = False, do_final_plot = False)
        DI_temp, perf_temp = inf_opt.performance()

        perf_headroom = D-perf_temp
        
        if perf_headroom < -0.0:
            print('Performance bound is BROKEN, D: {}, perf: {}; breaking loop'.format(D, perf_temp))
            break
            
        DI_hist = np.append(DI_hist, DI_temp)
        perf_hist = np.append(perf_hist, perf_temp)
        for par in params:
            param_hist[par] = np.append(param_hist[par], params[par])
        
        DI_imp = DI_hist[-2]-DI_hist[-1]
        total_imp = DI_hist[-2] + beta*np.log(perf_hist[-2]) - DI_hist[-1]-beta*np.log(D-perf_hist[-1])
        
        grad = inf_opt.grad(D-perf_temp, 1, beta = beta)
        
        t = 15.0*(np.tanh(0.05*abs(perf_headroom)))
         
        for par in params:
            if np.std(grad[par]) > abs(np.mean(grad[par])):
                print('Lots of noise on the gradient of {}: mean {} std {}'.format(par, np.mean(grad[par]), np.std(grad[par])))
            params[par] -= t*np.mean(grad[par])
        
        params['k1'] = max(params['k1'],20)
        params['k1'] = min(params['k1'],120)
        params['k2'] = max(params['k2'],3)
        params['k2'] = min(params['k2'],50)
        print('K1: {:>6.2f}, K2: {:>6.2f}'.format(params['k1'], params['k2']))
        iters += 1
        if iters > max_iter: 
            print('Phi optimization hit maximum iterations')
            break
        sys.update_params(params)
    return DI_hist, perf_hist, param_hist
            
def param_grid(k1_grid, k2_grid):
    params = OrderedDict()
    grid_size = k1_grid.shape[0]
    k1_grid = k1_grid
    k2_grid = k2_grid  
    perf = np.zeros((grid_size, grid_size))
    DI = np.zeros((grid_size, grid_size))
    
    i = 0
    for k1 in k1_grid:
        j = 0
        for k2 in k2_grid:
            params['k1'] = k1
            params['k2'] = k2
            
            sys = two_mass_sys(N = 125, params = params, dt = 0.02, x_w_cov = 1e-4)
            iLQR_ctrl = iLQR(sys) 
            inf_opt = info_optimizer(iLQR_ctrl)
            
            iLQR_ctrl.run(max_iter = 15, do_plots = False, do_final_plot = False)
            DI_temp, perf_temp = inf_opt.performance(num_iter = 2)
                        
            perf[i,j] = perf_temp
            DI[i,j] = DI_temp
            
            j += 1
        i += 1

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    k11, k22 = np.meshgrid(k1_grid, k2_grid)
 
    # Plot the surface.
    surf1 = ax.plot_surface(k11, k22, perf, color=[0,0,1,0.5],
                           linewidth=0.5, antialiased=True, label='Total Cost')
    surf1._facecolors2d=surf1._facecolors3d
    surf1._edgecolors2d=surf1._edgecolors3d
    surf2 = ax.plot_surface(k11, k22, DI, color=[1,0,0,0.5],
                           linewidth=0.5, antialiased=True, label='Directed Information')
    surf2._facecolors2d=surf2._facecolors3d
    surf2._edgecolors2d=surf2._edgecolors3d
    ax.set_xlabel('Stiffness K_1')
    ax.set_ylabel('Stiffness K_2')
    ax.legend()
    
    plt.show()
    return perf, DI

def multiple_pts_compare():
    dum1, dum2, p1 = grad_descent_test(D = 35.0, k10 = 85, k20 = 20)
    dum1, dum2, p2 = grad_descent_test(D = 35.0, k10 = 75, k20 = 25)
    dum1, dum2, p3 = grad_descent_test(D = 35.0, k10 = 65, k20 = 30)
    dum1, dum2, p4 = grad_descent_test(D = 35.0, k10 = 55, k20 = 35)
    dum1, dum2, p5 = grad_descent_test(D = 35.0, k10 = 45, k20 = 40)
    dum1, dum2, p6 = grad_descent_test(D = 35.0, k10 = 35, k20 = 45)
    
    grid_size = 5
    k1_grid = np.linspace(20, 120, num = grid_size)
    k2_grid = np.linspace(3, 50, num = grid_size)  
    perf, DI = param_grid(k1_grid, k2_grid)
    
    plt.figure()
    plt.title('Directed info')
    plt.xlabel('Stiffness $K_1$')
    plt.ylabel('Stiffness $K_2$')
    plt.contour(k1_grid, k2_grid, DI, levels = 10, cmap = plt.cm.get_cmap("winter"))
    for i in range(6):
        plt.plot(p1['k1'],p1['k2'],'k-o', linewidth=2, markersize = 8)
    plt.show()
    
    plt.figure()
    plt.title('Performance')
    plt.xlabel('Stiffness $K_1$')
    plt.ylabel('Stiffness $K_2$')
    plt.contour(k1_grid, k2_grid, perf, levels = 10, cmap = plt.cm.get_cmap("winter"))
    for i in range(6):
        plt.plot(p1['k1'],p1['k2'],'k-o', linewidth=2, markersize = 8)
    plt.show()
    return p1, p2, p3, p4, p5, p6

def simple_ilqg():
    params = OrderedDict()
    params['k1'] = 60
    params['k2'] = 30
    sys = two_mass_sys(N = 150, params = params, dt = 0.02, x_w_cov = 1e-4)
    iLQR_ctrl = iLQR(sys)
    inf_opt = info_optimizer(iLQR_ctrl)
    iLQR_ctrl.run(do_final_plot = False, do_fancy_plot = True)
    di, perf  = inf_opt.performance()

    '''
    params['k1'] = 20
    params['k2'] = 17.3
    sys = two_mass_sys(N = 150, params = params, dt = 0.02, x_w_cov = 3e-3)
    iLQR_ctrl = iLQR(sys)
    inf_opt = info_optimizer(iLQR_ctrl)
    iLQR_ctrl.run(do_final_plot = True)
    di, perf  = inf_opt.performance()
    print(di)
    print(perf)
    '''
np.set_printoptions(precision=3)

#grid_size = 5
#k1_grid = np.linspace(20, 120, num = grid_size)
#k2_grid = np.linspace(3, 50, num = grid_size)  
#perf, DI = param_grid(k1_grid, k2_grid)

#grad_descent_test(D = 30.0)
#multiple_pts_compare()
