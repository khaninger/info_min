# Minimizing Directed Information

## Overview
Code to minimize the directed information from system state to controller while maintaining a performance bound.  This is done on nonlinear systems by means of: 
- iLQG
- EKF
- Gradient descent on Directed Information

In addition to the derivatives with respect to state and control (for iLQG/EKF); this approach requires the second derivative with respect to the design parameters over which the optimization should occur.

## Dependencies
This requires an install of [pydrake](https://drake.mit.edu/python_bindings.html) for autodiff functionalities. Tested with commit c4fd7bcd9cecf3de188c074f663a15f0ca76dee3. 

The iiwa demosntration uses the meshcat-based visualizer, and the .SDF files are hard-coded in iiwa_sys.py, and must be appropriately adjusted to run the demo.

## What's all here:

### info_optimizer
- iLQG
- EKF
- derivatives: helper class with all the derivatives req'd for linearization and the info minimization.
- info_optimizer: calculates the gradient of the (i) belief covariance, (ii) directed information, (iii) performance with respect to the design variables, and provides a barrier-method gradient to maintain performance while reducing directed info.

### info_optimzer_unit_test
Unit tests for verifying the gradient of belief and directed info are correct

### info_optimizer_test
Tests of gradient descent via barrier method, iLQG, and a couple other sanity-checks

### iiwa_sys
A door-opening example, can be run as script 

### two_mass_sys
Implements a simple two inertia system for the tests

