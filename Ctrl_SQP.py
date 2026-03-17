from scipy.optimize import minimize
from sympy import *
import time

start = time.time()

### input
mul = 1 # efficacy test
xyz = [40,3,0] # external forces, Fx, Fy, Mz
ths = [[-9,-6.73],[-9,6.73],[10,6.73],[10,-6.73]]*mul # thruster's place [x,y]
tcn = [[0,1e9],[0,1e9],[],[]]*mul # thruster's constraint force, [], [fixed value], or [min,max], ignore values of i>1
acn = [[],[],[pi/2],[-pi/2]]*mul # thruster's constraint angel, same as above
nt = len(ths)
eps = 1e-6 # smoothing parameter

### Define the objective function
def objective(x): # use square and add eps to prevent form minus x value that makes x**1.5 error  
    return sum([(abs(x[2*i])+eps)**1.5 for i in range(nt)])

### Define the equality constraint Fx, Fy, Mz
def constraint_eqx(x): # Fx-sum(force_x) = 0
    return sum([x[2*i]*cos(x[2*i+1]) for i in range(nt)]) - xyz[0] 
def constraint_eqy(x):
    return sum([x[2*i]*sin(x[2*i+1]) for i in range(nt)]) - xyz[1]
def constraint_eqz(x): # Mz-sum(Fy*x-Fx*y) = 0
    return sum([x[2*i]*sin(x[2*i+1])*ths[i][0]-x[2*i]*cos(x[2*i+1])*ths[i][1]
                for i in range(nt)]) - xyz[2] 
constraints = [{'type': 'eq', 'fun': constraint_eqx},
               {'type': 'eq', 'fun': constraint_eqy},
               {'type': 'eq', 'fun': constraint_eqz}]

### Define the other equality and inequality constraints
def constraint_eq(index,value): # output == value
    return lambda x: x[index]-value
def constraint_ineq_min(index,value): # output >= value
    return lambda x: x[index]-value
def constraint_ineq_max(index,value): # value >= output
    return lambda x: value-x[index]
for i in range(nt):
    if len(tcn[i]) == 1:
        constraints += [{'type': 'eq', 'fun': constraint_eq(i*2,tcn[i])}]
    elif len(tcn[i]) > 1:
        constraints += [{'type': 'ineq', 'fun': constraint_ineq_min(i*2,tcn[i][0])},
                        {'type': 'ineq', 'fun': constraint_ineq_max(i*2,tcn[i][1])}]
    if len(acn[i]) == 1:
        constraints += [{'type': 'eq', 'fun': constraint_eq(i*2+1,acn[i])}]
    elif len(acn[i]) > 1:
        constraints += [{'type': 'ineq', 'fun': constraint_ineq_min(i*2,acn[i][0])},
                        {'type': 'ineq', 'fun': constraint_ineq_max(i*2,acn[i][1])}]

### Initial guess
x0 = [0]*nt*2
rfx = xyz[0]/nt # average for each thruster
rfy = xyz[1]/nt
for i in range(nt):
    # thrusts
    xi,yi = ths[i]
    if xi or yi:
        rfx -= yi*xyz[2]/(nt*(xi**2+yi**2))
        rfy += xi*xyz[2]/(nt*(xi**2+yi**2))
    x0[2*i] = (rfx**2+rfy**2)**0.5+eps
    # angles
    if len(acn[i]) == 1:
        x0[2*i+1] = acn[i][0]
    elif rfx or rfy:
        ang = atan2(rfy,rfx)
        if len(acn[i]) > 1:
            x0[2*i+1] = min(max(ang,acn[i][0]),acn[i][1])
        else:
            x0[2*i+1] = ang
print('Initial Value:')
print(x0)

### Solve the optimization problem
solution = minimize(objective,
                    x0, 
                    constraints=constraints, 
                    method='SLSQP',
                    tol=1e-6, 
                    options={'disp': True, 'maxiter': 100})

### Extract results
x_opt = solution.x
for i in range(nt):
    x_opt[2*i+1] = x_opt[2*i+1]%(2*pi)*180/pi
f_opt = solution.fun

### Print the results
print(f'Execution Time: {(time.time()-start)*1000:.3f} ms')
print("Optimal Solution:")
print(x_opt)
print(f"Optimal Objective Value = {f_opt:.3f}")
