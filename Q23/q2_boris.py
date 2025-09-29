# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 18:33:10 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
m = 9.10938e-31 # 1.67262192e-27
q = -1.60217663e-19 # 1.60217663e-19
B_0 = 3.03e-5

# field strength
L = 4
E_x = 0 # 100e-3
B_z = B_0/(L**3)

# simulation parameters
dt = 1e-6
omega_g = q*B_z/m
T_g = 2 * np.pi / np.abs(omega_g)
iterations = int(np.ceil(4*T_g/dt))

print(f"Number of iterations: {iterations}")
print(f"Gyration frequency: {omega_g} rad/s")

# initial conditions
r = [[0, 0, 0]]*iterations
E_k = [1000*np.abs(q)]*iterations
v = [[np.sqrt(2*E_k[0]/m), 0, 0]]*iterations

# Larmor radius
r_g = v[0][0]/np.abs(omega_g)
print(f"Gyration radius: {r_g} m")

# Main loop
for i in range(iterations-1):
    u = v[i][:]
    u[0] += q*E_x/(2*m) * dt
    
    h = omega_g/2 * dt
    s = 2*h/(1+h**2)
    
    u_intermediate = u[:]
    u_intermediate[0] += u[1] * h
    u_intermediate[1] -= u[0] * h
    
    u_prime = u[:]
    u_prime[0] += u_intermediate[1] * s
    u_prime[1] -= u_intermediate[0] * s
    
    v[i+1] = u_prime[:]
    v[i+1][0] += q*E_x/(2*m) * dt
    
    r[i+1] = [r[i][j] + v[i+1][j] * dt for j in range(3)]
    
    E_k[i+1] = m/2*(v[i+1][0]**2 + v[i+1][1]**2)
    
x = [pos[0] for pos in r]
y = [pos[1] for pos in r]

dt_label = f"dt = {dt}"
iterations_label = f"n = {iterations}"

plt.plot(x, y)
plt.plot([], [], ' ', label=dt_label)
plt.plot([], [], ' ', label=iterations_label)
plt.legend(loc="upper right")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.show()

t = [dt * i for i in range(iterations)]

plt.plot(t, E_k)
plt.plot([], [], ' ', label=dt_label)
plt.plot([], [], ' ', label=iterations_label)
plt.legend(loc="upper right")
plt.xlabel('t (s)')
plt.ylabel('E_k (J)')
plt.show()