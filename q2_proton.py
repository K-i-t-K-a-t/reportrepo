# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 18:33:10 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
m_e = 9.10938e-31
e = -1.60217663e-19
B_0 = 3.03e-5

# field strength
L = 4
E_x = 100e-3
B_z = B_0/(L**3)

# simulation parameters
dt = 1e-8
omega_g = e*B_z/m_e
T_g = 2 * np.pi / np.abs(omega_g)
iterations = int(np.ceil(4*T_g/dt))

print(f"Number of iterations: {iterations}")
print(f"Gyration frequency: {omega_g} rad/s")

# initial conditions
r = [[0, 0, 0]]*iterations
E_k = [1000*e]*iterations
v = [[np.sqrt(2*E_k[0]/m_e), 0, 0]]*iterations

# Larmor radius
r_g = v[0][0]/omega_g
print(f"Gyration radius: {r_g} m")

# Main loop
for i in range(iterations-1):
    F = [-e*B_z*v[i][1] + e*E_x, e*B_z*v[i][0], 0]
    r[i+1] = [v[i][j]*dt + r[i][j] for j in range(3)]
    v[i+1] = [F[j]/m_e*dt + v[i][j] for j in range(3)]
    E_k[i+1] = m_e/2*(v[i+1][0]**2+v[i+1][1]**2)
    
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
plt.show()

# t = [dt * i for i in range(iterations)]

# plt.plot(t, E_k)
# plt.show()