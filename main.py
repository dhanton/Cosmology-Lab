#!/usr/bin/env python

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import fsolve
import scipy.io as io
import time

import matplotlib.pyplot as plt

def dl(z, omega_m, w):
    def g(x):
        return 1/(np.sqrt(omega_m * (1 + x)**3 + (1 - omega_m) * (1 + x) ** (3 * (1 + w))))
        
    return (1 + z) * quad(lambda y: g(y), 0, z)[0]

#Exercise 1
vectorized_dl = np.vectorize(dl)

#Exercise 2
z = np.arange(0, 1.5, 0.1)
plot_result = vectorized_dl(z, 0.3, -1)

plt.figure()
plt.plot(z, plot_result)
plt.xlabel("$z$",fontsize = 15)
plt.ylabel("$d_L$",fontsize = 15, rotation=0)
plt.savefig("ejercicio_2.png")
plt.show()

print("d_L(0.5, 0.3. -1)", dl(0.5, 0.3, -1))

#Exercise 3
def marginalized_M(z, omega_m, w, data_mu, data_error):
    return np.sum((data_mu - 5 * np.log10(vectorized_dl(z, omega_m, w)))/data_error**2)/np.sum(1/data_error**2)

data = io.loadmat('Union2.mat')
z, = data['z']
m, = data['m']
dm, = data['dm']

#Exercise 4
def model_mu(z, omega_m, w, M):
    return 5 * np.log10(vectorized_dl(z, omega_m, w)) + M

def marginalized_chi(omega_m, w):
    M_m = marginalized_M(z, omega_m, w, m, dm)
    m_model = model_mu(z, omega_m, w, M_m)
    
    return np.sum(((m - m_model)**2)/dm**2)

print("Marginalized Chi for (0.3, -1)", marginalized_chi(0.3, -1))

#Exercise 5
min_result = minimize(marginalized_chi, 0.25, args=(-1,))

best_omega = min_result.x
best_chi = min_result.fun

print("Best Omega", best_omega)
print("Best Chi", best_chi)

#Exercise 6
def error_fun(omega_m, w):
    return np.abs(marginalized_chi(omega_m, w) - best_chi - 1)

left_omega = fsolve(error_fun, 0.23, args=(-1,))
right_omega = fsolve(error_fun, 0.28, args=(-1,))

left_distance = best_omega - left_omega
right_distance = best_omega - right_omega

print("Left error", left_distance)
print("Rigth error", right_distance)

#Exercise 7
einstein_desitter_chi = marginalized_chi(1, -1)
print("Einstein-de Sitter Chi", einstein_desitter_chi)

def marginalized_model_mu(z, omega_m, w):
    M_m = marginalized_M(z, omega_m, -1, m, dm)
    
    return model_mu(z, omega_m, w, M_m)

plt.figure()
plt.errorbar(z, m, yerr=dm, fmt='.')
plt.plot(z, marginalized_model_mu(z, best_omega, -1), color='green')
plt.plot(z, marginalized_model_mu(z, 1, -1))
plt.xlabel("z",fontsize = 15)
plt.ylabel("$\mu$",fontsize = 15, rotation=0)
plt.savefig("ejercicio_7.png")
plt.show()

print("Delta of Chi(Einstein-de Sitter) and Best Chi", np.sqrt(einstein_desitter_chi - best_chi))

#Exercise 8
supernova_distance = 4285 * dl(1, best_omega, -1)
print("Supernova distance", supernova_distance)

def comoving_distance(z):
    a = 1/(1 + z)
    
    return supernova_distance*a
    

print("Comoving distance", comoving_distance(1))

#Exercise 9
initial_time = time.time()

N = 100
grid_omega = np.linspace(0, 0.5, N)
grid_w = np.linspace(-2, -0.4, N)

Z = np.zeros((np.size(grid_omega), np.size(grid_w)))
for i in range(np.size(grid_omega)):
    for j in range(np.size(grid_w)):
        Z[j, i] = marginalized_chi(grid_omega[i], grid_w[j])

best_fit_indices = np.unravel_index(Z.argmin(), (N, N))
multi_best_omega = grid_omega[best_fit_indices[0]]
multi_best_w = grid_w[best_fit_indices[1]]

print("wCDM best omega", multi_best_omega)
print("wCDM best w", multi_best_w)

levels = Z.min() + [0, 2.30, 6.18, 11.83]
xx, yy = np.meshgrid(grid_omega, grid_w)

plt.figure()
plt.contour(xx, yy, Z, levels, colors='k')
plt.contourf(xx, yy, Z, levels)
plt.xlabel("$\Omega_M$", fontsize=15)
plt.ylabel("$\omega$", fontsize=15, rotation=0)
plt.yticks([-2, -1.6, -1.2, -0.8, -0.4])
plt.savefig("ejercicio_9.png")
plt.show()

final_time = time.time() - initial_time

print("Countour runtime: ", int(final_time/60), " minutes, ", int(final_time%60), " seconds.")
