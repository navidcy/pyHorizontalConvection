# import matplotlib.pyplot as plt
import numpy as np
import h5py

diags = h5py.File("diagnostics/diagnostics_s1.h5")

# simulation parameters
Lx = 4
Lz = Lx / 4
Ra, Pr = 5e9, 1.

kappa = 1 / (Ra*Pr)**(1/2)
k=2*np.pi/Lx

Tkappa = (Lz**2)/kappa 

# the conductive solution
chic = (kappa * k / (2*Lz)) * np.tanh(k * Lz)

# get diagnostics
time = diags["scales/sim_time"][:]
ke = diags['tasks']['ke'][:,0,0]
chi = diags['tasks']['chi'][:,0,0]
wb = diags['tasks']['wb'][:,0,0]
u2 = diags['tasks']['u2'][:,0,0]
v2 = diags['tasks']['v2'][:,0,0]
w2 = diags['tasks']['w2'][:,0,0]
bx2 = diags['tasks']['bx2'][:,0,0]
by2 = diags['tasks']['by2'][:,0,0]
bz2 = diags['tasks']['bz2'][:,0,0]

#ep = diags['tasks']['ep'][:,0,0]
#b = (diags['tasks']['b'][:,:,0]).mean(axis=1) # bottom buoyancy

#ke_t = np.gradient(ke,time)

# Nusselt number
Nu=chi/chic

np.savez("NuAndKE_3D_nostress_5e9.npz", time=time, Nu=Nu, ke=ke)
