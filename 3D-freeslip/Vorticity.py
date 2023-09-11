import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob, os

plt.close("all")

# load snapshots
fnis = glob.glob("snapshots/*.h5")
fnis.sort(key=os.path.getmtime)

fni = fnis[-1]
snaps = h5py.File(fni)

# snap time
times = snaps["scales/sim_time"][-1]
x = snaps['scales/x/1.0'][:]
y = snaps['scales/y/1.0'][:]
z = snaps['scales/z/1.0'][:]
b = snaps['tasks']['b'][-1,64]
v = snaps['tasks']['v'][-1,64]
w = snaps['tasks']['w'][-1,64]

wy = np.gradient(w,y,axis=0)
vz = np.gradient(v,z,axis=1)

wy[:,0] = np.nan


from scipy.interpolate import InterpolatedUnivariateSpline



viz  = np.zeros_like(vz)

for j in range(y.size):
    f = InterpolatedUnivariateSpline(z, v[j], k=3)
    for i in range(z.size):
        viz[j,i]  = f.derivatives(z[i])[0]


omgx = wy-viz

fig = plt.figure(figsize=(10,4))

cv = np.linspace(-0.04,0.04,20)
co = np.linspace(-0.02,0.02,20)

ax = fig.add_subplot(131,aspect=1)
plt.contourf(y,z,v.T,cv,cmap='RdBu_r',extend='both')

plt.xlabel(r'$y/h$')
plt.ylabel(r'$z/h$')
plt.title(r'$v$')

ax = fig.add_subplot(132,aspect=1)
plt.contourf(y,z,w.T,cv,cmap='RdBu_r',extend='both')
plt.yticks([])

plt.xlabel(r'$y/h$')
plt.title(r'$w$')

ax = fig.add_subplot(133,aspect=1)
plt.contourf(y,z,omgx.T,co,cmap='RdBu_r',extend='both')
plt.yticks([])

plt.xlabel(r'$y/h$')
plt.title(r'$w_y - v_z$')

plt.savefig('vorticity.png')
