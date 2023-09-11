import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob, os
from matplotlib import gridspec

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

plt.close("all")

# load snapshots
fnis = glob.glob("2d_averages/*.h5")
fnis.sort(key=os.path.getmtime)


snaps = h5py.File(fnis[-1])

# snap time
time = snaps["scales/sim_time"][:]
x = snaps['scales/x/1.0'][:]
z = snaps['scales/z/1.0'][:]

b =  np.squeeze(snaps['tasks']['b'][0])
usnap =  np.squeeze(snaps['tasks']['u'][0])
wsnap =  np.squeeze(snaps['tasks']['w'][0])

# Create bases and domain
# Parameters
Lx, Lz = (4., 1.)
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 64, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.LBVP(domain, variables=['psi','psiz'])

u = domain.new_field()
uz = domain.new_field()
wx = domain.new_field()

u['g'] = usnap
w = domain.new_field()
w['g'] = wsnap

u.differentiate(1,out=uz)
w.differentiate(0,out=wx)

problem.parameters['uz'] = uz
problem.parameters['u'] = u
problem.parameters['w'] = w
problem.parameters['wx'] = wx
problem.add_equation("dz(psiz) + dx(dx(psi))  = - uz + wx")
problem.add_equation("psiz - dz(psi) = 0")
problem.add_bc("right(psi) = 0")
problem.add_bc("left(psi) = 0")

# Build solver
solver = problem.build_solver()
logger.info('Solver built')

solver.solve()

psi = solver.state['psi']['g']


fig = plt.figure(figsize=(10.5,4.85))

gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2.5])

cp = np.linspace(-1.2,1.2,20)

ax1 = plt.subplot(gs[0])
#plt.contourf(x,z[200:],psi[:,200:].T*100,cp)
plt.contour(x,z[50:],psi[:,50:].T*100,cp,vmin=cp.min(),vmax=cp.max())

plt.ylim(0.9,1.)
plt.xlim(0,4)
#plt.xlabel(r'$x/h$')
#plt.ylabel(r'$z/h$')
plt.text(0,1.0025,"(a)")

#ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])


plt.text(2.75,1.0025,"Streamfunction")

ax2 = plt.subplot(gs[2])
#im_psi  = plt.contourf(x,z,psi.T*100,cp)
plt.contour(x,z,psi.T*100,cp,vmin=cp.min(),vmax=cp.max())

#ax2.fill_between(x, 0.9, 1.,  facecolor='.5', interpolate=True,alpha=0.3)
#ax2.plot([0,4],[.9]*2,'k--')

plt.xlabel(r'$x/h$')
plt.ylabel(r'         $\,\,\,\,\,z/h$')
plt.xlim(0,4)
plt.ylim(0,0.9)

plt.xticks([0,1,2,3,4])
plt.yticks([0,0.2,.4,.6,.8],['0.00','0.20','0.40','0.60','0.80'])

plt.subplots_adjust(wspace=.275, hspace=.04)

ax3 = plt.subplot(gs[1])

cb = np.linspace(-1.,1,25)
plt.contourf(x,z[50:],b[:,50:].T,cb,cmap='RdBu_r',vmin=cb.min(),vmax=cb.max())
plt.contour(x,z[50:],b[:,50:].T,[-.751,-.75],colors='0.2',linewidths=0.65)
#plt.pcolormesh(x,z[200:],b[:,200:].T,cmap='RdBu_r',vmin=cb.min(),vmax=cb.max())
#plt.contour(x,z[200:],b[:,200:].T,cb,colors='w',vmin=cb.min(),vmax=cb.max())
#plt.pcolormesh(x,z[200:],b[:,200:].T,cmap='RdBu_r',vmin=-1,vmax=1)
plt.ylim(0.9,1.)
#plt.xlabel(r'$x/h$')
#plt.ylabel(r'      $z/h$')
plt.text(0,1.0025,"(b)")

#ax1.spines['bottom'].set_visible(False)
ax3.set_xticks([])
plt.xlim(0,4)
plt.ylim(.9,1)
plt.text(3.2,1.0025,"Buoyancy")

ax4 = plt.subplot(gs[3])

cb = np.linspace(-1.,1,60)
im_b=plt.contourf(x,z,b.T,cb,cmap='RdBu_r',vmin=cb.min(),vmax=cb.max())
#im_b = plt.pcolormesh(x,z,b.T,cmap='RdBu_r',vmin=cb.min(),vmax=cb.max())

plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(x,z,b.T,[-.750001,-.75],colors='0.2',linewidths=0.65)
#plt.clabel(CS, fontsize=9, inline=1)


#plt.contour(x,z,b.T,cb,colors='w',vmin=cb.min(),vmax=cb.max())

#im_b=plt.pcolormesh(x,z,b.T,cmap='RdBu_r',vmin=-1.,vmax=1)
#ax4.fill_between(x, 0.9, 1.,  facecolor='.5', interpolate=True,alpha=0.3)
#ax4.plot([0,4],[.9]*2,'k--')

plt.xlabel(r'$x/h$')
plt.ylabel(r'        $\,\,\,\,\,z/h$')
plt.xlim(0,4)
plt.ylim(0,.9)
plt.xticks([0,1,2,3,4])
plt.yticks([0,0.2,.4,.6,.8],['0.00','0.20','0.40','0.60','0.80'])
# cbar_ax = fig.add_axes([.43, .25, 0.02, 0.5])
# fig.colorbar(im_psi, cax=cbar_ax)

cbar_ax = fig.add_axes([.93, .225, 0.02, 0.5])
fig.colorbar(im_b, cax=cbar_ax,ticks=[-1,-0.5,0,.5,1.],label=r'$b/b_*$')


plt.savefig("Figure1_alt.png",dpi=800, bbox_inches = 'tight',pad_inches = 0)
plt.savefig("Figure1_alt.eps", bbox_inches = 'tight',pad_inches = 0)


#fig = plt.figure(figsize=(12,5))
#ax = fig.add_subplot(121)
#plt.contourf(x,z,b.T)
