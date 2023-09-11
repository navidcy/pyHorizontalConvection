import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob, os

plt.close("all")

# load snapshots
fnis = glob.glob("2d_averages/*.h5")
fnis.sort(key=os.path.getmtime)

time, b = np.array([]),np.array([])



for fni in fnis[:]:

    snaps = h5py.File(fni)

    # snap time
    times = snaps["scales/sim_time"][:]
    time = np.hstack([time,snaps["scales/sim_time"][:]])
    #bb =  snaps['tasks']['b'][...,0].mean(axis=1)
    b = np.hstack([b, (snaps['tasks']['b'][...,0,0]).mean(axis=1)])

time, b = np.array(time), np.array(b)

fig = plt.figure(figsize=(8.5,4))

ax = fig.add_subplot(111)

plt.plot(time,b)

plt.xlabel("Time")
plt.ylabel(r"Bottom buoyancy [$b/b_*$]")

plt.savefig("Figure_BottomBuoyancy.png")
plt.savefig("Figure_BottomBuoyancy.eps")

np.savez('BottomBuoyancy_3D_noslip_1e6.npz',time=time,b=b)



