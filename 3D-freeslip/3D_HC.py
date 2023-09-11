import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Ly, Lz = (4., 1., 1.)
Ra = 5e9
Pr = 1
k = np.pi / Lx

# Create bases and domain
x_basis = de.Fourier('x', 1024, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', 256, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', 256, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=[64, 32])

# 3D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p', 'b', 'u', 'v', 'w', 'bz', 'uz', 'vz'])
problem.meta[:]['z']['dirichlet'] = True
problem.parameters['P'] = (Ra * Pr)**(-1/2)
problem.parameters['R'] = (Ra / Pr)**(-1/2)
problem.parameters['k'] = k

problem.substitutions["bx"] = "dx(b)"
problem.substitutions["by"] = "dy(b)"
problem.substitutions["ux"] = "dx(u)"
problem.substitutions["uy"] = "dy(u)"
problem.substitutions["vx"] = "dx(v)"
problem.substitutions["vy"] = "dy(v)"
problem.substitutions["wx"] = "dx(w)"
problem.substitutions["wy"] = "dy(w)"

problem.add_equation("ux + vy + dz(w) = 0")
problem.add_equation("dt(b) - P * (dx(bx) + dy(by) + dz(bz))                      = - (u*bx + v*by + w*bz)")
problem.add_equation("dt(u) - R * (dx(ux) + dy(uy) + dz(uz)) + dx(p)              = - (u*ux + v*uy + w*uz)")
problem.add_equation("dt(v) - R * (dx(vx) + dy(vy) + dz(vz)) + dy(p)              = - (u*vx + v*vy + w*vz)")
problem.add_equation("dt(w) - R * (dx(wx) + dy(wy) - dx(uz) - dy(vz)) + dz(p) - b = - (u*wx + v*wy - w*(ux+vy))")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_bc("left(bz) = 0")
problem.add_bc("left(uz) = 0")
problem.add_bc("left(vz) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = cos(2*k*x)")
problem.add_bc("right(uz) = 0")
problem.add_bc("right(vz) = 0")
problem.add_bc("right(w) = 0", condition="(nx !=0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF3)

logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
# cold start
b['g'] = -0.76
b.differentiate('z', out=bz)

# Initial timestep
dt = 0.00125


# Integration parameters
solver.stop_sim_time = 6000
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=20, max_writes=20)
snapshots.add_system(solver.state)
 
# Averaged sections
analysis1 = solver.evaluator.add_file_handler("2d_averages", sim_dt=1, max_writes=50)
analysis1.add_task("integ(b,'y')", name="b")
analysis1.add_task("integ(bz,'y')", name="bz")
analysis1.add_task("integ(u,'y')", name="u")
analysis1.add_task("integ(w,'y')", name="w")

# diagnostics
analysis2 = solver.evaluator.add_file_handler("diagnostics", iter=10, max_writes=200)
analysis2.add_task("integ(0.5 * (u*u + v*v +  w*w))/4", name="ke")
analysis2.add_task("integ(0.5 * (u*u))/4", name="u2")
analysis2.add_task("integ(0.5 * (v*v))/4", name="v2")
analysis2.add_task("integ(0.5 * (w*w))/4", name="w2")
analysis2.add_task("integ( P*(bx*bx + by*by + bz*bz))/4", name="chi")
analysis2.add_task("integ( P*(bx*bx) )/4", name="bx2")
analysis2.add_task("integ( P*(by*by) )/4", name="by2")
analysis2.add_task("integ( P*(bz*bz) )/4", name="bz2")
analysis2.add_task("integ( R*( ux*ux + uy*uy + uz*uz + vx*vx + vy*vy + vz*vz + wx*wx + wy*wy + (ux+vy)*(ux+vy) ) )/4", name="ep")
analysis2.add_task("integ(w*b)/4", name="wb")

# new diagnostics
analysis2.add_task("integ( integ( P*(bx*bx + by*by + bz*bz), 'x'), 'y')/4", layout='g', name="Fchi")
analysis2.add_task("integ( integ( R*(ux*ux + uy*uy + uz*uz + vx*vx + vy*vy + vz*vz + wx*wx + wy*wy + (ux+vy)*(ux+vy)), 'x'), 'y')/4", layout='g', name="Fep")
analysis2.add_task("interp(integ(integ(bz*cos(1.5707963267948966*x), 'x'), 'y')/4, z=1)", layout='g', name="b1_z")

analysis2.add_task("interp(integ(b , 'y'), z=0)", layout='g', name='botb')
analysis2.add_task("interp(integ(bz, 'y'), z=1)", layout='g', name="bztop")
analysis2.add_task("interp(integ(u , 'y'), z=1)", layout='g', name='utop')  # set to uz in noslip
analysis2.add_task("interp(integ(v , 'y'), z=1)", layout='g', name='vtop')  # set to vz in noslip
analysis2.add_task("interp(integ(dz(uz), 'y'), z=1)", layout='g', name='uzztop')


# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=2, safety=0.4,
                      max_dt=0.125, threshold=0.)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w) / R", name='Re')
flow.add_property("(u*u + v*v + w*w)/2", name='K')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 50 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max KE = %f' %flow.max('K'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
