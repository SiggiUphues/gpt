import gpt as g
import numpy as np

# Double-precision 8^4 grid
grid = g.grid([8,8,8,8], g.double)

# Parallel random number generator
#rng = g.random("seed text")

# Random gauge field
U = g.qcd.gauge.unit(grid)

# Mobius domain-wall fermion
fermion = g.qcd.fermion.mobius(U, mass=0.1, M5=1.8, b=1.5, c=0.5, Ls=12,
                               boundary_phases=[1,1,1,-1])

# Short-cuts
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner

# Even-odd-preconditioned CG solver
slv_5d = inv.preconditioned(pc.eo2_ne(), inv.cg(eps = 1e-8, maxiter = 1000))

# 5D propagator
D5_inv=slv_5d(fermion)

# Create 4D point source
src4D = g.mspincolor(U[0].grid)
g.create.point(src4D, [0, 0, 0, 0])

# Short-cuts
imp = fermion.ImportPhysicalFermionSource
exp = fermion.ExportPhysicalFermionSolution

# get 5D source from 4D source
src5D =  g(imp * src4D)

# Solve propagator on 12 spin-color components

g.message("Calculate 5D propagator")
prop5D = g( D5_inv * src5D )
g.message("Extract 4D propagator")
prop4D = g( exp * prop5D )
#
#g.message(prop5D)
#g.message(prop4D)
#
g.message("Dimension of 5D propagator:")
#g.message(len(prop5D))
g.message("Shape of 5D propagator:")
g.message(np.shape(prop5D))
g.message("Dimension of 4D propagator:")
#g.message(len(prop4D))
g.message("Shape of 4D propagator:")
g.message(np.shape(prop4D))

# Pion correlator
#g.message(g.slice(g.trace(prop * g.adj(prop)), 3))
