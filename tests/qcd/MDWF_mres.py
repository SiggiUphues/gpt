import gpt as g
import numpy as np



# Double-precision 8^4 grid
grid = g.grid([8,8,8,16], g.double)
Ls=12

# Parallel random number generator
#rng = g.random("seed text")

# Random gauge field
U = g.qcd.gauge.unit(grid)

# Mobius domain-wall fermion
fermion = g.qcd.fermion.mobius(U, mass=0.1, M5=1.8, b=1.5, c=0.5, Ls=Ls,
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
#g.message(np.shape(prop5D[Ls-1,:,:,:,:]))
#g.message(prop5D.otype)
#g.message(prop4D.otype)
g.message("Ls = ",len(prop5D[:,0,0,0,0]))
g.message("Nx = ",len(prop5D[0,:,0,0,0]))
g.message("Ny = ",len(prop5D[0,0,:,0,0]))
g.message("Nz = ",len(prop5D[0,0,0,:,0]))
g.message("Nt = ",len(prop5D[0,0,0,0,:]))
#

#p_plus=g(prop4D[0,:,:,:])# + g.gamma[5] * prop5D[(Ls/2)+1,:,:,:,:])
#g.message(np.shape(prop5D.otype))
#g.message(np.shape(prop5D[int((Ls/2)+1),:,:,:,:]))
#g.message(g.lattice(prop5D[int((Ls/2)+1),:,:,:,:]))
propLshm1=g.mspincolor(grid)
#g.message("propLshm1")
#g.message("Shape before the values are set:")
#g.message(np.shape(propLshm1[:]))
propLshm1[:]=prop5D[int((Ls/2)-1),:,:,:,:]
#g.message("Shape After the values are set:")
#g.message(np.shape(propLshm1[:]))
#g.message(propLshm1.otype)
p_plus=g(propLshm1 + g.gamma[5] * propLshm1)

propLsh=g.lattice(prop4D)
#g.message("propLsh")
#g.message("Shape before the values are set:")
#g.message(np.shape(propLsh[:]))
propLsh[:]=prop5D[int((Ls/2)),:,:,:,:]
#g.message("Shape After the values are set:")
#g.message(np.shape(propLsh[:]))
#g.message(propLsh.otype)
p_minus=g(propLsh - g.gamma[5] * propLsh)

p= g(0.5 * (p_plus + p_minus ))

g.message("J5q:")
Jq5=g.slice(g.trace(p * g.adj(p)),3)

g.message(len(Jq5))
g.message("real\t\t\timag")
for i in range(len(Jq5)):
    g.message(f"{Jq5[i].real}\t{Jq5[i].imag}")

g.message("Pion correlator")
pion=g.slice(g.trace(prop4D * g.adj(prop4D)), 3)

g.message("real\t\t\timag")
for i in range(len(pion)):
    #g.message("{}\t{}".format(pion[i].real,pion[i].imag))
    g.message(f"{pion[i].real}\t{pion[i].imag}")
