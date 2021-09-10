import gpt as g



# Fill cache before doing wanted calculations

grid2 = g.grid([16,16,16,16], g.double)

g.message(grid2)

g.mem_report()

#create field of complex numbers

comp = g.complex(grid2)
comp[:]=0

g.message(c)

g.mem_report()

# Double-precision 8^4 grid
grid = g.grid([8,8,8,8], g.double)

# Parallel random number generator
#rng = g.random("seed text")

# Random gauge field
U = g.qcd.gauge.unit(grid)

# Mobius domain-wall fermion
fermion = g.qcd.fermion.mobius(U, mass=0.1, M5=1.8, b=1.5, c=0.5, Ls=12,
                               boundary_phases=[1,1,1,-1])
#g.mem_report()
# Short-cuts
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner

# Even-odd-preconditioned CG solver
slv_5d = inv.preconditioned(pc.eo2_ne(), inv.cg(eps = 1e-4, maxiter = 1000))

#D5_inv=slv_5d(fermion)

# Abstract fermion propagator using this solver

#fermion_propagator = fermion.propagator(slv_5d)

# Create 4D point source
src4D = g.mspincolor(U[0].grid)
g.create.point(src4D, [0, 0, 0, 0])

imp = fermion.ImportPhysicalFermionSource
exp = fermion.ExportPhysicalFermionSolution

def prop5D(dst_sc,src_sc):
    g(dst_sc, slv_5d(fermion)  * imp * src_sc)
    
#g.message(fermion.otype)
#g.message(imp.otype)

g.mem_report()

D5_inv=g.matrix_operator(
            prop5D,
            otype=(fermion.otype[0], imp.otype[1]),
            grid=(fermion.F_grid, imp.grid[1]),
            accept_list=True,
        ).grouped(1)

#propagator5D = g( D5_inv * src4D)
#propagator4D = g( exp * D5_inv * src4D)

g.mem_report()
# Solve propagator on 12 spin-color components

#g.message("Calculate 5D propagator")
#prop5D = g( D5_inv * src5D )
#g.message("Extract 4D propagator")
#prop4D = g( exp * prop5D )
#
#g.message(prop5D)
#g.message(prop4D)
#
#g.message("Dimension of 5D propagator:")
#g.message(len(prop5D))
#g.message("Shape of 5D propagator:")
#g.message(np.shape(prop5D))
#g.message("Dimension of 4D propagator:")
#g.message(len(prop4D))
#g.message("Shape of 4D propagator:")
#g.message(np.shape(prop4D))

# Pion correlator
#g.message(g.slice(g.trace(prop * g.adj(prop)), 3))
