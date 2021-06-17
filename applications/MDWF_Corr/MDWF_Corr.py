import gpt as g
import sys

###
# function to check for a bool
###
def get_bool(tag,default):
    res = default
    for i, x in enumerate(sys.argv):
        if x == tag:
            res = not default
    return res


# get input parameter
grid=get_ivec("--grid",[8,8,8,8])
Ls=get_int("-Ls",8)
M5=get_float("-M5",1.8)
b5=get_float("-b5",1.5)
c5=get_float("-c5",0.5)
mass_dict={}
ml_val=get_float("-ml",None)
if ml_val != None:
    mass_dict["l"]=ml_val

ml_val=get_float("-ms",None)
if ms_val != None:
    mass_dict["s"]=ms_val

conf_name=get("-conf","unit")


# Double-precision grid
grid = g.grid(grid, g.double)


if conf_name == "random":

    # Parallel random number generator
    rng = g.random("seed text")

    # Random gauge field
    U = g.qcd.gauge.random(grid, rng)
elif conf_name == "unit":
    U = g.qcd.gauge.unit(grid)
else:
    U = g.load(conf_name)


fermion_dict=dict()
# Mobius domain-wall fermion
for flavor in mass_dict:
    fermion_dict[flavor] = g.qcd.fermion.mobius(U, mass=mass_dict[flavor], M5=M5, b=b5, c=c5, Ls=Ls,
                               boundary_phases=[1,1,1,-1])

# Short-cuts
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner

# Even-odd-preconditioned CG solver
slv_5d = inv.preconditioned(pc.eo2_ne(), inv.cg(eps = 1e-16, maxiter = 10000))

# Abstract fermion propagator using this solver
fermion_prop_dict=dict()
for fermion in fermion_dict:
    fermion_propagator[fermion] = fermion_dict[fermion].propagator(slv_5d)

# Create point source
src = g.mspincolor(U[0].grid)
g.create.point(src, [0, 0, 0, 0])

prop_dict=dict()
# Solve propagator on 12 spin-color components
for ferm_prop in fermion_prop_dict:
    prop_dict[ferm_prop] = g( fermion_propagator[ferm_prop] * src )

key,prop = zip(*prop_dict.items())
# Pion correlator
for i in range(len(key)):
    for j in range(i,len(key)):
        g.message(g.slice(g.trace(prop[i] * g.adj(prop[j])), 3))
