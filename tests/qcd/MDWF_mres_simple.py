import gpt as g
import numpy as np

def get_Jq5(prop5D):
    # get all Ls slices
    prop4DLs=g.separate(prop5D,0)
    # create Correlator at the midpoint of the 5-th direction
    p_plus=g(prop4DLs[int((Ls/2)-1)] + g.gamma[5] * prop4DLs[int((Ls/2)-1)] )
    p_minus=g(prop4DLs[int(Ls/2)] - g.gamma[5] * prop4DLs[int(Ls/2)] )

    p=g(0.5 * (p_plus + p_minus))
    # evaluate Jq5 and return it
    return g.slice(g.trace(p * g.adj(p)),3)

# create a string array with all wanted gamma combinations
G_single=np.array(["I","5","X","Y","Z","T"])
Gammas=np.array(G_single)
tmp=np.array([])
for mu in G_single[1:]:
    for nu in G_single[2:]:
        if mu == nu:
            continue
        else:
            Gammas=np.append(Gammas,"{}.{}".format(mu,nu))
            if mu != "5":
                tmp=np.append(tmp,"5.{}.{}".format(mu,nu))

Gammas=np.append(Gammas,tmp)
# check length
#g.message("#GAMMAS = {}".format(len(Gammas)))
# check different combinations
#g.message("Gammas:")
#for i in Gammas:
#    g.message(i)

# Double-precision 8^4 grid
Dims=[8,8,8,16]
grid = g.grid(Dims, g.double)
Ls=12
# momentum
# additional correlator channel
Gopt=g.default.get_single("-G",None)

if Gopt:
    Gammas=np.append(Gammas,Gopt)

#k=1
k=g.default.get_int("-k",0)
p= 2.0 * np.pi * np.array([0,0,int(k),0])/(Dims[0])
# exp(ix*p)
P=g.exp_ixp(p)

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
#g.message("Ls = ",len(prop5D[:,0,0,0,0]))
#g.message("Nx = ",len(prop5D[0,:,0,0,0]))
#g.message("Ny = ",len(prop5D[0,0,:,0,0]))
#g.message("Nz = ",len(prop5D[0,0,0,:,0]))
#g.message("Nt = ",len(prop5D[0,0,0,0,:]))
#




Jq5=get_Jq5(prop5D)

g.message("Jq5:")
#g.message("real\t\t\timag")
#for i in range(len(Jq5)):
#    g.message(f"{Jq5[i].real}\t{Jq5[i].imag}")

header='t\t\t\t\tJq5'
data=np.array([[i for i in range(len(Jq5))],[Jq5[i].real for i in range(len(Jq5))]])

for comb in Gammas:
    GMats=comb.split('.')
    #g.message(GMats)
    col="G" + "G".join(GMats)    
    g.message(col + " correlator")
    if GMats[0] == "5":
        G=g.gamma[5]
    else:        
        G=g.gamma[GMats[0]]

    for ind in GMats[1:]:
        if ind == "5":
            G = G * g.gamma[5]
        else:
            G= G * g.gamma[ind]
        

    tCorr=g.slice(g.trace( P * G * prop4D * G * g.gamma[5] * g.adj(prop4D) * g.gamma[5] ), 3)

    #g.message("real\t\t\timag")
    #for i in range(len(tCorr)):
    #    #g.message("{}\t{}".format(tCorr[i].real,tCorr[i].imag))
    #    g.message(f"{tCorr[i].real}\t{tCorr[i].imag}")

    header+='\t\t\t' + col
    data=np.append(data,[[tCorr[i].real for i in range(len(tCorr))]],axis = 0)

np.savetxt("./test_output_kz{}".format(k),data.T,header=header,delimiter="\t",comments='#')