import gpt as g
import numpy as np

def get_vec(tag,type, default, ndim):
    res = g.default.get_all(tag, None)
    if res[0] is None:
        return default
    if type == "s":
        for x in res:
            v = [y for y in x.split(".")]
            if len(v) == ndim:
                return v
    if type == "f":
        for x in res:
            v = [float(y) for y in x.split(",")]
            if len(v) == ndim:
                return v
    if type == "i":
        for x in res:
            v = [int(y) for y in x.split(".")]
            if len(v) == ndim:
                return v
    return default

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
# check different combinationsd
#g.message("Gammas:")
#for i in Gammas:
#    g.message(i)
# command line options
# Ls,M5,b5,c5,out_folder,out_name,conf_name,ms,ml,--grid,momentum,additional corr-channel
Dims=get_vec("--grid","i",[8,8,8,8],4)
Ls=g.default.get_int("-Ls",12)
M5=g.default.get_float("-M5",1.8)
b5=g.default.get_float("-b5",1.5)
c5=g.default.get_float("-c5",0.5)
nflavs=g.default.get_int("-nflavs",2)
flav_names=get_vec("-flav-names","s",["l","s"],nflavs)
flav_masses=get_vec("-flav-masses","f",[0.01,0.1],nflavs)
assert len(flav_names) == len(flav_masses), "-flav_names and -flav_masses must have the same length"
assert len(flav_names) == nflavs, "-nflavs specifies the number of flavors and is not equal to the number of "

resid=g.default.get_float("-resid",1e-8)
max_it=g.default.get_int("-max-it",1000)
conf_name=g.default.get_single("-conf-name","unit")
out_name_add=g.default.get_single("-out-name-add","res")
out_folder=g.default.get_single("-out-folder",".")
# momentum
#k=1
k=np.array(get_vec("-k","i",[0,0,0,0],4))
k=k.astype(int)
p= 2.0 * np.pi * np.array(k/(Dims[0]))
# additional correlator channel
Gopt=g.default.get_single("-G",None)
if Gopt:
    Gammas=np.append(Gammas,Gopt)

# Double-precision 8^4 grid
#Dims=[8,8,8,16]
grid = g.grid(Dims, g.double)

# exp(ix*p)
P=g.exp_ixp(p)

# Parallel random number generator
#rng = g.random("seed text")

# Random gauge field
if conf_name == "unit":
    U = g.qcd.gauge.unit(grid)
else:
    U = g.load(conf_name)

# Create 4D point source
src4D = g.mspincolor(U[0].grid)
g.create.point(src4D, [0, 0, 0, 0])

# Short-cuts
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner

# Even-odd-preconditioned CG solver
slv_5d = inv.preconditioned(pc.eo2_ne(), inv.cg(eps = resid, maxiter = max_it))

# Define expressions for fermions
for i in range(len(flav_names)):
    # Mobius domain-wall fermion
    fermion = g.qcd.fermion.mobius(U, mass=flav_masses[i], M5=M5, b=b5,
           c=c5, Ls=Ls,boundary_phases=[1,1,1,-1])
    D5_inv=slv_5d(fermion)
    # Short-cuts
    imp = fermion.ImportPhysicalFermionSource
    exp = fermion.ExportPhysicalFermionSolution
    # get 5D source from 4D source
    src5D =  g(imp * src4D)
    # Solve propagator on 12 spin-color components

    g.message("Calculate 5D propagator for flavor {f} with m = {m}".format(
              f=flav_names[i],m=flav_masses[i]))
    t0=g.time()
    prop5D = g( D5_inv * src5D )
    t1=g.time()
    g.message("")
    g.message("Time needed for the 12 inversions %g sec" % ((t1 - t0)))
    g.message("")

    g.message("Extract 4D from 5D propagator for flavor {f} with m = {m}".format(
              f=flav_names[i],m=flav_masses[i]))
    t0 = g.time()
    exec("prop4D_{f} = g( exp * prop5D )".format(f=flav_names[i]))
    t1 = g.time()
    g.message("")
    g.message("Time needed to extract the 4D from the 5D propagator: {} sec".format((t1-t0)))
    g.message("")
    exec("Jq5_{f}=get_Jq5(prop5D)".format(f=flav_names[i]))

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

#contraction
t0 = g.time()
for i in range(len(flav_names)):
    for j in range(i,len(flav_names)):
        header=""
        #g.message("Jq5:")
        #g.message("real\t\t\timag")
        #for i in range(len(Jq5)):
        #    g.message(f"{Jq5[i].real}\t{Jq5[i].imag}")
        print(flav_names[i],flav_names[j])
        if flav_names[i] == flav_names[j]:
            print("inside if")
            header='t\t\t\t\tJq5'
            exec("data=np.array([[t for t in range(len(Jq5_{f}))],\
                 [Jq5_{f}[t].real for t in range(len(Jq5_{f}))]])".format(f=flav_names[i]))

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


            exec("tCorr=g.slice(g.trace( P * G * prop4D_{f1} * G * g.gamma[5] *\
                  prop4D_{f2} * g.gamma[5] ), 3)".format(f1=flav_names[i],
                                                         f2=flav_names[j]))

            #g.message("real\t\t\timag")
            #for i in range(len(tCorr)):
            #    #g.message("{}\t{}".format(tCorr[i].real,tCorr[i].imag))
            #    g.message(f"{tCorr[i].real}\t{tCorr[i].imag}")

            header+='\t\t\t' + col
            data=np.append(data,[[tCorr[t].real for t in range(len(tCorr))]],axis = 0)

        np.savetxt("{out_folder}/{out_name_add}_pt_{f1}{f2}_ms{ms}ml{ml}Ls{Ls}b{b5}c{c5}M{M5}k{mom}\
_{conf_name}.txt".format(out_folder=out_folder,
                  out_name_add=out_name_add,
                  f1=flav_names[i],
                  f2=flav_names[j],
                  ml=str(flav_masses[0])[2:],
                  ms=str(flav_masses[1])[2:],
                  Ls=Ls,
                  b5=str(b5)[0]+str(b5)[2:],
                  c5=str(c5)[0]+str(c5)[2:],
                  M5=str(M5)[0]+str(M5)[2:],
                  mom="".join(str(elem) for elem in k),
                  conf_name=conf_name),
                  data.T,header=header,delimiter="\t",comments='#')
t1 = g.time()
g.message("")
g.message("All contractions were done in: {} sec".format((t1-t0)))
g.message("")
g.message("Total runtime: {} sec".format(g.time()))
g.message("")
