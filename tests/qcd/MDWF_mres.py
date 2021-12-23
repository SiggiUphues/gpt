import gpt as g
import numpy as np
import sys
import os
import itertools

def get_vec(tag,type, default, ndim):
    res = g.default.get_all(tag, None)
    if res[0] is None:
        return default
    if type == "s":
        for x in res:
            v = [y for y in x.split(".")]
            if len(v) == ndim:
                return v
            else:
                raise ValueError("len({tag}) != {ndim}".format(tag=tag,ndim=ndim))
    if type == "f":
        for x in res:
            v = [float(y) for y in x.split(",")]
            if len(v) == ndim:
                return v
            else:
                raise ValueError("len({tag}) != {ndim}".format(tag=tag,ndim=ndim))
    if type == "i":
        for x in res:
            v = [int(y) for y in x.split(".")]
            if len(v) == ndim:
                return v
            else:
                raise ValueError("len({tag}) != {ndim}".format(tag=tag,ndim=ndim))
    return default

def get_bool(tag,default = False):
    assert type(default) == bool, "The default value has to be a boolean"
    res = default
    for i, x in enumerate(sys.argv):
        if x == tag:
            res = not default
    return res

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
b5=g.default.get_float("-b5",1.50)
c5=g.default.get_float("-c5",0.50)
nflavs=g.default.get_int("-nflavs",2)
flav_names=get_vec("-flav-names","s",["l","s"],nflavs)
flav_masses=get_vec("-flav-masses","f",[0.01,0.1],nflavs)
assert len(flav_names) == len(flav_masses), "-flav_names and -flav_masses must have the same length"
assert len(flav_names) == nflavs, "-nflavs specifies the number of flavors and is not equal to the number of "

resid=g.default.get_float("-resid",1e-8)
max_it=g.default.get_int("-max-it",1000)
conf_name=g.default.get_single("-conf-name","unit")
out_name_add=g.default.get_single("-out-name-add","corrgpt")
out_folder=g.default.get_single("-out-folder",".")
tdir=get_bool("-tdir")
sdir=get_bool("-sdir")
# upper and lower bound for the momenta of the temporal correlator
# If lower bound is [0,0,0] and the upper bound [2,2,2] the program will
# calculate all temporal correlators with momenta:
# [0,0,0],[0,0,1],[0,0,2],[0,1,0], ... , [2,2,1], [2,2,2]
kt_upbound=np.array(get_vec("-kt_upbound","i",[0,0,0],3))
kt_upbound=kt_upbound.astype(int)
kt_lowbound=np.array(get_vec("-kt_lowbound","i",[0,0,0],3))
kt_lowbound=kt_lowbound.astype(int)
kt_x=np.arange(kt_lowbound[0],kt_upbound[0]+1)
kt_y=np.arange(kt_lowbound[1],kt_upbound[1]+1)
kt_z=np.arange(kt_lowbound[2],kt_upbound[2]+1)
kt_array=list(itertools.product(kt_x,kt_y,kt_z))

# upper and lower bound for the momenta of the spatial correlator
# If lower bound is [0,0,0] and the upper bound [2,2,2] the program will
# calculate all spatial correlators with momenta:
# [0,0,0],[0,0,1],[0,0,2],[0,1,0], ... , [2,2,1], [2,2,2]
ks_upbound=np.array(get_vec("-ks_upbound","i",[0,0,0],3))
ks_upbound=ks_upbound.astype(int)
ks_lowbound=np.array(get_vec("-ks_lowbound","i",[0,0,0],3))
ks_lowbound=ks_lowbound.astype(int)
#g.message("ks_x=np.arange({},{}+1)".format(ks_lowbound[0],ks_upbound[0]))
ks_x=np.arange(ks_lowbound[0],ks_upbound[0]+1)
ks_y=np.arange(ks_lowbound[1],ks_upbound[1]+1)
ks_t=np.arange(ks_lowbound[2],ks_upbound[2]+1)
#g.message(ks_x)
#g.message(ks_y)
#g.message(ks_t)
ks_array=list(itertools.product(ks_x,ks_y,ks_t))

#g.message(ks_array)

# additional correlator channel
Gopt=g.default.get_single("-G",None)
if Gopt:
    Gammas=np.append(Gammas,Gopt)

# Double-precision 8^4 grid
#Dims=[8,8,8,16]
grid = g.grid(Dims, g.double)



# Parallel random number generator
#rng = g.random("seed text")

# Random gauge field
if conf_name == "unit":
    U = g.qcd.gauge.unit(grid)
    conf_name = "unit" + "{Ns}{Nt}".format(Ns = Dims[0],Nt = Dims[3])
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

    g.message("Calculate 5D propagator for flavor {f} with m{f} = {m}".format(
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
    #g.mem_report()
    exec("prop4D_{f} = g( exp * prop5D )".format(f=flav_names[i]))
    t1 = g.time()
    #g.mem_report()
    g.message("")
    g.message("Time needed to extract the 4D from the 5D propagator: {} sec".format((t1-t0)))
    g.message("")
    exec("Jq5_{f}=get_Jq5(prop5D)".format(f=flav_names[i]))
    # delete 5D prop since only 4D prop is needed for the meson correlators
    del prop5D
    #g.mem_report()

#contraction
t0 = g.time()
for i in range(len(flav_names)):
    for j in range(i,len(flav_names)):
        theader=""
        sheader=""
        tdata=np.array([])
        sdata=np.array([])
        #g.message("Jq5:")
        #g.message("real\t\t\timag")
        #for i in range(len(Jq5)):
        #    g.message(f"{Jq5[i].real}\t{Jq5[i].imag}")
        #print(flav_names[i],flav_names[j])
        if(tdir):
            theader='t\t'
            tdata=np.array([[int(t) for t in range(Dims[3])]])
        if(sdir):
            sheader='z\t'
            sdata=np.array([[int(z) for z in range(Dims[2])]])
        if flav_names[i] == flav_names[j]:
            #print("inside if")
            if(tdir):
                theader+='\t\t\tJq5'
                exec("tdata=np.append(tdata,\
                 [[Jq5_{f}[t].real for t in range(len(Jq5_{f}))]], axis = 0)".format(f=flav_names[i]))

            out_namet="{out_name_add}_pt_{f}{f}_t_m{f}{m}".format(out_name_add=out_name_add,f=flav_names[i],m=str(flav_masses[i])[2:])
            out_names="{out_name_add}_pt_{f}{f}_s_m{f}{m}".format(out_name_add=out_name_add,f=flav_names[i],m=str(flav_masses[i])[2:])
        else:
            out_namet="{out_name_add}_pt_{f1}{f2}_t_m{f1}{m1}m{f2}{m2}".format(
                       out_name_add=out_name_add,
                       f1=flav_names[i],
                       f2=flav_names[j],
                       m1=str(flav_masses[i])[2:],
                       m2=str(flav_masses[j])[2:])
            out_names="{out_name_add}_pt_{f1}{f2}_s_m{f1}{m1}m{f2}{m2}".format(
                       out_name_add=out_name_add,
                       f1=flav_names[i],
                       f2=flav_names[j],
                       m1=str(flav_masses[i])[2:],
                       m2=str(flav_masses[j])[2:])

        if(tdir):
            g.message("Do contraction in temporal direction for {out_name}".format(out_name=out_namet))
            for kt in kt_array:
                g.message("Calculate temporal correlator for momentun kt = {}".format(kt))
                # cast tuple to np.array for operations
                kt = np.array(kt)
                if np.sum(kt > 0) != 0:
                    momt_str="kt" + "".join(str(elem) for elem in kt)
                else:
                    momt_str=""

                pt= 2.0 * np.pi * np.hstack((kt/(Dims[0]),0))
                # exp(ix*pt)
                Pt=g.exp_ixp(pt)
                for comb in Gammas:
                    GMats=comb.split('.')
                    #g.message(GMats)
                    col="G" + "G".join(GMats)
                    #g.message(col + " correlator")
                    if GMats[0] == "5":
                        G=g.gamma[5]
                    else:
                        G=g.gamma[GMats[0]]

                    for ind in GMats[1:]:
                        if ind == "5":
                            G = G * g.gamma[5]
                        else:
                            G= G * g.gamma[ind]


                    exec("tCorr=g.slice(g.trace( Pt * G * prop4D_{f1} * G * g.gamma[5] *\
                          prop4D_{f2} * g.gamma[5] ), 3)".format(f1=flav_names[i],
                                                                 f2=flav_names[j]))
                    theader+='\t\t\t' + col
                    tdata=np.append(tdata,[[tCorr[t].real for t in range(len(tCorr))]],axis = 0)

                    #g.message("real\t\t\timag")
                    #for i in range(len(tCorr)):
                    #    #g.message("{}\t{}".format(tCorr[i].real,tCorr[i].imag))
                    #    g.message(f"{tCorr[i].real}\t{tCorr[i].imag}")

                os.makedirs(out_folder + "/tmesons",exist_ok=True)
                np.savetxt("{out_folder}/tmesons/{out_name}Ls{Ls}b{b5}c{c5}M{M5}{mom}\
_{conf_name}.txt".format(out_folder=out_folder,
                         out_name=out_namet,
                         Ls=Ls,
                         b5=str(b5)[0]+str(b5)[2:],
                         c5=str(c5)[0]+str(c5)[2:],
                         M5=str(M5)[0]+str(M5)[2:],
                         mom=momt_str,
                         conf_name=conf_name),
                         tdata.T,header=theader,delimiter="\t",comments='#')
        if(sdir):
            g.message("Do contraction in spatial direction for {out_name}".format(out_name=out_names))
            for ks in ks_array:
                # cast tuple to np.array for operations
                ks = np.array(ks)
                if np.sum( ks > 0) != 0:
                    moms_str="ks" + "".join(str(elem) for elem in ks)
                else:
                    moms_str=""

                g.message("Calculate spatial correlator for momentun ks = {}".format(ks))
                #g.message(moms_str)
                ps= 2.0 * np.pi * np.hstack((ks[0:2]/(Dims[0]),0,ks[2]/(Dims[3])))
                # exp(ix_funny*ps_funny) x_funny * ps_funny = x*px + y*py + t*pt
                Ps=g.exp_ixp(ps)
                for comb in Gammas:
                    GMats=comb.split('.')
                    #g.message(GMats)
                    col="G" + "G".join(GMats)
                    #g.message(col + " correlator")
                    if GMats[0] == "5":
                        G=g.gamma[5]
                    else:
                        G=g.gamma[GMats[0]]

                    for ind in GMats[1:]:
                        if ind == "5":
                            G = G * g.gamma[5]
                        else:
                            G= G * g.gamma[ind]

                    exec("sCorr=g.slice(g.trace( Ps * G * prop4D_{f1} * G * g.gamma[5] *\
                        prop4D_{f2} * g.gamma[5] ), 2)".format(f1=flav_names[i],
                                                               f2=flav_names[j]))
                    sheader+='\t\t\t' + col
                    sdata=np.append(sdata,[[sCorr[s].real for s in range(len(sCorr))]],axis = 0)

                os.makedirs(out_folder + "/smesons",exist_ok=True)
                np.savetxt("{out_folder}/smesons/{out_name}Ls{Ls}b{b5}c{c5}M{M5}{mom}\
_{conf_name}.txt".format(out_folder=out_folder,
                         out_name=out_names,
                         Ls=Ls,
                         b5=str(b5)[0]+str(b5)[2:],
                         c5=str(c5)[0]+str(c5)[2:],
                         M5=str(M5)[0]+str(M5)[2:],
                         mom=moms_str,
                         conf_name=conf_name),
                         sdata.T,header=sheader,delimiter="\t",comments='#')


t1 = g.time()
g.message("")
g.message("All contractions were done in: {} sec".format((t1-t0)))
g.message("")
g.message("Total runtime: {} sec".format(g.time()))
g.message("")
