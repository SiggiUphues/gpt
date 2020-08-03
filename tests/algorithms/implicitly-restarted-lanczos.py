#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import sys

# load configuration
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.single), g.random("test"))

# wilson, eo prec
parity = g.odd
w = g.qcd.fermion.preconditioner.eo1_ne(parity=parity)(
    g.qcd.fermion.wilson_clover(
        U,
        {
            "kappa": 0.137,
            "csw_r": 0,
            "csw_t": 0,
            "xi_0": 1,
            "nu": 1,
            "isAnisotropic": False,
            "boundary_phases": [1.0, 1.0, 1.0, 1.0],
        },
    )
)


# cheby
c = g.algorithms.polynomial.chebyshev({"low": 0.5, "high": 2.0, "order": 10})

# implicitly restarted lanczos
irl = g.algorithms.eigen.irl(
    {
        "Nk": 60,
        "Nstop": 60,
        "Nm": 80,
        "resid": 1e-8,
        "betastp": 0.0,
        "maxiter": 20,
        "Nminres": 7,
        #    "maxapply" : 100
    }
)

# start vector
start = g.vspincolor(w.F_grid_eo)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start.checkerboard(parity)

# generate eigenvectors
evec, ev = irl(c(w.Mpc), start)  # , g.checkpointer("checkpoint")

# memory info
g.mem_report()

# print eigenvalues of NDagN as well
evals = g.algorithms.eigen.evals(w.Mpc, evec, check_eps2=1e-11, real=True)

# deflated solver
cg = g.algorithms.inverter.cg({"eps": 1e-6, "maxiter": 1000})
defl = g.algorithms.eigen.deflate(cg, evec, evals)

sol_cg = g.eval(cg(w.Mpc) * start)
eps2 = g.norm2(w.Mpc * sol_cg - start) / g.norm2(start)
niter_cg = len(cg.history)
g.message("Test resid/iter cg: ", eps2, niter_cg)
assert eps2 < 1e-8

sol_defl = g.eval(defl(w.Mpc) * start)
eps2 = g.norm2(w.Mpc * sol_defl - start) / g.norm2(start)
niter_defl = len(cg.history)
g.message("Test resid/iter deflated cg: ", eps2, niter_defl)
assert eps2 < 1e-8

assert niter_defl < niter_cg
