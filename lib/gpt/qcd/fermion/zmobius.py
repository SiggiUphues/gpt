#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt, copy
from gpt.qcd.fermion.operator import differentiable_fine_operator
import numpy as np


@gpt.params_convention(
    omega=None, mass=None, b=None, c=None, M5=None, boundary_phases=None
)
def zmobius(U, params):
    params = copy.deepcopy(params)  # save current parameters
    params["Ls"] = len(params["omega"])

    op = differentiable_fine_operator(
        "zmobius", U, params, otype=gpt.ot_vector_spin_color(4, 3)
    )

    def zmobius_kappa_create():
        b = params["b"]
        c = params["c"]
        M5 = params["M5"]
        bs = [ 1./2. * (  1./omega_s*(b+c) + (b-c) ) for omega_s in params["omega"] ]

        kappa = np.array([ 1.0 / ( 2.0 * (bsi *(4. - M5) + 1.0) ) for bsi in bs ], np.complex128)
        adj_kappa = np.conj(kappa)
        inv_kappa = 1.0 / kappa
        adj_inv_kappa = np.conj(inv_kappa)

        def _mat(dst, src):
            gpt.scale_per_coordinate(dst, src, kappa, 0)

        def _inv_mat(dst, src):
            gpt.scale_per_coordinate(dst, src, inv_kappa, 0)

        def _adj_mat(dst, src):
            gpt.scale_per_coordinate(dst, src, adj_kappa, 0)

        def _adj_inv_mat(dst, src):
            gpt.scale_per_coordinate(dst, src, adj_inv_kappa, 0)

        return gpt.matrix_operator(
            mat = _mat,
            inv_mat = _inv_mat,
            adj_mat = _adj_mat,
            adj_inv_mat = _adj_inv_mat
        )
    
    op.kappa = zmobius_kappa_create
    return op
