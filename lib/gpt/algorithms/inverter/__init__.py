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
from gpt.algorithms.inverter.cg import cg
from gpt.algorithms.inverter.bicgstab import bicgstab
from gpt.algorithms.inverter.fgcr import fgcr
from gpt.algorithms.inverter.fgmres import fgmres
from gpt.algorithms.inverter.mr import mr
from gpt.algorithms.inverter.mg import mg_setup, mg_prec
from gpt.algorithms.inverter.defect_correcting import defect_correcting
from gpt.algorithms.inverter.mixed_precision import mixed_precision
from gpt.algorithms.inverter.split import split
from gpt.algorithms.inverter.preconditioned import preconditioned
