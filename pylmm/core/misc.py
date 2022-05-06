
# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2015  Nicholas A. Furlotte (nick.furlotte@gmail.com)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#!/usr/bin/python

import sys
import numpy as np
from pylmm.core import lmm


def fitTwo(y,K1,K2,X0=None,wgrids=100):
	"""
		Simple function to fit a model with two variance components.
		It works by running the standard pylmm algorithm in a loop
		where at each iteration of the loop a new kinship is generated
		as a linear combination of the original two.
	"""
	# Create a uniform grid
	W = np.array(range(wgrids)) / float(wgrids)
	Res = []
	LLs = []

	for w in W:
		# heritability will be estimated for linear combo of kinships
		K = w*K1 + (1.0 - w)*K2
		sys.stderr.write("Fitting weight %0.2f\n" % (w))
		L = lmm.LMM(y,K,X0=X0)
		R = L.fit()
		Res.append(R)
		LLs.append(R[-1])

		del K

	L = np.array(LLs)
	i = np.where(L == L.max())[0]
	if len(i) > 1:
		sys.stderr.write("WARNING: Found multiple maxes using first one\n")

	i = i[0]
	hmax,beta,sigma,LL = Res[i]
	w = W[i]

	h1 = w * hmax 
	h2 = (1.0 - w) * hmax 
	e = (1.0 - hmax)

	return (h1,h2,e,beta,sigma,LL)


## Below functions are taken to make dot product efficient
### https://stackoverflow.com/questions/20983882/efficient-dot-products-of-large-memory-mapped-arrays

def _block_slices(dim_size, block_size):
    """Generator that yields slice objects for indexing into 
    sequential blocks of an array along a particular axis
    """
    count = 0
    while True:
        yield slice(count, count + block_size, 1)
        count += block_size
        if count > dim_size:
            raise StopIteration

def blockwise_dot(A, B, max_elements=int(2**27), out=None):
    """
    Computes the dot product of two matrices in a block-wise fashion. 
    Only blocks of `A` with a maximum size of `max_elements` will be 
    processed simultaneously.
    """

    m,  n = A.shape
    n1, o = B.shape

    if n1 != n:
        raise ValueError('matrices are not aligned')

    if A.flags.f_contiguous:
        # prioritize processing as many columns of A as possible
        max_cols = max(1, max_elements / m)
        max_rows =  max_elements / max_cols

    else:
        # prioritize processing as many rows of A as possible
        max_rows = max(1, max_elements / n)
        max_cols =  max_elements / max_rows

    if out is None:
        out = np.empty((m, o), dtype=np.result_type(A, B))
    elif out.shape != (m, o):
        raise ValueError('output array has incorrect dimensions')

    for mm in _block_slices(m, max_rows):
        out[mm, :] = 0
        for nn in _block_slices(n, max_cols):
            A_block = A[mm, nn].copy()  # copy to force a read
            out[mm, :] += np.dot(A_block, B[nn, :])
            del A_block

    return out