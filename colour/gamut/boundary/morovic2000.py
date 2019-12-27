# -*- coding: utf-8 -*-
"""
Gamut Boundary Descriptor (GDB) - Morovic and Luo (2000)
========================================================

Defines the * Morovic and Luo (2000)* *Gamut Boundary Descriptor (GDB)*
computation objects:

-   :func:`colour.gamut.gamut_boundary_descriptor_Morovic2000`

See Also
--------
`Gamut Boundary Descriptor Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/gamut/boundary.ipynb>`_

References
----------
-   :cite:`` :
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.gamut.boundary import close_gamut_boundary_descriptor
from colour.models import Jab_to_JCh, JCh_to_Jab
from colour.utilities import (as_int, as_int_array, as_float_array,
                              linear_conversion, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['gamut_boundary_descriptor_Morovic2000']


def gamut_boundary_descriptor_Morovic2000(Jab,
                                          E=np.array([50, 0, 0]),
                                          m=16,
                                          n=16):
    Jab = as_float_array(Jab)
    E = as_float_array(E)

    JCh = Jab_to_JCh(np.reshape(Jab, [-1, 3]) - E)

    phi, r, alpha = tsplit(JCh)
    alpha = np.radians(alpha)

    GDB_m = np.full([m, n, 3], np.nan)

    # Lightness :math:`J` is in range [-E_{J}, E_{J}], converted to range
    # [0, m], :math:`\\phi` indices are in range [0, m - 1].
    phi_i = linear_conversion(phi, (-E[0], E[0]), (0, m))
    phi_i = as_int_array(np.clip(np.floor(phi_i), 0, m - 1))

    # Polar coordinates are in range [0, 2 * pi], converted to range [0, n],
    # :math:`\\alpha` indices are in range [0, n - 1].
    alpha_i = linear_conversion(alpha, (0, 2 * np.pi), (0, n))
    alpha_i = as_int_array(np.clip(np.floor(alpha_i), 0, n - 1))

    for i in np.arange(m):
        for j in np.arange(n):
            i_j = np.intersect1d(
                np.argwhere(phi_i == i), np.argwhere(alpha_i == j))

            if i_j.size == 0:
                continue

            GDB_m[i, j] = JCh[i_j[np.argmax(r[i_j])]]

    # Naive non-vectorised implementation kept for reference.
    # :math:`r_m` is used to keep track of the maximum :math:`r` value.
    # r_m = np.full([m, n, 1], np.nan)
    # for i, Jab_i in enumerate(Jab):
    #     p_i, a_i = phi_i[i], alpha_i[i]
    #     r_i_j = r_m[p_i, a_i]
    #
    #     if r[i] > r_i_j or np.isnan(r_i_j):
    #         GDB_m[p_i, a_i] = Jab_i
    #         r_m[p_i, a_i] = r[i]


    return GDB_m
