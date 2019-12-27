# -*- coding: utf-8 -*-
"""
Common Gamut Boundary Descriptor (GDB) Utilities
================================================

Defines various *Gamut Boundary Descriptor (GDB)* common utilities.

-   :func:`colour.interpolate_gamut_boundary_descriptor`

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
import scipy.interpolate

from colour.constants import DEFAULT_INT_DTYPE
from colour.models import Jab_to_JCh, JCh_to_Jab

from colour.utilities import (as_float_array, as_int_array,
                              is_trimesh_installed, orient, tsplit, tstack,
                              warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'close_gamut_boundary_descriptor', 'interpolate_gamut_boundary_descriptor',
    'tessellate_gamut_boundary_descriptor'
]


def close_gamut_boundary_descriptor(GDB_m, Jab, E=np.array([50, 0, 0])):

    GDB_m = np.copy(as_float_array(GDB_m))
    Jab = as_float_array(Jab)
    E = as_float_array(E)

    if not np.allclose(GDB_m[0, ...], GDB_m[0, ...][0]):
        JCh_l = np.mean(
            Jab_to_JCh(Jab[Jab[..., 0] == np.min(Jab[..., 0])] - E), axis=0)

        warning(
            'Inserting a singularity at the bottom of GBD: {0}'.format(JCh_l))
        GDB_m = np.insert(
            GDB_m, 0, np.tile(JCh_l, [1, GDB_m.shape[1], 1]), axis=0)

    if not np.allclose(GDB_m[-1, ...], GDB_m[-1, ...][0]):
        JCh_h = np.mean(
            Jab_to_JCh(Jab[Jab[..., 0] == np.max(Jab[..., 0])] - E), axis=0)

        warning('Inserting a singularity at the top of GBD: {0}'.format(JCh_h))

        GDB_m = np.insert(
            GDB_m,
            GDB_m.shape[0],
            np.tile(JCh_h, [1, GDB_m.shape[1], 1]),
            axis=0)

    return GDB_m


def interpolate_gamut_boundary_descriptor(GDB_m):
    GDB_m = as_float_array(GDB_m)

    GDB_m_i = np.copy(GDB_m)
    shape_r, shape_c = GDB_m.shape[0], GDB_m.shape[1]

    r_slice = np.s_[0:shape_r]
    c_slice = np.s_[0:shape_c]

    # If bounding columns have NaN, :math:`GDB_m` matrix is tiled
    # horizontally so that right values interpolate with left values and
    # vice-versa.
    if np.any(np.isnan(GDB_m[..., 0])) or np.any(np.isnan(GDB_m[..., -1])):
        warning(
            'Gamut boundary descriptor matrix bounding columns contains NaN '
            'and will be horizontally tiled!')
        c_slice = np.s_[shape_r:shape_r * 2]
        GDB_m_i = np.hstack([GDB_m] * 3)

    # If bounding rows have NaN, :math:`GDB_m` matrix is reflected vertically
    # so that top and bottom values are replicated via interpolation, i.e.
    # equivalent to nearest-neighbour interpolation.
    if np.any(np.isnan(GDB_m[0, ...])) or np.any(np.isnan(GDB_m[-1, ...])):
        warning('Gamut boundary descriptor matrix bounding rows contains NaN '
                'and will be vertically reflected!')
        r_slice = np.s_[shape_c:shape_c * 2]
        GDB_m_f = orient(GDB_m_i, 'Flop')
        GDB_m_i = np.vstack([GDB_m_f, GDB_m_i, GDB_m_f])

    mask = np.any(~np.isnan(GDB_m_i), axis=-1)
    for i in range(3):
        x = np.linspace(0, 1, GDB_m_i.shape[0])
        y = np.linspace(0, 1, GDB_m_i.shape[1])
        x_g, y_g = np.meshgrid(x, y, indexing='ij')
        values = GDB_m_i[mask]

        GDB_m_i[..., i] = scipy.interpolate.griddata(
            (x_g[mask], y_g[mask]),
            values[..., i], (x_g, y_g),
            method='linear')

    return GDB_m_i[r_slice, c_slice, :]


def tessellate_gamut_boundary_descriptor(GDB_m):
    if is_trimesh_installed():
        import trimesh

        vertices = JCh_to_Jab(GDB_m)

        shape_r, shape_c = vertices.shape[0], vertices.shape[1]

        faces = []
        for i in np.arange(shape_r - 1):
            for j in np.arange(shape_c - 1):
                a_i = [i, j]
                b_i = [i, j + 1]
                c_i = [i + 1, j]
                d_i = [i + 1, j + 1]

                # Avoiding overlapping triangles when tessellating the bottom.
                if not i == 0:
                    faces.append([a_i, b_i, c_i])

                # Avoiding overlapping triangles when tessellating the top.
                if not i == shape_r - 2:
                    faces.append([c_i, b_i, d_i])

        indices = np.ravel_multi_index(
            np.transpose(as_int_array(faces)), [shape_r, shape_c])

        GDB_t = trimesh.Trimesh(
            vertices=vertices.reshape([-1, 3]),
            faces=np.transpose(indices),
            validate=True)

        if not GDB_t.is_watertight:
            warning('Tessellated mesh has holes!')

        return GDB_t


if __name__ == '__main__':
    # 9c91accdd8ea9c39437694bb3265fa6b09fd87d2
    # rd
    # source Environments/plotly/bin/activate
    # export PYTHONPATH=$PYTHONPATH:/Users/kelsolaar/Documents/Development/colour-science/colour:/Users/kelsolaar/Documents/Development/colour-science/trimesh
    # python /Users/kelsolaar/Documents/Development/colour-science/colour/colour/gamut/boundary/common.py

    import trimesh
    import trimesh.smoothing

    import colour
    import colour.plotting
    from colour.gamut import gamut_boundary_descriptor_Morovic2000

    np.set_printoptions(
        formatter={'float': '{:0.2f}'.format}, linewidth=2048, suppress=True)

    m, n = 3, 8
    t = 3
    Hab = np.tile(np.arange(-180, 180, 45) / 360, t)
    C = np.hstack([
        np.ones(int(len(Hab) / t)) * 0.25,
        np.ones(int(len(Hab) / t)) * 0.5,
        np.ones(int(len(Hab) / t)) * 1.0,
    ])
    L = np.hstack([
        np.ones(int(len(Hab) / t)) * 1.0,
        np.ones(int(len(Hab) / t)) * 0.5,
        np.ones(int(len(Hab) / t)) * 0.25,
    ])

    LCHab = tstack([L, C, Hab])
    Jab = colour.convert(
        LCHab, 'CIE LCHab', 'CIE Lab', verbose={'describe': 'short'}) * 100

    np.random.seed(16)
    RGB = np.random.random([200, 200, 3])

    s = 128
    RGB = colour.plotting.geometry.cube(
        width_segments=s, height_segments=s, depth_segments=s)

    Jab_E = colour.convert(
        RGB, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100

    m, n = 16, 16
    Jab = Jab_E
    print(Jab_E)

    GDB_m = gamut_boundary_descriptor_Morovic2000(Jab, [50, 0, 0], m, n)
    print('-' * 79)
    print(GDB_m[..., 0])

    # a = np.full([5, 9], np.nan)
    #
    # a[0, 0] = 1
    # a[0, -1] = 5
    # a[-1, 0] = 10
    # a[-1, -1] = 20
    #
    # a = np.full([5, 9], np.nan)
    #
    # a[0, 4] = 1
    # a[2, 4] = 3

    GDB_c = close_gamut_boundary_descriptor(GDB_m, Jab, [50, 0, 0])

    GDB_m_i = interpolate_gamut_boundary_descriptor(GDB_c)
    print('^' * 79)
    print(GDB_m_i[..., 0])
    # print(GDB_m[..., 1])
    # print(GDB_m_i[..., 1])
    # print(GDB_m[..., 2])
    # print(GDB_m_i[..., 2])

    import matplotlib.pyplot as plt

    figure, all_axes = plt.subplots(
        GDB_m_i.shape[1] // 3,
        3,
        sharex='col',
        sharey='row',
        gridspec_kw={
            'hspace': 0,
            'wspace': 0
        })

    all_axes = np.ravel(all_axes)

    cycle = colour.plotting.colour_cycle(colour_cycle_count=GDB_m_i.shape[1])
    for i in range(GDB_m_i.shape[1]):
        all_axes[i].plot(
            GDB_m_i[..., i, 1],
            orient(GDB_m_i[..., i, 0], 'Flop'),
            label='{0:d} deg'.format(int(i / GDB_m_i.shape[1] * 360)),
            color=next(cycle))

        all_axes[i].legend(loc='lower right')

    plt.show()

    GDB_t = tessellate_gamut_boundary_descriptor(GDB_m_i)

    # trimesh.smoothing.filter_laplacian(GDB_t, iterations=25)
    # trimesh.repair.broken_faces(GDB_t, color=(255, 0, 0, 255))

    GDB_t.export('/Users/kelsolaar/Downloads/mesh.dae', 'dae')

    GDB_t.show()
