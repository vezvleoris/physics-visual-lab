"""
Geometry for the FCC tetrahedral-site visualization.

This module contains only geometric calculations.
All positions are represented using fractional coordinates
of the conventional cubic unit cell.

Main idea
---------
1. Construct FCC lattice points.
2. Select one corner and its three neighboring face centers.
3. Show that these four points form a regular tetrahedron.
4. Compute the center of the tetrahedron.
5. The center is (1/4, 1/4, 1/4), an FCC tetrahedral site.
"""

from itertools import product

import numpy as np


def get_fcc_lattice_points():
    """
    Return FCC lattice points inside one conventional unit cell.

    Returns
    -------
    dict
        {
            "corners": ndarray, shape (8, 3),
            "face_centers": ndarray, shape (6, 3)
        }

    Notes
    -----
    Coordinates are fractional coordinates.

    For example,

        (1, 0, 0)

    means the Cartesian position

        (a, 0, 0)

    where a is the lattice constant.
    """

    # Eight corners of the conventional cubic cell
    corners = np.array(
        list(product([0.0, 1.0], repeat=3)),
        dtype=float,
    )

    # Six face centers
    face_centers = np.array(
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=float,
    )

    return {
        "corners": corners,
        "face_centers": face_centers,
    }


def get_tetrahedron_vertices():
    """
    Return four FCC lattice points forming a regular tetrahedron.

    We choose the corner at (0, 0, 0) and the three neighboring
    face-centered lattice points:

        A = (0,   0,   0)
        B = (0,   1/2, 1/2)
        C = (1/2, 0,   1/2)
        D = (1/2, 1/2, 0)

    Every pair of these points is separated by the same distance,
    so they form a regular tetrahedron.

    Returns
    -------
    ndarray, shape (4, 3)
        Fractional coordinates of the tetrahedron vertices.
    """

    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )


def get_tetrahedral_site(vertices):
    """
    Calculate the center of a tetrahedron.

    Parameters
    ----------
    vertices : array-like, shape (4, 3)
        Fractional coordinates of the four tetrahedron vertices.

    Returns
    -------
    ndarray, shape (3,)
        Fractional coordinate of the tetrahedron center.

    For the FCC tetrahedron used here,

        center = (A + B + C + D) / 4
               = (1/4, 1/4, 1/4)

    This position is an FCC tetrahedral site.
    """

    vertices = np.asarray(vertices, dtype=float)

    if vertices.shape != (4, 3):
        raise ValueError("vertices must have shape (4, 3)")

    return np.mean(vertices, axis=0)


def get_pairwise_distances(vertices):
    """
    Calculate all six pairwise distances between four vertices.

    This is useful for verifying that the selected FCC points
    actually form a regular tetrahedron.

    Returns
    -------
    dict
        Dictionary such as

        {
            "AB": distance,
            "AC": distance,
            ...
        }

    Distances are expressed in units of the lattice constant a.
    """

    vertices = np.asarray(vertices, dtype=float)

    labels = ["A", "B", "C", "D"]
    distances = {}

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            key = labels[i] + labels[j]
            distances[key] = np.linalg.norm(vertices[j] - vertices[i])

    return distances


def get_basis_position(t):
    """
    Return the position (t, t, t) along the [111] direction.

    Parameters
    ----------
    t : float
        Fractional displacement along x, y, and z.

    Returns
    -------
    ndarray, shape (3,)

    Examples
    --------
    t = 0.25 -> (1/4, 1/4, 1/4)

    At t = 1/4, the position coincides with the FCC
    tetrahedral site used as the second basis position
    of the diamond cubic structure.
    """

    return np.array([t, t, t], dtype=float)