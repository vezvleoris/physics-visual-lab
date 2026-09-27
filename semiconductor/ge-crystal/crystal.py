"""
Diamond cubic crystal geometry.

diamond cubic 구조 생성 + periodic image + nearest neighbor 탐색

Conceptual hierarchy
--------------------
Bravais lattice:
    FCC lattice points R

Basis:
    Ge atoms at
        b1 = (0, 0, 0)
        b2 = (1/4, 1/4, 1/4)

Crystal:
    r = R + b

All crystal coordinates are fractional unless stated otherwise.
"""

from itertools import product
import numpy as np


# ============================================================
# 1. FCC Bravais lattice
# ============================================================

def get_fcc_lattice_points():
    """
    Return the four FCC Bravais lattice points represented
    inside one conventional cubic cell.

    These are lattice points, not Ge atoms.
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


# ============================================================
# 2. Basis
# ============================================================

def get_basis(basis_fractional):
    """
    Return the two-atom basis of diamond cubic.

    For Ge:
        b1 = (0, 0, 0)
        b2 = (1/4, 1/4, 1/4)

    Both basis sites are occupied by Ge atoms.
    """
    basis = np.asarray(
        basis_fractional,
        dtype=float,
    )

    if basis.shape != (2, 3):
        raise ValueError(
            "Diamond cubic requires a two-atom basis."
        )

    return basis


# ============================================================
# 3. Crystal = lattice + basis
# ============================================================

def build_crystal(lattice_points, basis):
    """
    Construct atomic positions using

        r = R + b

    R : Bravais lattice point
    b : basis position
    """
    lattice_points = np.asarray(
        lattice_points,
        dtype=float,
    )

    basis = np.asarray(
        basis,
        dtype=float,
    )

    atoms = []

    for R in lattice_points:
        for b in basis:
            atoms.append((R + b) % 1.0)

    return np.asarray(atoms, dtype=float)


def build_diamond_crystal(basis_fractional):
    """
    Construct one conventional diamond cubic cell.

    Returns lattice, basis and resulting atomic positions
    separately so that they can be visualized independently.
    """
    lattice_points = get_fcc_lattice_points()
    basis = get_basis(basis_fractional)

    atoms = build_crystal(
        lattice_points,
        basis,
    )

    return {
        "lattice_points": lattice_points,
        "basis": basis,
        "atoms": atoms,
    }


def get_diamond_conventional_display_atoms():
    """
    Atomic positions used to DRAW a complete conventional
    diamond-cubic unit cell.

    These are visual atoms on and inside the cell, not the
    coordinates used for periodic nearest-neighbor search.

    Effective atom count:
        corners : 8 * 1/8 = 1
        faces   : 6 * 1/2 = 3
        internal: 4 * 1   = 4

        total = 8 atoms / conventional cell
    """

    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )

    faces = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
            [0.5, 0, 0.5],
            [0.5, 1, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
        ],
        dtype=float,
    )

    internal = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ],
        dtype=float,
    )

    return {
        "corners": corners,
        "faces": faces,
        "internal": internal,
    }


def get_fcc_conventional_display_points():
    """
    FCC lattice points drawn on and inside one conventional
    cell: 8 corners + 6 face centers.
    """
    points = []

    for R in get_fcc_lattice_points():
        for T in product((0, 1), repeat=3):
            points.append(R + np.asarray(T, dtype=float))

    points = np.asarray(points)

    inside = np.all(
        points <= 1.0 + 1e-9,
        axis=1,
    )

    return np.unique(
        np.round(points[inside], 6),
        axis=0,
    )


# ============================================================
# 4. Supercell (repeated conventional cells)
# ============================================================

def build_display_supercell(
    cell_positions,
    nx=1,
    ny=1,
    nz=1,
):
    """
    Repeat conventional-cell display positions nx × ny × nz times.

    Parameters
    ----------
    cell_positions : array or dict of arrays
        Fractional positions drawn for one cell
        (e.g. the dict from get_diamond_conventional_display_atoms).
    nx, ny, nz : int
        Number of conventional cells along x, y, z.

    Returns
    -------
    ndarray
        Unique fractional positions (in units of a) inside
        [0, nx] × [0, ny] × [0, nz]. Atoms on a shared cell
        boundary appear once.
    """
    if isinstance(cell_positions, dict):
        cell_positions = np.vstack(
            list(cell_positions.values())
        )

    cell_positions = np.asarray(
        cell_positions,
        dtype=float,
    )

    repeated = [
        cell_positions + np.asarray(T, dtype=float)
        for T in product(
            range(nx),
            range(ny),
            range(nz),
        )
    ]

    return np.unique(
        np.round(np.vstack(repeated), 6),
        axis=0,
    )


def clip_to_box(
    positions,
    lower,
    upper,
    tol=1e-6,
):
    """Keep positions inside [lower, upper] (fractional, per axis)."""

    positions = np.asarray(
        positions,
        dtype=float,
    ).reshape(-1, 3)

    inside = np.all(
        (positions >= np.asarray(lower) - tol)
        & (positions <= np.asarray(upper) + tol),
        axis=1,
    )

    return positions[inside]


_SITE_KINDS = {
    0: "internal",
    1: "face",
    2: "edge",
    3: "corner",
}


def classify_cell_site(
    position,
    cell=(0, 0, 0),
    tol=1e-6,
):
    """
    Classify a site relative to ONE conventional cell.

    corner / face / internal only make sense with respect to a
    chosen cell: an atom on a shared face is a "face" atom for
    both neighboring cells.

    Returns None if the site lies outside the cell, otherwise
        kind         : "corner" | "edge" | "face" | "internal"
        shared_by    : number of cells sharing the site
        contribution : fraction belonging to this cell
    """
    local = (
        np.asarray(position, dtype=float)
        - np.asarray(cell, dtype=float)
    )

    if np.any(local < -tol) or np.any(local > 1.0 + tol):
        return None

    on_boundary = int(
        np.sum(
            (np.abs(local) < tol)
            | (np.abs(local - 1.0) < tol)
        )
    )

    shared_by = 2 ** on_boundary

    return {
        "kind": _SITE_KINDS[on_boundary],
        "shared_by": shared_by,
        "contribution": 1.0 / shared_by,
    }


def get_basis_pairs(
    lattice_points,
    basis,
    nx=1,
    ny=1,
    nz=1,
):
    """
    Pairs (R + b1, R + b2) attached to the same FCC lattice
    point R, keeping only pairs fully inside the supercell.

    These are basis pairs, not a list of chemical bonds.

    Returns
    -------
    ndarray, shape (n_pairs, 2, 3), fractional
    """
    lattice_points = np.asarray(
        lattice_points,
        dtype=float,
    )

    basis = np.asarray(
        basis,
        dtype=float,
    )

    upper = np.array(
        [nx, ny, nz],
        dtype=float,
    ) + 1e-6

    pairs = []

    for R in lattice_points:
        first = R + basis[0]
        second = R + basis[1]

        if np.all(first <= upper) and np.all(second <= upper):
            pairs.append([first, second])

    return np.asarray(pairs, dtype=float).reshape(-1, 2, 3)


def fold_to_cell_index(
    position,
    fractional_atoms,
):
    """
    Index of the unit-cell atom equivalent to a supercell
    position under lattice translations.
    """
    atoms = np.asarray(
        fractional_atoms,
        dtype=float,
    )

    delta = (
        atoms
        - np.mod(np.asarray(position, dtype=float), 1.0)
    )

    delta -= np.round(delta)

    return int(
        np.argmin(np.linalg.norm(delta, axis=1))
    )


# ============================================================
# 5. Coordinate conversion
# ============================================================

def fractional_to_cartesian(
    positions,
    a_angstrom,
):
    """Convert fractional coordinates to Å."""
    return (
        np.asarray(positions, dtype=float)
        * a_angstrom
    )


# ============================================================
# 6. Periodic crystal
# ============================================================

def generate_periodic_images(
    fractional_atoms,
    shell=1,
):
    """
    Generate neighboring unit-cell copies.

    Required because nearest neighbors of a boundary atom
    may belong to an adjacent unit cell.
    """
    atoms = np.asarray(
        fractional_atoms,
        dtype=float,
    )

    positions = []
    source_indices = []
    translations = []

    for translation in product(
        range(-shell, shell + 1),
        repeat=3,
    ):
        T = np.asarray(
            translation,
            dtype=float,
        )

        for index, atom in enumerate(atoms):
            positions.append(atom + T)
            source_indices.append(index)
            translations.append(T.copy())

    return (
        np.asarray(positions),
        np.asarray(source_indices),
        np.asarray(translations),
    )


# ============================================================
# 7. Nearest neighbors
# ============================================================

def find_nearest_neighbors(
    selected_index,
    fractional_atoms,
    a_angstrom,
    number_of_neighbors=4,
):
    """
    Find nearest neighbors using periodic boundary conditions.

    Diamond cubic:
        coordination number = 4
        d_NN = sqrt(3) a / 4
    """
    atoms = np.asarray(
        fractional_atoms,
        dtype=float,
    )

    center_fractional = atoms[selected_index]

    (
        periodic_atoms,
        source_indices,
        translations,
    ) = generate_periodic_images(
        atoms,
        shell=1,
    )

    displacement_fractional = (
        periodic_atoms
        - center_fractional
    )

    displacement_cartesian = (
        displacement_fractional
        * a_angstrom
    )

    distances = np.linalg.norm(
        displacement_cartesian,
        axis=1,
    )

    # Exclude the selected atom itself.
    valid = distances > 1e-10

    periodic_atoms = periodic_atoms[valid]
    source_indices = source_indices[valid]
    translations = translations[valid]

    displacement_fractional = (
        displacement_fractional[valid]
    )

    displacement_cartesian = (
        displacement_cartesian[valid]
    )

    distances = distances[valid]

    order = np.argsort(distances)[
        :number_of_neighbors
    ]

    neighbor_fractional = (
        periodic_atoms[order]
    )

    return {
        "center_fractional":
            center_fractional,

        "center_cartesian":
            center_fractional * a_angstrom,

        "neighbor_fractional":
            neighbor_fractional,

        "neighbor_cartesian":
            neighbor_fractional * a_angstrom,

        "displacement_fractional":
            displacement_fractional[order],

        "displacement_cartesian":
            displacement_cartesian[order],

        "distances":
            distances[order],

        "source_indices":
            source_indices[order],

        "translations":
            translations[order],
    }