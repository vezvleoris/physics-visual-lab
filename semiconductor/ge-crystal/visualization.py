"""
PyVista visualization utilities.

Visual distinction
------------------
FCC lattice point:
    small red marker

Basis:
    two atoms attached to one lattice point

Ge crystal atom:
    blue sphere

Selected Ge atom:
    orange sphere

Nearest neighbors:
    yellow spheres

Basis pair R+b1 -> R+b2:
    thin dashed green line (not a chemical bond)

Unit cells:
    thin gray grid for every repeated cell,
    thick black box for the highlighted cell
"""

import numpy as np
import pyvista as pv


ATOM_RADIUS_RATIO = 0.055
LATTICE_RADIUS_RATIO = 0.025
BASIS_RADIUS_RATIO = 0.065
NN_RADIUS_RATIO = 0.065


# ============================================================
# Helpers
# ============================================================

def _spheres(centers, radius):
    """All spheres of one group merged into a single mesh."""

    cloud = pv.PolyData(
        np.asarray(centers, dtype=float).reshape(-1, 3)
    )

    return cloud.glyph(
        geom=pv.Sphere(
            radius=radius,
            theta_resolution=20,
            phi_resolution=20,
        ),
        orient=False,
        scale=False,
    )


def _lines(segments):
    """PolyData made of independent line segments [(p0, p1), ...]."""

    points = np.asarray(
        segments,
        dtype=float,
    ).reshape(-1, 3)

    n = len(points) // 2

    cells = np.column_stack(
        [
            np.full(n, 2),
            np.arange(0, 2 * n, 2),
            np.arange(1, 2 * n, 2),
        ]
    ).ravel()

    return pv.PolyData(points, lines=cells)


def _cell_bounds(a, cell):

    ix, iy, iz = cell

    return [
        ix * a, (ix + 1) * a,
        iy * a, (iy + 1) * a,
        iz * a, (iz + 1) * a,
    ]


# ============================================================
# Unit cells
# ============================================================

def add_supercell_grid(
    plotter,
    a,
    lower,
    upper,
):
    """
    Draw conventional-cell boundaries (integer planes, units of a)
    clipped to the box [lower, upper].
    """

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    planes = [
        np.arange(
            np.ceil(lower[axis] - 1e-6),
            np.floor(upper[axis] + 1e-6) + 1,
        )
        for axis in range(3)
    ]

    segments = []

    for axis in range(3):

        u, v = [k for k in range(3) if k != axis]

        for pu in planes[u]:
            for pv_ in planes[v]:

                start = np.empty(3)
                start[axis] = lower[axis]
                start[u] = pu
                start[v] = pv_

                end = start.copy()
                end[axis] = upper[axis]

                segments.append((start, end))

    if not segments:
        return

    plotter.add_mesh(
        _lines(np.asarray(segments, dtype=float) * a),
        color="gray",
        opacity=0.45,
        line_width=1,
        pickable=False,
        name="supercell_grid",
    )


def highlight_unit_cell(
    plotter,
    a,
    cell=(0, 0, 0),
):
    """Thick box around one conventional cell (volume a³)."""

    box = pv.Box(
        bounds=_cell_bounds(a, cell)
    )

    plotter.add_mesh(
        box,
        style="wireframe",
        color="black",
        line_width=5,
        pickable=False,
        name="selected_unit_cell",
    )


def _add_dimension(
    plotter,
    start,
    end,
    tick,
    text,
    name,
):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    segments = [
        (start, end),
        (start - tick, start + tick),
        (end - tick, end + tick),
    ]

    plotter.add_mesh(
        _lines(segments),
        color="black",
        line_width=2,
        pickable=False,
        name=f"{name}_line",
    )

    plotter.add_point_labels(
        [(start + end) / 2.0],
        [text],
        font_size=14,
        show_points=False,
        always_visible=True,
        shape_color="white",
        shape_opacity=0.85,
        text_color="black",
        name=f"{name}_label",
    )


def add_roi_box(
    plotter,
    a,
    lower,
    upper,
):
    """Outline of the region of interest."""

    box = pv.Box(
        bounds=[
            lower[0] * a, upper[0] * a,
            lower[1] * a, upper[1] * a,
            lower[2] * a, upper[2] * a,
        ]
    )

    plotter.add_mesh(
        box,
        style="wireframe",
        color="teal",
        line_width=3,
        pickable=False,
        name="roi_box",
    )


def add_length_labels(
    plotter,
    a,
    cell=(0, 0, 0),
    span=None,
):
    """
    Dimension line for one cell edge (a) and, when the visible
    region is longer than one cell, for the span along x.

    span : (x_start, x_end) in units of a, or None
    """

    ix, _, iz = cell
    offset = 0.18 * a

    # Vertical edge for a, horizontal for nx·a, so labels never overlap.
    _add_dimension(
        plotter,
        start=((ix + 1) * a, -offset, iz * a),
        end=((ix + 1) * a, -offset, (iz + 1) * a),
        tick=np.array([0.05 * a, 0.0, 0.0]),
        text=f"a = {a:.3f} Å",
        name="a_dimension",
    )

    if span is None:
        return

    x0, x1 = span
    length = x1 - x0

    if length > 1.0 + 1e-6:
        _add_dimension(
            plotter,
            start=(x0 * a, -2.0 * offset, iz * a),
            end=(x1 * a, -2.0 * offset, iz * a),
            tick=np.array([0.0, 0.05 * a, 0.0]),
            text=f"{length:g}a = {length * a:.3f} Å",
            name="total_dimension",
        )


# ============================================================
# FCC Bravais lattice
# ============================================================

def add_lattice_points(
    plotter,
    lattice_points,
    a,
):
    """
    Display FCC Bravais lattice points.

    They are deliberately rendered differently from atoms.
    """

    plotter.add_mesh(
        _spheres(
            np.asarray(lattice_points) * a,
            LATTICE_RADIUS_RATIO * a,
        ),
        color="red",
        name="lattice_points",
    )


# ============================================================
# Basis
# ============================================================

def add_basis(
    plotter,
    basis,
    a,
):
    """
    Show the two-Ge-atom basis attached to R=(0,0,0).
    """

    positions = (
        np.asarray(basis)
        * a
    )

    radius = (
        BASIS_RADIUS_RATIO
        * a
    )

    colors = [
        "mediumseagreen",
        "limegreen",
    ]

    for i, position in enumerate(positions):

        sphere = pv.Sphere(
            radius=radius,
            center=position,
            theta_resolution=24,
            phi_resolution=24,
        )

        plotter.add_mesh(
            sphere,
            color=colors[i],
            smooth_shading=True,
            name=f"basis_{i}",
        )

    # Show displacement b1 -> b2.
    if len(positions) == 2:

        line = pv.Line(
            positions[0],
            positions[1],
        )

        plotter.add_mesh(
            line,
            color="green",
            line_width=4,
            name="basis_displacement",
        )


def add_basis_pairs(
    plotter,
    pairs,
    a,
    dashes=7,
):
    """
    Dashed line R+b1 -> R+b2 for every basis pair.

    Labelled as a basis pair: it marks which two atoms share
    one lattice point, not a chemical bond.
    """

    pairs = np.asarray(pairs, dtype=float) * a

    if len(pairs) == 0:
        return

    t = np.linspace(0.0, 1.0, 2 * dashes)
    segments = []

    for start, end in pairs:
        for k in range(0, len(t) - 1, 2):
            segments.append(
                (
                    start + t[k] * (end - start),
                    start + t[k + 1] * (end - start),
                )
            )

    plotter.add_mesh(
        _lines(segments),
        color="seagreen",
        line_width=2,
        pickable=False,
        name="basis_pairs",
    )

    start, end = pairs[0]

    plotter.add_point_labels(
        [(start + end) / 2.0],
        ["basis pair: R to R+b2 (not a bond)"],
        font_size=12,
        show_points=False,
        always_visible=True,
        shape_color="white",
        shape_opacity=0.8,
        text_color="seagreen",
        name="basis_pair_label",
    )


# ============================================================
# Diamond crystal
# ============================================================

def add_atoms(
    plotter,
    positions,
    a,
    selected_position=None,
):
    """Display resulting Ge crystal atoms."""

    radius = (
        ATOM_RADIUS_RATIO
        * a
    )

    plotter.add_mesh(
        _spheres(positions, radius),
        color="royalblue",
        smooth_shading=True,
        name="atoms",
    )

    if selected_position is not None:

        plotter.add_mesh(
            pv.Sphere(
                radius=1.1 * radius,
                center=selected_position,
                theta_resolution=24,
                phi_resolution=24,
            ),
            color="orange",
            smooth_shading=True,
            name="selected_atom",
        )


# ============================================================
# Nearest neighbors
# ============================================================

def add_neighbor_bonds(
    plotter,
    center,
    neighbors,
    a,
):
    """Draw four nearest-neighbor bonds."""

    bond_radius = 0.012 * a

    for i, neighbor in enumerate(neighbors):

        direction = (
            neighbor - center
        )

        length = np.linalg.norm(
            direction
        )

        midpoint = (
            center + neighbor
        ) / 2.0

        cylinder = pv.Cylinder(
            center=midpoint,
            direction=direction,
            radius=bond_radius,
            height=length,
            resolution=24,
        )

        plotter.add_mesh(
            cylinder,
            color="orange",
            name=f"nn_bond_{i}",
        )


def add_neighbor_atoms(
    plotter,
    neighbors,
    a,
):
    """Highlight four nearest neighbors."""

    plotter.add_mesh(
        _spheres(neighbors, NN_RADIUS_RATIO * a),
        color="gold",
        smooth_shading=True,
        name="nn_atoms",
    )


# ============================================================
# Conventional-cell atom counting
# ============================================================

COUNTING_STYLE = {
    "corner": ("red", 0.050),
    "edge": ("purple", 0.052),
    "face": ("orange", 0.055),
    "internal": ("royalblue", 0.060),
}


def add_counting_atoms(
    plotter,
    groups,
    a,
):
    """
    Atoms of the highlighted cell, colored by how they are
    shared with neighboring cells.

    groups : {"corner": positions, "face": ..., "internal": ...}
        fractional positions

    corner   -> red
    face     -> orange
    internal -> blue
    """

    for kind, positions in groups.items():

        if len(positions) == 0:
            continue

        color, radius_ratio = COUNTING_STYLE[kind]

        plotter.add_mesh(
            _spheres(
                np.asarray(positions) * a,
                radius_ratio * a,
            ),
            color=color,
            smooth_shading=True,
            name=f"count_{kind}",
        )


def add_background_atoms(
    plotter,
    positions,
    a,
):
    """Faint atoms of the rest of the crystal."""

    if len(positions) == 0:
        return

    plotter.add_mesh(
        _spheres(
            np.asarray(positions) * a,
            0.045 * a,
        ),
        color="lightsteelblue",
        opacity=0.25,
        smooth_shading=True,
        name="background_atoms",
    )


def add_picked_marker(
    plotter,
    center,
    a,
):
    """Wire shell around the atom picked in counting mode."""

    plotter.add_mesh(
        pv.Sphere(
            radius=0.09 * a,
            center=center,
            theta_resolution=16,
            phi_resolution=16,
        ),
        style="wireframe",
        color="black",
        line_width=1,
        pickable=False,
        name="picked_marker",
    )
