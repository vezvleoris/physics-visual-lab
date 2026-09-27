"""
Plotly visualization utilities for FCC tetrahedral geometry.

This module does not calculate crystal geometry.
It receives coordinates from geometry.py and renders them.
"""

from itertools import combinations

import numpy as np
import plotly.graph_objects as go


def create_unit_cell():
    """
    Create an empty 3D figure containing the cubic unit-cell wireframe.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    fig = go.Figure()

    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )

    # Pairs of corner indices connected by cube edges
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    for i, j in edges:
        p1 = corners[i]
        p2 = corners[j]

        fig.add_trace(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(color="gray", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


def add_lattice_points(fig, fcc_points):
    """
    Add FCC corner and face-centered lattice points to the figure.
    """

    corners = fcc_points["corners"]
    face_centers = fcc_points["face_centers"]

    fig.add_trace(
        go.Scatter3d(
            x=corners[:, 0],
            y=corners[:, 1],
            z=corners[:, 2],
            mode="markers",
            name="FCC corner",
            marker=dict(
                size=7,
                color="royalblue",
            ),
            hovertemplate=(
                "FCC corner<br>"
                "(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=face_centers[:, 0],
            y=face_centers[:, 1],
            z=face_centers[:, 2],
            mode="markers",
            name="FCC face center",
            marker=dict(
                size=7,
                color="cornflowerblue",
            ),
            hovertemplate=(
                "FCC face center<br>"
                "(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                "<extra></extra>"
            ),
        )
    )


def add_coordinate_labels(fig, points):
    """
    Add fractional-coordinate labels to selected points.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    """

    points = np.asarray(points)

    labels = [
        f"({x:g}, {y:g}, {z:g})"
        for x, y, z in points
    ]

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="text",
            text=labels,
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        )
    )


def add_tetrahedron(fig, vertices):
    """
    Draw the regular tetrahedron formed by four FCC lattice points.
    """

    vertices = np.asarray(vertices)

    # Draw all six edges
    for i, j in combinations(range(4), 2):
        p1 = vertices[i]
        p2 = vertices[j]

        fig.add_trace(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(
                    color="orange",
                    width=6,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers+text",
            text=["A", "B", "C", "D"],
            textposition="top center",
            name="FCC tetrahedron",
            marker=dict(
                size=10,
                color="orange",
            ),
        )
    )


def add_tetrahedral_site(fig, site):
    """
    Highlight the tetrahedral site calculated by geometry.py.
    """

    site = np.asarray(site)

    fig.add_trace(
        go.Scatter3d(
            x=[site[0]],
            y=[site[1]],
            z=[site[2]],
            mode="markers+text",
            text=["(1/4, 1/4, 1/4)"],
            textposition="top center",
            name="Tetrahedral site",
            marker=dict(
                size=12,
                color="crimson",
                symbol="diamond",
            ),
            hovertemplate=(
                "Tetrahedral site<br>"
                "x=%{x:.3f}<br>"
                "y=%{y:.3f}<br>"
                "z=%{z:.3f}"
                "<extra></extra>"
            ),
        )
    )


def add_basis_atom(fig, position):
    """
    Add the movable second basis atom at (t, t, t).
    """

    position = np.asarray(position)

    fig.add_trace(
        go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode="markers",
            name="Second basis atom",
            marker=dict(
                size=9,
                color="limegreen",
                symbol="circle",
            ),
            hovertemplate=(
                "Second basis atom<br>"
                "(%{x:.3f}, %{y:.3f}, %{z:.3f})"
                "<extra></extra>"
            ),
        )
    )


def add_displacement_line(fig, position):
    """
    Draw the displacement from (0,0,0) to the movable basis atom.
    """

    position = np.asarray(position)

    fig.add_trace(
        go.Scatter3d(
            x=[0, position[0]],
            y=[0, position[1]],
            z=[0, position[2]],
            mode="lines",
            name="[111] displacement",
            line=dict(
                color="limegreen",
                width=5,
                dash="dash",
            ),
        )
    )


def configure_figure(fig):
    """
    Apply common layout settings to the crystal visualization.
    """

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x / a", range=[-0.05, 1.05]),
            yaxis=dict(title="y / a", range=[-0.05, 1.05]),
            zaxis=dict(title="z / a", range=[-0.05, 1.05]),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.25)
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        legend=dict(
            x=0,
            y=1,
        ),
        height=700,
    )

    return fig