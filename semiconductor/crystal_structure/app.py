"""
Interactive explanation of the (1/4, 1/4, 1/4)
basis position in the diamond cubic structure.

Run
---
streamlit run app.py
"""

import numpy as np
import streamlit as st

from geometry import (
    get_basis_position,
    get_fcc_lattice_points,
    get_pairwise_distances,
    get_tetrahedral_site,
    get_tetrahedron_vertices,
)

from visualization import (
    add_basis_atom,
    add_coordinate_labels,
    add_displacement_line,
    add_lattice_points,
    add_tetrahedral_site,
    add_tetrahedron,
    configure_figure,
    create_unit_cell,
)


st.set_page_config(
    page_title="Why 1/4? — Diamond Cubic",
    layout="wide",
)


def main():
    st.title("Why (1/4, 1/4, 1/4)?")
    st.caption(
        "FCC geometry → tetrahedral site → diamond cubic basis"
    )

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    fcc_points = get_fcc_lattice_points()

    tetra_vertices = get_tetrahedron_vertices()

    tetra_site = get_tetrahedral_site(tetra_vertices)

    distances = get_pairwise_distances(tetra_vertices)

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------

    st.sidebar.header("Visualization")

    show_fcc = st.sidebar.checkbox(
        "FCC lattice points",
        value=True,
    )

    show_coordinates = st.sidebar.checkbox(
        "Coordinate labels",
        value=False,
    )

    show_tetrahedron = st.sidebar.checkbox(
        "FCC tetrahedron",
        value=True,
    )

    show_site = st.sidebar.checkbox(
        "Tetrahedral site",
        value=True,
    )

    st.sidebar.divider()

    st.sidebar.subheader("Second basis atom")

    t = st.sidebar.slider(
        "Displacement t",
        min_value=0.0,
        max_value=0.5,
        value=0.25,
        step=0.01,
    )

    show_basis = st.sidebar.checkbox(
        "Show second basis atom",
        value=True,
    )

    basis_position = get_basis_position(t)

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------

    plot_col, explanation_col = st.columns(
        [1.6, 1.0]
    )

    # ---------------------------------------------------------
    # 3D visualization
    # ---------------------------------------------------------

    with plot_col:

        fig = create_unit_cell()

        if show_fcc:
            add_lattice_points(
                fig,
                fcc_points,
            )

        if show_coordinates:
            add_coordinate_labels(
                fig,
                tetra_vertices,
            )

        if show_tetrahedron:
            add_tetrahedron(
                fig,
                tetra_vertices,
            )

        if show_site:
            add_tetrahedral_site(
                fig,
                tetra_site,
            )

        if show_basis:
            add_basis_atom(
                fig,
                basis_position,
            )

            add_displacement_line(
                fig,
                basis_position,
            )

        configure_figure(fig)

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    with explanation_col:

        st.subheader("1. Start from FCC")

        st.markdown(
            r"""
Choose one FCC corner

$$
A=(0,0,0)
$$

and the three neighboring face centers

$$
B=(0,\frac12,\frac12)
$$

$$
C=(\frac12,0,\frac12)
$$

$$
D=(\frac12,\frac12,0).
$$
"""
        )

        st.subheader("2. These points form a tetrahedron")

        values = np.array(list(distances.values()))

        st.latex(
            r"|AB|=|AC|=|AD|=|BC|=|BD|=|CD|"
        )

        st.write(
            f"Pairwise distance ≈ {values[0]:.4f} a"
        )

        if np.allclose(values, values[0]):
            st.success(
                "All six edges have equal length → regular tetrahedron"
            )

        st.subheader("3. Find its center")

        st.latex(
            r"""
\mathbf r_{\mathrm{tet}}
=
\frac{
\mathbf A+\mathbf B+\mathbf C+\mathbf D
}{4}
"""
        )

        st.latex(
            r"""
=
\left(
\frac14,\,
\frac14,\,
\frac14
\right)
"""
        )

        st.write(
            "Calculated by geometry.py:",
            tetra_site,
        )

        st.subheader("4. Move the second basis atom")

        st.latex(
            rf"""
\mathbf b_2
=
(t,t,t)
=
({t:.2f},{t:.2f},{t:.2f})
"""
        )

        distance_to_site = np.linalg.norm(
            basis_position - tetra_site
        )

        st.metric(
            "Distance from tetrahedral site",
            f"{distance_to_site:.4f} a",
        )

        if np.isclose(t, 0.25):
            st.success(
                "t = 1/4: the basis atom coincides "
                "with the tetrahedral site."
            )
        elif t < 0.25:
            st.info(
                "The basis atom has not reached "
                "the tetrahedral site yet."
            )
        else:
            st.info(
                "The basis atom has passed "
                "the tetrahedral site."
            )

        st.divider()

        st.subheader("Diamond cubic")

        st.latex(
            r"""
\mathrm{Diamond}
=
\mathrm{FCC}
+
\left\{
(0,0,0),
\left(\frac14,\frac14,\frac14\right)
\right\}
"""
        )

        st.markdown(
            """
The **1/4 is not obtained from the Ge lattice constant**.

It comes from the geometry of the FCC tetrahedral site.
The lattice constant only converts these fractional coordinates
into actual lengths.
"""
        )


if __name__ == "__main__":
    main()