"""
Ge Diamond Cubic Visualizer

Run:
    python3 app.py
"""

import sys
from pathlib import Path

import numpy as np
import yaml

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyvistaqt import QtInteractor

from crystal import (
    build_diamond_crystal,
    build_display_supercell,
    classify_cell_site,
    clip_to_box,
    find_nearest_neighbors,
    fold_to_cell_index,
    get_basis_pairs,
    get_diamond_conventional_display_atoms,
    get_fcc_conventional_display_points,
)

from properties import (
    number_density,
    mass_density,
    valence_electron_density,
)

_COUNTING_LEGEND = (
    "<span style='color:#e74c3c;'>●</span> corner: "
    "1/8 belongs to this cell<br>"
    "<span style='color:#e67e22;'>●</span> face: "
    "1/2 belongs to this cell<br>"
    "<span style='color:#4169e1;'>●</span> internal: "
    "1 belongs to this cell<br>"
    "<i>Black box = one unit cell (side a). "
    "Click an atom to see how many cells share it.</i>"
    "<br><br>"
)

_FRACTION_TEXT = {
    1: "1",
    2: "1/2",
    4: "1/4",
    8: "1/8",
}

MAX_REPEAT = 4

# ROI side length in hundredths of a (1.00a → 2.00a).
ROI_MIN = 100
ROI_MAX = 200


from visualization import (
    add_atoms,
    add_background_atoms,
    add_basis,
    add_basis_pairs,
    add_counting_atoms,
    add_lattice_points,
    add_length_labels,
    add_neighbor_atoms,
    add_neighbor_bonds,
    add_picked_marker,
    add_roi_box,
    add_supercell_grid,
    highlight_unit_cell,
)


# ============================================================
# Config
# ============================================================

HERE = Path(__file__).resolve().parent
SEMICONDUCTOR_DIR = HERE.parent

GE_CONFIG = (
    SEMICONDUCTOR_DIR
    / "config"
    / "materials"
    / "ge.yaml"
)

DIAMOND_CONFIG = (
    SEMICONDUCTOR_DIR
    / "config"
    / "structures"
    / "diamond.yaml"
)


def load_yaml(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# Main application
# ============================================================

class CrystalViewer(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Ge Diamond Cubic | Physics Visual Lab"
        )

        self.resize(
            1250,
            820,
        )

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        self.material = load_yaml(
            GE_CONFIG
        )

        self.structure = load_yaml(
            DIAMOND_CONFIG
        )

        self.a = float(
            self.material[
                "crystal"
            ][
                "lattice_constant_angstrom"
            ]
        )

        self.molar_mass = float(
            self.material[
                "atomic"
            ][
                "molar_mass_g_mol"
            ]
        )

        self.valence = int(
            self.material[
                "atomic"
            ][
                "valence_electrons"
            ]
        )

        # ----------------------------------------------------
        # Build crystal
        # ----------------------------------------------------

        crystal = build_diamond_crystal(
            self.structure[
                "basis_fractional"
            ]
        )

        self.lattice_points = (
            crystal["lattice_points"]
        )

        self.basis = (
            crystal["basis"]
        )

        # One conventional cell, used for periodic NN search.
        self.fractional_atoms = (
            crystal["atoms"]
        )

        # Drawn cell for atom counting. Not used for NN search.
        self.display_atoms = (
            get_diamond_conventional_display_atoms()
        )

        self.fcc_display_points = (
            get_fcc_conventional_display_points()
        )

        # Fractional positions (units of a) in the supercell.
        self.selected_position = None
        self.picked_position = None

        # structure: Crystal structure combo owns the 3D view.
        # homework: Homework combo owns the 3D view.
        self.scene_mode = "homework"

        # ----------------------------------------------------
        # Main layout
        # ----------------------------------------------------

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(
            central
        )

        # 3D viewport
        self.plotter = QtInteractor(
            self
        )

        main_layout.addWidget(
            self.plotter.interactor,
            stretch=4,
        )

        # Right panel. Homework steps are long, so this scrolls.
        panel = QWidget()

        panel_layout = QVBoxLayout(
            panel
        )

        scroll = QScrollArea()

        scroll.setWidgetResizable(True)
        scroll.setFrameShape(
            QScrollArea.Shape.NoFrame
        )
        scroll.setWidget(panel)
        scroll.setMaximumWidth(400)

        main_layout.addWidget(
            scroll,
            stretch=1,
        )

        # ----------------------------------------------------
        # Structure view
        # ----------------------------------------------------

        panel_layout.addWidget(
            QLabel(
                "<h2>Crystal structure</h2>"
            )
        )

        self.view_combo = QComboBox()

        self.view_combo.addItems(
            [
                "FCC Bravais lattice",
                "Ge basis",
                "Ge diamond crystal",
            ]
        )

        # Start with complete crystal.
        self.view_combo.setCurrentIndex(
            2
        )

        panel_layout.addWidget(
            self.view_combo
        )

        self.view_description = QLabel()

        self.view_description.setWordWrap(
            True
        )

        panel_layout.addWidget(
            self.view_description
        )

        # ----------------------------------------------------
        # Homework steps
        # ----------------------------------------------------

        panel_layout.addSpacing(20)

        panel_layout.addWidget(
            QLabel("<h2>Homework</h2>")
        )

        self.homework_combo = QComboBox()

        self.homework_combo.addItems(
            [
                "(a) Unit cell",
                "(b) Nearest-neighbor distance",
                "(c) Number density",
                "(d) Mass density",
                "(e) Valence electron density",
            ]
        )

        panel_layout.addWidget(
            self.homework_combo
        )

        self.homework_explanation = QLabel()

        self.homework_explanation.setWordWrap(
            True
        )

        panel_layout.addWidget(
            self.homework_explanation
        )

        # ----------------------------------------------------
        # Lattice constant
        # ----------------------------------------------------

        panel_layout.addSpacing(15)

        form = QFormLayout()

        self.a_spin = QDoubleSpinBox()

        self.a_spin.setRange(
            3.0,
            10.0,
        )

        self.a_spin.setDecimals(3)
        self.a_spin.setSingleStep(0.01)
        self.a_spin.setValue(self.a)
        self.a_spin.setSuffix(" Å")

        form.addRow(
            "Lattice constant a:",
            self.a_spin,
        )

        panel_layout.addLayout(
            form
        )

        # ----------------------------------------------------
        # Crystal repetition
        # ----------------------------------------------------

        panel_layout.addSpacing(15)

        panel_layout.addWidget(
            QLabel(
                "<h3>Crystal repetition</h3>"
            )
        )

        repeat_form = QFormLayout()

        self.repeat_spins = []

        for axis in ("X", "Y", "Z"):

            spin = QSpinBox()
            spin.setRange(1, MAX_REPEAT)
            spin.setValue(2)
            spin.setSuffix(" cells")

            repeat_form.addRow(
                f"{axis} cells:",
                spin,
            )

            self.repeat_spins.append(spin)

        cell_row = QHBoxLayout()

        self.cell_spins = []

        for axis in ("i", "j", "k"):

            spin = QSpinBox()
            spin.setPrefix(f"{axis}=")
            spin.setRange(0, 1)

            cell_row.addWidget(spin)

            self.cell_spins.append(spin)

        repeat_form.addRow(
            "Highlighted cell:",
            cell_row,
        )

        panel_layout.addLayout(
            repeat_form
        )

        # ----------------------------------------------------
        # Region of interest
        # ----------------------------------------------------

        self.roi_check = QCheckBox(
            "Clip crystal to ROI (starts at highlighted cell)"
        )
        self.roi_check.setChecked(True)

        panel_layout.addWidget(self.roi_check)

        roi_row = QHBoxLayout()

        self.roi_slider = QSlider(Qt.Horizontal)
        self.roi_slider.setRange(ROI_MIN, ROI_MAX)
        self.roi_slider.setSingleStep(5)
        self.roi_slider.setPageStep(25)
        self.roi_slider.setTickInterval(25)
        self.roi_slider.setTickPosition(
            QSlider.TicksBelow
        )
        self.roi_slider.setValue(ROI_MAX)

        self.roi_value_label = QLabel()
        self.roi_value_label.setMinimumWidth(48)

        roi_row.addWidget(QLabel("ROI size:"))
        roi_row.addWidget(self.roi_slider, 1)
        roi_row.addWidget(self.roi_value_label)

        panel_layout.addLayout(roi_row)

        self.highlight_check = QCheckBox(
            "Highlight one unit cell"
        )
        self.highlight_check.setChecked(True)

        self.boundaries_check = QCheckBox(
            "Show cell boundaries"
        )
        self.boundaries_check.setChecked(True)

        self.pairs_check = QCheckBox(
            "Show basis pairs (R → R + b₂)"
        )
        self.pairs_check.setChecked(True)

        self.grid_check = QCheckBox(
            "Show coordinate grid"
        )
        self.grid_check.setChecked(False)

        for check in (
            self.highlight_check,
            self.boundaries_check,
            self.pairs_check,
            self.grid_check,
        ):
            panel_layout.addWidget(check)

        self.cell_info_label = QLabel()

        self.cell_info_label.setWordWrap(
            True
        )

        panel_layout.addWidget(
            self.cell_info_label
        )

        # ----------------------------------------------------
        # Selection
        # ----------------------------------------------------

        panel_layout.addSpacing(15)

        panel_layout.addWidget(
            QLabel(
                "<h3>Selected atom</h3>"
            )
        )

        self.atom_label = QLabel(
            "Selected atom: none"
        )

        self.atom_label.setWordWrap(
            True
        )

        self.nn_label = QLabel()

        self.nn_label.setWordWrap(
            True
        )

        panel_layout.addWidget(
            self.atom_label
        )

        panel_layout.addWidget(
            self.nn_label
        )

        reset_button = QPushButton(
            "Reset selection"
        )

        panel_layout.addWidget(
            reset_button
        )

        panel_layout.addStretch()

        # ----------------------------------------------------
        # Signals
        # ----------------------------------------------------

        self.view_combo.currentIndexChanged.connect(
            self.on_view_changed
        )

        self.homework_combo.currentIndexChanged.connect(
            self.on_homework_changed
        )

        self.a_spin.valueChanged.connect(
            self.on_lattice_constant_changed
        )

        for spin in self.repeat_spins:
            spin.valueChanged.connect(
                self.on_repetition_changed
            )

        for spin in self.cell_spins:
            spin.valueChanged.connect(
                self.on_roi_changed
            )

        self.roi_check.toggled.connect(
            self.on_roi_changed
        )

        self.roi_slider.valueChanged.connect(
            self.on_roi_changed
        )

        for check in (
            self.highlight_check,
            self.boundaries_check,
            self.pairs_check,
            self.grid_check,
        ):
            check.toggled.connect(
                self.on_display_option_changed
            )

        reset_button.clicked.connect(
            self.reset_selection
        )

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        self._rebuild_supercell()
        self.on_homework_changed()
        self.enable_atom_picking()

    # ========================================================
    # Supercell state
    # ========================================================

    @property
    def repeats(self):

        return tuple(
            spin.value()
            for spin in self.repeat_spins
        )

    @property
    def highlighted_cell(self):

        return tuple(
            spin.value()
            for spin in self.cell_spins
        )

    def _rebuild_supercell(self):

        nx, ny, nz = self.repeats

        for spin, n in zip(self.cell_spins, self.repeats):
            spin.blockSignals(True)
            spin.setMaximum(n - 1)
            spin.blockSignals(False)

        self._all_atoms = build_display_supercell(
            self.display_atoms,
            nx,
            ny,
            nz,
        )

        self._all_lattice_points = build_display_supercell(
            self.fcc_display_points,
            nx,
            ny,
            nz,
        )

        self._all_basis_pairs = get_basis_pairs(
            self._all_lattice_points,
            self.basis,
            nx,
            ny,
            nz,
        )

        self._apply_roi()

    @property
    def roi_length(self):
        """ROI side length in units of a."""

        return self.roi_slider.value() / 100.0

    def _visible_bounds(self):
        """(lower, upper) of the displayed region, in units of a."""

        repeats = np.array(self.repeats, dtype=float)

        if not self.roi_check.isChecked():
            return np.zeros(3), repeats

        lower = np.array(self.highlighted_cell, dtype=float)
        upper = np.minimum(lower + self.roi_length, repeats)

        return lower, upper

    def _apply_roi(self):
        """Restrict atoms / lattice points / pairs to the ROI."""

        self.roi_value_label.setText(
            f"{self.roi_length:.2f}a"
        )
        self.roi_slider.setEnabled(
            self.roi_check.isChecked()
        )

        lower, upper = self._visible_bounds()

        self.super_atoms = clip_to_box(
            self._all_atoms,
            lower,
            upper,
        )

        self.super_lattice_points = clip_to_box(
            self._all_lattice_points,
            lower,
            upper,
        )

        pairs = self._all_basis_pairs

        if len(pairs):
            tol = 1e-6
            inside = np.all(
                (pairs >= lower - tol) & (pairs <= upper + tol),
                axis=(1, 2),
            )
            pairs = pairs[inside]

        self.basis_pairs = pairs

        for attr in ("selected_position", "picked_position"):

            position = getattr(self, attr, None)

            if position is not None and not len(
                clip_to_box(position, lower, upper)
            ):
                setattr(self, attr, None)

    def _is_counting_step(self):

        return (
            self.scene_mode == "homework"
            and self.homework_combo.currentIndex() >= 2
        )

    def _nn_pickable(self):

        return (
            not self._is_counting_step()
            and self.view_combo.currentText()
            == "Ge diamond crystal"
        )

    def _show_basis_pairs(self):

        # (b) draws NN bonds on the same atoms; pairs would clutter it.
        in_nn_step = (
            self.scene_mode == "homework"
            and self.homework_combo.currentIndex() == 1
        )

        return (
            self.pairs_check.isChecked()
            and not in_nn_step
        )

    def _cell_groups(self):
        """Supercell atoms split by role in the highlighted cell."""

        groups = {
            "corner": [],
            "edge": [],
            "face": [],
            "internal": [],
        }

        outside = []

        for position in self.super_atoms:

            site = classify_cell_site(
                position,
                self.highlighted_cell,
            )

            if site is None:
                outside.append(position)
            else:
                groups[site["kind"]].append(position)

        return groups, outside

    def _update_cell_info(self):

        ix, iy, iz = self.highlighted_cell
        nx, ny, nz = self.repeats
        a = self.a

        groups, _ = self._cell_groups()

        shares = {
            "corner": 8,
            "edge": 4,
            "face": 2,
            "internal": 1,
        }

        terms = []
        total = 0.0

        for kind in ("corner", "edge", "face", "internal"):

            count = len(groups[kind])

            if count == 0:
                continue

            terms.append(
                f"{count}×{_FRACTION_TEXT[shares[kind]]}"
            )

            total += count / shares[kind]

        if self.roi_check.isChecked():
            lower, upper = self._visible_bounds()
            size = upper - lower
            roi_text = (
                "ROI: "
                + " × ".join(f"{s:g}a" for s in size)
                + f" ({len(self.super_atoms)} atoms shown)"
                "<br><br>"
            )
        else:
            roi_text = ""

        self.cell_info_label.setText(
            f"Crystal: {nx}×{ny}×{nz} cells "
            f"= {nx * ny * nz} unit cells<br>"
            + roi_text
            + "<br>"
            "<b>Highlighted unit cell</b><br>"
            f"x : {ix}a → {ix + 1}a<br>"
            f"y : {iy}a → {iy + 1}a<br>"
            f"z : {iz}a → {iz + 1}a<br>"
            f"Volume = a³ = {a ** 3:.2f} Å³<br>"
            "Effective Ge atoms = "
            + " + ".join(terms)
            + f" = <b>{total:g}</b><br>"
            f"n<sub>Ge</sub> = {total:g}/a³"
        )

    # ========================================================
    # Scene
    # ========================================================

    def _add_cell_frame(self):

        lower, upper = self._visible_bounds()

        if self.boundaries_check.isChecked():
            add_supercell_grid(
                self.plotter,
                self.a,
                lower,
                upper,
            )

        if self.roi_check.isChecked():
            add_roi_box(
                self.plotter,
                self.a,
                lower,
                upper,
            )

        if self.highlight_check.isChecked():
            highlight_unit_cell(
                self.plotter,
                self.a,
                self.highlighted_cell,
            )

        add_length_labels(
            self.plotter,
            self.a,
            self.highlighted_cell,
            span=(lower[0], upper[0]),
        )

    def draw_scene(self, reset_camera=True):

        self.plotter.clear()

        self._add_cell_frame()

        view = (
            self.view_combo.currentText()
        )

        # ----------------------------------------------------
        # FCC lattice
        # ----------------------------------------------------

        if view == "FCC Bravais lattice":

            add_lattice_points(
                self.plotter,
                self.super_lattice_points,
                self.a,
            )

            self.view_description.setText(
                "<b>FCC Bravais lattice</b><br>"
                "Red points are lattice points R.<br><br>"
                "They describe translational symmetry. "
                "They are not the Ge basis atoms."
            )

        # ----------------------------------------------------
        # Basis
        # ----------------------------------------------------

        elif view == "Ge basis":

            # Show one lattice point for reference.
            add_lattice_points(
                self.plotter,
                np.array(
                    [[0.0, 0.0, 0.0]]
                ),
                self.a,
            )

            add_basis(
                self.plotter,
                self.basis,
                self.a,
            )

            self.view_description.setText(
                "<b>Ge basis</b><br>"
                "Attach two Ge atoms to each FCC "
                "lattice point:<br><br>"
                "b₁ = (0, 0, 0)<br>"
                "b₂ = (1/4, 1/4, 1/4)"
            )

        # ----------------------------------------------------
        # Diamond crystal
        # ----------------------------------------------------

        else:

            selected = (
                None
                if self.selected_position is None
                else self.selected_position * self.a
            )

            add_atoms(
                self.plotter,
                self.super_atoms * self.a,
                self.a,
                selected_position=selected,
            )

            if self._show_basis_pairs():
                add_basis_pairs(
                    self.plotter,
                    self.basis_pairs,
                    self.a,
                )

            self.view_description.setText(
                "<b>Ge diamond cubic</b><br>"
                "Crystal positions are generated by:<br><br>"
                "r = R + b<br><br>"
                "FCC lattice + 2-Ge-atom basis "
                "→ diamond cubic.<br><br>"
                "Dashed green lines join the two basis "
                "atoms R and R + (¼,¼,¼) that share one "
                "lattice point. They mark basis pairs, "
                "not chemical bonds."
            )

        self.setup_camera(reset_camera)

    def setup_camera(self, reset_camera=True):

        self.plotter.add_axes()

        if self.grid_check.isChecked():
            self.plotter.show_grid(
                xtitle="x (Å)",
                ytitle="y (Å)",
                ztitle="z (Å)",
            )

        if reset_camera:
            # Pure isometric looks along [111] = b2 direction,
            # which hides every basis pair behind its partner.
            self.plotter.view_isometric()
            self.plotter.camera.Azimuth(22)
            self.plotter.camera.Elevation(-12)
            self.plotter.reset_camera()

    def _show_counting_cell(self, reset_camera=True):

        self.plotter.clear()

        self._add_cell_frame()

        if self.highlight_check.isChecked():

            groups, outside = self._cell_groups()

            add_counting_atoms(
                self.plotter,
                groups,
                self.a,
            )

            add_background_atoms(
                self.plotter,
                outside,
                self.a,
            )

        else:

            add_atoms(
                self.plotter,
                self.super_atoms * self.a,
                self.a,
            )

        if self.picked_position is not None:
            add_picked_marker(
                self.plotter,
                self.picked_position * self.a,
                self.a,
            )

        self.setup_camera(reset_camera)

    def _clear_neighbor_readout(self):

        self.atom_label.setText(
            "Selected atom: none"
        )

        if self._is_counting_step():
            self.nn_label.setText(
                "Click an atom to see how it is "
                "shared with neighboring cells."
            )
        else:
            self.nn_label.setText(
                "In (b), or in Ge diamond crystal, "
                "click an atom."
            )

    # ========================================================
    # Homework
    # ========================================================

    def on_homework_changed(self, _index=None):

        self.scene_mode = "homework"
        self.selected_position = None
        self.picked_position = None
        self._clear_neighbor_readout()
        self._render_homework()

    def _render_homework(self, reset_camera=True):

        step = self.homework_combo.currentIndex()

        # ====================================================
        # (a) Unit cell
        # ====================================================

        if step == 0:

            self.draw_scene(reset_camera)

            self.homework_explanation.setText(
                "<h3>(a) Ge unit cell</h3>"
                "Ge has the diamond-cubic structure."
                "<br><br>"
                "<b>FCC Bravais lattice</b>"
                "<br>"
                "+ "
                "<b>2-Ge-atom basis</b>"
                "<br>"
                "↓"
                "<br>"
                "<b>Diamond cubic</b>"
                "<br><br>"
                "The whole crystal is the black-boxed "
                "cube repeated along x, y, z."
            )

        # ====================================================
        # (b) Nearest-neighbor distance
        # ====================================================

        elif step == 1:

            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentText(
                "Ge diamond crystal"
            )
            self.view_combo.blockSignals(False)

            self.draw_scene(reset_camera)

            self.homework_explanation.setText(
                "<h3>(b) Nearest neighbor</h3>"
                "Click any Ge atom."
                "<br><br>"
                "The four nearest neighbors "
                "will be highlighted."
                "<br><br>"
                "For example:"
                "<br>"
                "(0,0,0)"
                " → "
                "(1/4,1/4,1/4)"
                "<br><br>"
                "Therefore:"
                "<br>"
                "<b>"
                "d = √[(a/4)²+(a/4)²+(a/4)²]"
                "</b>"
                "<br>"
                "<b>d = √3 a / 4</b>"
            )

        # ====================================================
        # (c) Number density
        # ====================================================

        elif step == 2:

            self._show_counting_cell(reset_camera)

            n_ge = number_density(
                self.a
            )

            self.homework_explanation.setText(
                "<h3>(c) Number density</h3>"
                f"{_COUNTING_LEGEND}"
                "<b>① Diamond conventional cell</b>"
                "<br>"
                "The crystal is this cube repeated; "
                "count only one cube."
                "<br><br>"
                "<b>② Effective atoms per cell</b>"
                "<br>"
                "corners : 8 × 1/8 = 1"
                "<br>"
                "faces : 6 × 1/2 = 3"
                "<br>"
                "inside : 4 × 1 = 4"
                "<br>"
                "<b>total = 8 atoms</b>"
                "<br><br>"
                "<b>③ Cell volume</b>"
                "<br>"
                "V = a³"
                "<br><br>"
                "<b>④ Number density</b>"
                "<br>"
                "n<sub>Ge</sub> = 8/a³"
                "<br><br>"
                f"a = {self.a:.2f} Å"
                "<br>"
                f"= {self.a:.2f} × 10⁻⁸ cm"
                "<br><br>"
                f"n<sub>Ge</sub> = "
                f"{n_ge:.4e} atoms/cm³"
            )

        # ====================================================
        # (d) Mass density
        # ====================================================

        elif step == 3:

            self._show_counting_cell(reset_camera)

            n_ge = number_density(
                self.a
            )

            rho = mass_density(
                self.a,
                self.molar_mass,
            )

            self.homework_explanation.setText(
                "<h3>(d) Mass density</h3>"
                f"{_COUNTING_LEGEND}"
                "From (c):"
                "<br>"
                f"n<sub>Ge</sub> = "
                f"{n_ge:.4e} atoms/cm³"
                "<br><br>"
                "<b>Mass of one Ge atom</b>"
                "<br>"
                "m<sub>Ge</sub> = M/N<sub>A</sub>"
                "<br>"
                f"= {self.molar_mass} / "
                "6.022×10²³"
                "<br>"
                "g/atom"
                "<br><br>"
                "<b>Mass density</b>"
                "<br>"
                "ρ = n<sub>Ge</sub>"
                " × m<sub>Ge</sub>"
                "<br><br>"
                f"<b>ρ = {rho:.4f} g/cm³</b>"
            )

        # ====================================================
        # (e) Valence electron density
        # ====================================================

        elif step == 4:

            self._show_counting_cell(reset_camera)

            n_ge = number_density(
                self.a
            )

            n_e = valence_electron_density(
                self.a,
                self.valence,
            )

            self.homework_explanation.setText(
                "<h3>(e) Valence electron density</h3>"
                f"{_COUNTING_LEGEND}"
                "Ge is group 14."
                "<br>"
                "<b>1 Ge atom → 4 valence electrons</b>"
                "<br><br>"
                "From (c):"
                "<br>"
                f"n<sub>Ge</sub> = "
                f"{n_ge:.4e} atoms/cm³"
                "<br><br>"
                "Therefore:"
                "<br>"
                "n<sub>e</sub> "
                "= 4 n<sub>Ge</sub>"
                "<br><br>"
                f"<b>"
                f"n<sub>e</sub> = "
                f"{n_e:.4e} electrons/cm³"
                f"</b>"
            )

        self._update_cell_info()

    # ========================================================
    # Picking
    # ========================================================

    def enable_atom_picking(self):

        self.plotter.enable_point_picking(
            callback=self.on_point_picked,
            left_clicking=True,
            show_message=False,
            show_point=False,
            picker="point",
            tolerance=0.03,
        )

    def on_point_picked(
        self,
        point,
    ):

        if point is None:
            return

        if not (
            self._is_counting_step()
            or self._nn_pickable()
        ):
            return

        cartesian = self.super_atoms * self.a

        distances = np.linalg.norm(
            cartesian - np.asarray(point),
            axis=1,
        )

        index = int(
            np.argmin(distances)
        )

        # Reject empty-space clicks.
        if (
            distances[index]
            > 0.20 * self.a
        ):
            return

        position = self.super_atoms[index]

        if self._is_counting_step():
            self.show_cell_contribution(position)
        else:
            self.select_atom(position)

    def show_cell_contribution(
        self,
        position,
    ):

        self.picked_position = position

        add_picked_marker(
            self.plotter,
            position * self.a,
            self.a,
        )

        site = classify_cell_site(
            position,
            self.highlighted_cell,
        )

        coordinate = (
            f"({position[0]:.2f}, "
            f"{position[1]:.2f}, "
            f"{position[2]:.2f}) a"
        )

        if site is None:

            self.atom_label.setText(
                "<b>Outside the highlighted cell</b><br>"
                f"r = {coordinate}"
            )

            self.nn_label.setText(
                "contribution to this cell = 0"
            )

        else:

            shared = site["shared_by"]

            self.atom_label.setText(
                f"<b>{site['kind'].capitalize()} atom</b><br>"
                f"r = {coordinate}"
            )

            self.nn_label.setText(
                f"shared by {shared} "
                f"cell{'s' if shared > 1 else ''}<br>"
                f"contribution = "
                f"<b>{_FRACTION_TEXT[shared]}</b>"
            )

        self.plotter.render()

    def select_atom(
        self,
        position,
    ):

        self.selected_position = position

        nn = find_nearest_neighbors(
            selected_index=fold_to_cell_index(
                position,
                self.fractional_atoms,
            ),
            fractional_atoms=(
                self.fractional_atoms
            ),
            a_angstrom=self.a,
            number_of_neighbors=4,
        )

        center = position * self.a

        neighbors = (
            center
            + nn["displacement_cartesian"]
        )

        self.draw_scene(reset_camera=False)

        add_neighbor_bonds(
            self.plotter,
            center,
            neighbors,
            self.a,
        )

        add_neighbor_atoms(
            self.plotter,
            neighbors,
            self.a,
        )

        self.atom_label.setText(
            "<b>Selected Ge atom</b><br>"
            f"r = "
            f"({position[0]:.2f}, "
            f"{position[1]:.2f}, "
            f"{position[2]:.2f}) a"
        )

        # Show the actual geometric derivation.
        lines = []

        for i, (
            displacement,
            distance,
        ) in enumerate(
            zip(
                nn[
                    "displacement_fractional"
                ],
                nn["distances"],
            )
        ):

            dx, dy, dz = displacement

            lines.append(
                f"NN{i + 1}: "
                f"Δr = "
                f"({dx:+.2f}, "
                f"{dy:+.2f}, "
                f"{dz:+.2f})a"
                f"<br>"
                f"d = {distance:.4f} Å"
            )

        self.nn_label.setText(
            "<b>Four nearest neighbors</b>"
            "<br><br>"
            + "<br><br>".join(lines)
            + "<br><br>"
            "All four distances are equal."
            "<br>"
            "<b>d<sub>NN</sub> = √3 a / 4</b>"
        )

        self.plotter.render()

    # ========================================================
    # Events
    # ========================================================

    def _refresh(self, reset_camera):
        """Redraw the current scene, keeping any selection."""

        if self.scene_mode == "homework":
            self._render_homework(reset_camera)
        else:
            self.draw_scene(reset_camera)
            self._update_cell_info()

        if (
            self.selected_position is not None
            and self._nn_pickable()
        ):
            self.select_atom(self.selected_position)

        if (
            self.picked_position is not None
            and self._is_counting_step()
        ):
            self.show_cell_contribution(self.picked_position)

    def on_view_changed(self):

        self.scene_mode = "structure"
        self.selected_position = None
        self.picked_position = None
        self._clear_neighbor_readout()
        self.draw_scene()
        self._update_cell_info()

    def on_repetition_changed(self, _value=None):

        self.selected_position = None
        self.picked_position = None
        self._clear_neighbor_readout()
        self._rebuild_supercell()
        self._refresh(reset_camera=True)

    def on_display_option_changed(self, _value=None):

        self._refresh(reset_camera=False)

    def on_roi_changed(self, _value=None):

        had_selection = (
            self.selected_position is not None
            or self.picked_position is not None
        )

        self._apply_roi()

        if had_selection and (
            self.selected_position is None
            and self.picked_position is None
        ):
            self._clear_neighbor_readout()

        self._refresh(reset_camera=False)

    def on_lattice_constant_changed(
        self,
        value,
    ):

        self.a = float(value)
        self._refresh(reset_camera=True)

    def reset_selection(self):

        self.selected_position = None
        self.picked_position = None
        self._clear_neighbor_readout()
        self._refresh(reset_camera=False)


# ============================================================
# Entry point
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )

    window = CrystalViewer()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
