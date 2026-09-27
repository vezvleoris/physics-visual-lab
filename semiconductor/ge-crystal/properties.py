# properties.py

"""
단위변환 + 물리량 계산
"""
"""
Physical properties of Ge derived from the lattice constant.
"""

import numpy as np


ANGSTROM_TO_CM = 1e-8
AVOGADRO = 6.02214076e23

# Effective number of atoms in one conventional
# diamond cubic unit cell.
ATOMS_PER_CELL = 8


def nearest_neighbor_distance(a_angstrom):
    """
    d_NN = sqrt(3) a / 4
    """
    return (
        np.sqrt(3.0)
        * a_angstrom
        / 4.0
    )


def number_density(a_angstrom):
    """
    Ge number density in atoms/cm^3.

        n_Ge = 8 / a^3
    """
    a_cm = (
        a_angstrom
        * ANGSTROM_TO_CM
    )

    return (
        ATOMS_PER_CELL
        / a_cm**3
    )


def mass_density(
    a_angstrom,
    molar_mass_g_mol,
):
    """
    Ge mass density in g/cm^3.

        rho = n_Ge * M / N_A
    """
    n_ge = number_density(
        a_angstrom
    )

    mass_per_atom = (
        molar_mass_g_mol
        / AVOGADRO
    )

    return (
        n_ge
        * mass_per_atom
    )


def valence_electron_density(
    a_angstrom,
    valence_electrons,
):
    """
    Valence-electron density in electrons/cm^3.

        n_e = Z_valence * n_Ge
    """
    return (
        valence_electrons
        * number_density(a_angstrom)
    )