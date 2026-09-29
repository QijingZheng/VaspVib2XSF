#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import ase
from ase.io import read
from ase import Atoms

from vaspvib import (load_vibmodes_from_outcar, participation_ratio,
                     cm2rad_per_s, is_localized, LOCALIZATION_THRESHOLD)

CACHE = 'mode_energies.npz'

############################################################


def posFromXdatcar(inf='XDATCAR', direct=True, return_vel=False, dt=1.0):
    '''
    extract coordinates from XDATCAR and create POSCAR files.

    Input arguments:
        inf:        location of XDATCAR
        direct:     coordinates in fractional or cartesian
        return_vel: whether to calculate velocity or not
        dt:         time step of MD, in femtosecond

    Returns:
        geo: an ase.Atoms holding the chemical formula and the cell
        pos: (niter, natom, 3) coordinates, fractional if "direct"
        vel: (niter-1, natom, 3) velocities in Angstrom/fs, or None
    '''

    inp = open(inf).readlines()

    ElementNames = inp[5].split()
    ElementNumbers = np.array([int(x) for x in inp[6].split()], dtype=int)
    cell = np.array([line.split() for line in inp[2:5]], dtype=float)
    cell *= float(inp[1].split()[0])
    Natoms = ElementNumbers.sum()

    # construct ASE atoms object without positions.
    ChemFormula = ''.join(['%s%d' % (xx, nn) for xx, nn in zip(ElementNames, ElementNumbers)])
    geo = Atoms(ChemFormula, positions=np.zeros((Natoms, 3)), cell=cell, pbc=True)

    # the coordinates of atoms at each time step
    positions = [line.split() for line in inp[8:] if line.strip() and 'config'
                 not in line]
    positions = np.array(positions, dtype=float).reshape((-1, Natoms, 3))
    if not direct:
        pos = np.dot(positions, cell)
    else:
        pos = positions

    # the velocity of atoms at each time step
    if return_vel:
        # a simple way to estimate the velocities from positons.
        #
        # NB: the minimum-image wrap MUST be applied to the fractional
        # displacement itself, BEFORE dividing by dt.  Wrapping after the
        # division (as this function used to do) compares a quantity in
        # "fractional per fs" against the threshold 0.5 and is therefore only
        # correct when dt happens to be 1.0.
        dpos = np.diff(positions, axis=0)
        dpos -= np.round(dpos)
        vel = np.dot(dpos, cell) / dt
    else:
        vel = None

    return geo, pos, vel


def normal_mode_energies(xdatcar='XDATCAR', poscar='POSCAR', outcar='OUTCAR',
                         dt=1.0, remove_drift=True, exclude_imag=False):
    '''
    Decompose an MD trajectory into normal-mode coordinates and report the
    energy carried by each mode.

        Q_v = sum_i sqrt(M_i) e_vi . u_i          [sqrt(amu) * Angstrom]
        E_v = 0.5 * (dQ_v/dt)**2 + 0.5 * w_v**2 * Q_v**2

    NB on normalization: Q_v MUST NOT be rescaled by any power of the number
    of atoms.  The eigenvectors are complete and orthonormal, so

        sum_v 0.5 * (dQ_v/dt)**2  ==  0.5 * sum_i M_i |du_i/dt|**2,

    i.e. the mode energies sum to the true energy of the system, and classical
    equipartition gives <E_v> = k_B T per mode.  Dividing Q_v by sqrt(natom)
    (as this script used to do) breaks that identity and makes every mode
    energy too small by exactly a factor of natom.

    Inputs:
        xdatcar, poscar, outcar: input files. POSCAR is the equilibrium
                    geometry, OUTCAR supplies the vibration eigenvectors.
        dt:         MD time step in femtosecond
        remove_drift: subtract the mass-weighted centre-of-mass displacement
                    of every frame.  This removes both the overall drift of an
                    NVE run and the three spurious acoustic translations.
        exclude_imag: drop the imaginary-frequency modes

    Returns:
        w:  (nmodes,) frequencies in cm^-1
        En: (niter-1, nmodes) mode energies in eV
        real_freq: (nmodes,) bool, False for imaginary modes
        pr: (nmodes,) participation ratio of each mode
        natom: the number of atoms, so that pr can be normalized later
    '''

    geo, pa, _ = posFromXdatcar(xdatcar, direct=True)
    M = geo.get_masses()
    natom = pa.shape[1]

    # the equilibrium geometry
    geo0 = read(poscar, format='vasp')
    assert np.allclose(M, geo0.get_masses()), \
        'POSCAR and XDATCAR describe different systems!'
    p0 = geo0.get_scaled_positions()

    # read the vibration modes from OUTCAR
    w, v, real_freq = load_vibmodes_from_outcar(outcar, exclude_imag=exclude_imag)
    assert v.shape[1] == natom, 'OUTCAR and XDATCAR disagree on the atom count!'
    if not real_freq.all():
        print('WARNING: {} imaginary mode(s) included; the harmonic energy of '
              'those modes is not meaningful. Use --exclude-imag to drop '
              'them.'.format((~real_freq).sum()))

    # deviation from the equilibrium positions, minimum image convention.
    #
    # NB: np.round handles a displacement of any size.  Adding or subtracting
    # 1.0 once (as this script used to do) is not enough once the drift
    # correction has pushed a coordinate beyond the neighbouring cell.
    pd = pa - p0[np.newaxis, ...]
    pd -= np.round(pd)
    # to Cartesian, in Angstrom
    pd = np.dot(pd, geo.cell)

    if remove_drift:
        # Subtract the mass-weighted centre of mass of every frame.
        #
        # NB: the drift used to be estimated by fitting a straight line to the
        # x and y coordinates of ATOM 0 only.  That mixes the thermal motion
        # of one particular atom into the correction and leaves z untouched.
        com = np.einsum('n,tnx->tx', M, pd) / M.sum()
        pd -= com[:, np.newaxis, :]

    # phonon polarization vector multiplied by the mass square root
    v_sqrtM = np.sqrt(M[None, :, None]) * v
    # normal mode coordinates, in sqrt(amu) * Angstrom
    nc = np.einsum('tnx,vnx->tv', pd, v_sqrtM)
    # normal mode velocities, in sqrt(amu) * Angstrom / fs
    vc = np.diff(nc, axis=0) / dt

    # angular frequency in rad/s, via the same conversion phonon_traj.py uses
    w_rad = cm2rad_per_s(w)

    # NB: the mass unit of ASE atomic masses is the unified atomic mass unit,
    # so the conversion factor is ase.units._amu.  Using ase.units._mp (the
    # PROTON mass), as this script used to do, overestimates every energy by
    # a factor 1.007276.
    E1 = 0.5 * (vc * 1E-10 / 1E-15 * np.sqrt(ase.units._amu))**2
    E2 = 0.5 * (nc[:-1, ...] * 1E-10 * np.sqrt(ase.units._amu))**2 * w_rad[None, :]**2
    En = (E1 + E2) / ase.units._e

    # computed on the mass-weighted eigenvectors, the same convention
    # phonon_traj.py and vasp2xsf.py use
    pr = participation_ratio(v)

    return w, En, real_freq, pr, natom


def plot_mode_energies(w, En, real_freq, loc=None, nmax=5,
                       figname='kaka.png', show=True):
    '''
    Stick plot of the time-averaged energy of each normal mode.

    A label reads "N=<mode>", with "i" appended for an imaginary mode and "*"
    for one the participation ratio flags as localized.
    '''
    import matplotlib as mpl
    if not show:
        mpl.use('agg')
    mpl.rcParams['axes.unicode_minus'] = False

    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=300)
    fig.set_size_inches(4.0, 2.5)
    ax = plt.subplot()

    Nmode = En.shape[1]
    ModeI = np.arange(Nmode) + 1
    ########################################
    energy_of_mode = np.average(En, axis=0)
    loc_max_peak = np.argsort(energy_of_mode)[-nmax:]

    ax.vlines(w, ymin=0.0,
              ymax=energy_of_mode,
              lw=1.0, color='k')

    for pk in loc_max_peak:
        ax.text(w[pk], energy_of_mode[pk] * 1.01,
                'N=%d%s%s' % (ModeI[pk],
                              '' if real_freq[pk] else 'i',
                              '*' if loc is not None and loc[pk] else ''),
                ha='center', va='bottom',
                fontsize='x-small',
                color='red',
                )

    ax.set_xlabel('Wavenumber [cm$^{-1}$]', fontsize='small', labelpad=5)
    ax.set_ylabel('Energy [eV]', fontsize='small', labelpad=8)
    ax.tick_params(which='both', labelsize='x-small')

    ########################################
    plt.tight_layout(pad=0.2)
    plt.savefig(figname, dpi=360)
    if show:
        plt.show()
    plt.close(fig)


def parse_cml_args(cml):
    '''
    CML parser.
    '''
    arg = argparse.ArgumentParser(add_help=True)

    arg.add_argument('-x', dest='xdatcar', action='store', type=str,
                     default='XDATCAR',
                     help='Location of the VASP XDATCAR of an NVE run.')
    arg.add_argument('-p', dest='poscar', action='store', type=str,
                     default='POSCAR',
                     help='Location of the equilibrium VASP POSCAR.')
    arg.add_argument('-i', dest='outcar', action='store', type=str,
                     default='OUTCAR',
                     help='Location of the VASP OUTCAR with the vibration modes.')
    arg.add_argument('-dt', dest='dt', action='store', type=float,
                     default=1.0,
                     help='The MD time step [fs] used in the XDATCAR.')
    arg.add_argument('--no-drift', dest='remove_drift', action='store_false',
                     help='Do NOT subtract the mass-weighted centre-of-mass '
                          'displacement of each frame.')
    arg.add_argument('--exclude-imag', dest='exclude_imag', action='store_true',
                     help='Drop the imaginary-frequency modes. NOTE: this '
                          'shifts the mode indices relative to the OUTCAR '
                          'listing.')
    arg.add_argument('--nmax', dest='nmax', action='store', type=int,
                     default=5,
                     help='Label the NMAX modes with the highest energy.')
    arg.add_argument('-o', dest='figname', action='store', type=str,
                     default='kaka.png',
                     help='Name of the output figure.')
    arg.add_argument('--no-show', dest='show', action='store_false',
                     help='Save the figure without opening a window.')
    arg.add_argument('--reuse', dest='reuse', action='store_true',
                     help='Reuse E_n.npy/omega.npy from a previous run instead '
                          'of re-reading the trajectory. Off by default: the '
                          'cache carries no record of which inputs produced it.')

    return arg.parse_args(cml)


def main(cml):
    p = parse_cml_args(cml)

    if p.reuse and os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        w, En, real_freq = z['omegas'], z['En'], z['real_freq']
        pr, nions = z['pr'], int(z['nions'])
    else:
        if p.reuse:
            print('{} not found, re-reading the trajectory.'.format(CACHE))
        w, En, real_freq, pr, nions = normal_mode_energies(
            xdatcar=p.xdatcar, poscar=p.poscar, outcar=p.outcar,
            dt=p.dt, remove_drift=p.remove_drift,
            exclude_imag=p.exclude_imag,
        )
        # frequencies in cm^-1, energies in eV
        np.savez(CACHE, omegas=w, En=En, real_freq=real_freq,
                 pr=pr, nions=nions)

    loc = is_localized(pr, nions)

    print('Total energy of all modes, time averaged: {:.6f} eV'.format(
        np.average(np.sum(En, axis=1))))
    print('Mean energy per mode:                     {:.6f} eV  '
          '(k_B T = 0.0259 eV at 300 K)'.format(np.average(En)))
    print('Localized modes (PR/Nions < {:.2f}):        {:d} of {:d}'.format(
        LOCALIZATION_THRESHOLD, int(loc.sum()), len(w)))

    plot_mode_energies(w, En, real_freq, loc=loc, nmax=p.nmax,
                       figname=p.figname, show=p.show)


############################################################
if __name__ == '__main__':
    main(sys.argv[1:])
