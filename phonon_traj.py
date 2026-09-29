#!/usr/bin/env python

import sys, argparse
import ase
from ase.io import read, write
import numpy as np

from vaspvib import (load_vibmodes_from_outcar, participation_ratio,
                     is_localized, PlanckConstant, SpeedOfLight)

def msd_classical(w, T=300, m=1.0, freq_unit='cm-1'):
    """
    The classical mean square displacement (MSD) of a harmonic oscillator at
    temperature "T" with frequency "w" and mass "m".

       MSD_c:
           < x**2 > = k_B T / (M w**2)

    Inputs:
            w:  the frequency of the HO.
            m:  the mass of the HO in unified atomic mass unit.
            T:  the temperature in Kelvin.
    freq_unit:  the unit of the frequency.

    Return:
        MSD in Angstrom**2
    """
    T = np.asarray(T, dtype=float)

    if freq_unit.lower() == 'cm-1':
        # Change to angular frequency; 2 pi f, f in unit of Hz
        w = 2 * np.pi * (w * SpeedOfLight * 100)
    elif freq_unit.lower() == 'ev':
        # Change to angular frequency; 2 pi f, f in unit of Hz
        w = 2 * np.pi * (w / PlanckConstant)
    else:
        raise ValueError('Invalid unit of frequency!')

    return ase.units.kB * T * ase.units._e / (
                m * ase.units._amu * w**2 
            ) * ase.units.m**2

def msd_quantum(w, T=300, m=1.0, n=None, freq_unit='cm-1'):
    """
    The quantum mean square displacement (MSD) of a harmonic oscillator at
    temperature "T" with frequency "w" and mass "m".

       MSD_c:
                        hbar         hbar w
           < x**2 > = -------- coth(---------)
                       2 M w         2 k_B T

    Inputs:
            w:  the frequency of the HO.
            m:  the mass of the HO in unified atomic mass unit.
            T:  the temperature in Kelvin.
            n:  the number of phonons
    freq_unit:  the unit of the frequency.

    Return:
        MSD in Angstrom**2
    """
    CmToEv       = PlanckConstant * SpeedOfLight * 100

    if freq_unit.lower() == 'cm-1':
        HbarOmega = w * CmToEv 
        # Change to angular frequency; 2 pi f, f in unit of Hz
        w = 2 * np.pi * (w * SpeedOfLight * 100)
    elif freq_unit.lower() == 'ev':
        HbarOmega = w
        # Change to angular frequency; 2 pi f, f in unit of Hz
        w = 2 * np.pi * (w / PlanckConstant)
    else:
        raise ValueError('Invalid unit of frequency!')

    T = np.asarray(T, dtype=float)
    # the phonon population
    if n is None:
        n = np.zeros_like(T, dtype=float)
        n[T < 1E-12] = 0
        n[T > 1E-12] = 1. / (np.exp(HbarOmega / (ase.units.kB * T[T > 1E-12])) - 1.)

    return ase.units._hbar / (2 * m * ase.units._amu * w) * (1. + 2 * n) * ase.units.m**2

def phonon_traj(w, e, p0, q=0, temperature=300,
                dt=1.0, nsw=None, msd='quantum',
                linear_traj=False,
                nPhonon=None,
                saveMaxMin=True,
                scale_by_natoms=True,
                freq_unit='cm-1'):
    '''
    Generate the phonon animation. The relation between the atomic displacement
    and the phonon polarization vector (eigenvector of the dynamical matrix) can
    be found at:
        https://atztogo.github.io/phonopy/formulation.html#thermal-displacement

    Inputs:
        w: phonon vibration frequency
        e: phonon polarization vector, i.e. eigenvector of the dynamical matrix
        p0: the equilibrium configuration, an instance of ase.Atoms
        q: phonon momentum
        temperature: temperature in Kelvin
        dt: the time step in the ouput animation, unit [fs]
        nsw: total number of steps in the animation
        msd: the method to calculate mean-square displacement
        scale_by_natoms: scale the amplitude by sqrt(Natoms), see the NOTE
                         below; only appropriate for delocalized modes
        freq_unit: unit of the frequency
    '''
    M = p0.get_masses()
    Natoms = len(p0)

    if freq_unit.lower() == 'cm-1':
        T = 1E15 / (w * SpeedOfLight * 100)
    elif freq_unit.lower() == 'ev':
        T = 1E15 * PlanckConstant / w
    else:
        raise ValueError('Invalid unit of frequency!')

    if linear_traj:
        assert nsw is not None, "NSW can not be none!" 

    if nsw is None:
        nsw = int(T / dt) + 5
    else:
        nsw = nsw
        dt = T / nsw

    if msd.lower() == 'quantum':
        A = np.sqrt(2 * msd_quantum(w, T=temperature, m=M,
                    n=nPhonon,
                    freq_unit=freq_unit))
    elif msd.lower() == 'classical':
        A = np.sqrt(2 * msd_classical(w, T=temperature, m=M, freq_unit=freq_unit))
    else:
        A = 1.0 / np.sqrt(M)

    # NOTE: the sqrt(Natoms) factor below is a *convention*, not a physical
    # identity.  Read this before comparing amplitudes across supercells.
    #
    # Without the factor, the displacement u_i = A_i * e_i carries exactly
    # (n + 1/2) * hbar * w, i.e. ONE quantum for the WHOLE supercell.  Since a
    # delocalized mode has |e_i|^2 ~ 1/Natoms, the per-atom displacement would
    # then shrink as 1/sqrt(Natoms) and results obtained with different
    # supercell sizes are no longer comparable.
    #
    # Multiplying by sqrt(Natoms) assigns one quantum per ATOM, so that the
    # total energy is Natoms * (n + 1/2) * hbar * w and the per-atom
    # displacement becomes supercell-independent for DELOCALIZED modes.
    #
    # Two caveats:
    #   - It is WRONG for LOCALIZED modes (defect vibrations, adsorbed
    #     molecules), where |e_i|^2 ~ 1 on a few atoms regardless of Natoms.
    #     The displacement then grows as sqrt(Natoms).  Use "--no-scale".
    #   - For a zone-center mode folded from a primitive cell, the physically
    #     motivated factor is sqrt(N_cells), which differs from sqrt(Natoms)
    #     by a constant sqrt(atoms per primitive cell).  The supercell scaling
    #     is right either way, the absolute normalization is not.
    #
    # RELATION TO vib_proj.py: that script reports the STANDARD normal-mode
    # coordinate, Q = sum_i sqrt(M_i) e_i . u_i, with no Natoms factor of any
    # kind, because the energy sum rule forbids one there.  So with the
    # default scaling the Q.dat written below is LARGER by sqrt(Natoms) than
    # the coordinate vib_proj.py would assign to the same structure.  Pass
    # "--no-scale" when the two are meant to be compared directly.
    #
    # The participation ratio tells the delocalized and localized cases apart,
    # so warn rather than leave the caller to notice.  PR is computed on the
    # mass-weighted eigenvector, which is what "e" is here.
    pr = participation_ratio(e[np.newaxis, ...])[0]
    print("Participation ratio: {:.2f} of {:d} atoms (PR/Nions = {:.3f})".format(
        pr, Natoms, pr / Natoms))

    if scale_by_natoms:
        if is_localized(pr, Natoms):
            print("WARNING: this mode is LOCALIZED, so the default "
                  "sqrt(Natoms) amplitude scaling is not appropriate:\n"
                  "         it inflates the displacement by about "
                  "sqrt({:d}) = {:.1f} and will keep growing with the\n"
                  "         supercell size instead of converging. "
                  "Consider --no-scale.".format(Natoms, np.sqrt(Natoms)))
        A *= np.sqrt(Natoms)

    # NB: A_i is only the prefactor; the displacement of atom i is A_i * |e_i|,
    # which for a delocalized mode is smaller by roughly 1/sqrt(Natoms).
    chemical_symbols = p0.get_chemical_symbols()
    print("Amplitude prefactor A (displacement of atom i is A_i * |e_i|):")
    for elemnent in set(chemical_symbols):
        ind = chemical_symbols.index(elemnent)
        print("{:4s}: {:.4f} ang".format(elemnent, A[ind]))

    trajs = []
    pos0 = p0.positions.copy()
    ndigit = int(np.log10(nsw)) + 1
    fmt = 'traj_{{:0{}d}}.vasp'.format(ndigit)

    if saveMaxMin:
        pMax = pos0 + A[:, None] * e
        p0.set_positions(pMax)
        np.savetxt('maxD.dat', pMax, fmt='%22.16f')
        write('dmax.vasp', p0, vasp5=True, direct=True)

        pMin = pos0 - A[:, None] * e
        p0.set_positions(pMin)
        np.savetxt('minD.dat', pMin, fmt='%22.16f')
        write('dmin.vasp', p0, vasp5=True, direct=True)
    elif linear_traj:
        disp = np.linspace(-1, 1, nsw, endpoint=True)
        for ii in range(nsw):
            pos1 = pos0 + A[:, None] * e * disp[ii]
            p0.set_positions(pos1)
            write(fmt.format(ii + 1), p0, vasp5=True, direct=True)

        # normal mode coordinate
        Qmax = np.sum((np.sqrt(p0.get_masses()) * A)[:,None] * e**2)
        cc = 'Normal-mode Coordinate in "sqrt(amu) * Angstrom"\n'
        if scale_by_natoms:
            cc += 'Normalization: one quantum per ATOM, i.e. the amplitude A\n'
            cc += 'is scaled by sqrt(Natoms) and the total energy of the mode\n'
            cc += 'is Natoms * (n + 1/2) * hbar * w.'
        else:
            cc += 'Normalization: one quantum per SUPERCELL, i.e. the total\n'
            cc += 'energy of the mode is (n + 1/2) * hbar * w.'
        np.savetxt('Q.dat', Qmax*disp, fmt='%12.6f', header=cc)
    else:
        for ii in range(nsw):
            pos1 = pos0 + A[:, None] * e * np.sin(2 * np.pi * ii * dt / T)
            p0.set_positions(pos1)
            trajs.append(p0.copy())
            write(fmt.format(ii + 1), p0, vasp5=True, direct=True)

        write('traj.xyz', trajs, format='extxyz')

def parse_cml_args(cml):
    '''
    CML parser.
    '''
    arg = argparse.ArgumentParser(add_help=True)

    arg.add_argument('-i', dest='outcar', action='store', type=str,
                     default='OUTCAR',
                     help='Location of VASP OUTCAR.')
    arg.add_argument('-p', dest='poscar', action='store', type=str,
                     default='POSCAR',
                     help='Location of VASP POSCAR.')
    arg.add_argument('-m', dest='mode', action='store', type=int,
                     default=0,
                     help='Select the vibration mode, starting from 0.')
    arg.add_argument('-nph', dest='nPhonon', action='store', type=int,
                     default=None,
                     help='The phonon occupation of the selected mode. '
                          'NOTE: unless --no-scale is given, the amplitude is '
                          'scaled by sqrt(Natoms), so the total energy is '
                          'Natoms * (n + 1/2) * hbar * w, not (n + 1/2) * hbar * w.')
    arg.add_argument('-t', dest='temperature', action='store', type=float,
                     default=300,
                     help='The temperature.')
    arg.add_argument('-msd', dest='msd', action='store', type=str,
                     default='quantum', choices=['quantum', 'classical'],
                     help='Quantum or Classical harmonic oscillator.')
    arg.add_argument('-nsw', dest='nsw', action='store', type=int,
                     default=None,
                     help='The total number of steps in the phonon animation.')
    arg.add_argument('-dt', dest='dt', action='store', type=float,
                     default=1.0,
                     help='The time step [fs] used in the phonon animation.')
    arg.add_argument('--maxmin', dest='maxmin', action='store_true',
                     help='Whether to save the maximal/minimal displacement, default False.')
    arg.add_argument('--linear_traj', dest='linear_traj', action='store_true',
                     help='Linear interpolation betweewn maximal and minimal displacement.')
    arg.add_argument('--no-scale', dest='scale_natoms', action='store_false',
                     help='Do NOT scale the amplitude by sqrt(Natoms). The mode '
                          'then carries one quantum for the whole supercell. '
                          'Use this for LOCALIZED modes (defects, adsorbates), '
                          'where the default scaling diverges with supercell size.')

    return arg.parse_args(cml)

def main(cml):
    arg = parse_cml_args(cml)

    atoms = read(arg.poscar, format='vasp')
    omegas, modes, real_freq = load_vibmodes_from_outcar(arg.outcar)
    print("Generation phonon animation for mode {:d} with frequency {:8.4f} cm-1{}".format(
        arg.mode, omegas[arg.mode],
        '' if real_freq[arg.mode] else ' (IMAGINARY)'))

    phonon_traj(
            omegas[arg.mode], modes[arg.mode], atoms,
            temperature=arg.temperature,
            nsw=arg.nsw, dt=arg.dt,
            nPhonon=arg.nPhonon,
            msd=arg.msd,
            saveMaxMin=arg.maxmin,
            linear_traj=arg.linear_traj,
            scale_by_natoms=arg.scale_natoms,
    )
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
