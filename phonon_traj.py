#!/usr/bin/env python

import os, sys, argparse
import ase
from ase.io import read, write
import numpy as np

PlanckConstant = 4.13566733e-15         # [eV s]
SpeedOfLight = 299792458.               # [m/s]

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

def msd_quantum(w, T=300, m=1.0, freq_unit='cm-1'):
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
    freq_unit:  the unit of the frequency.

    Return:
        MSD in Angstrom**2
    """
    CmToEv       = PlanckConstant * SpeedOfLight * 100

    if freq_unit.lower() == 'cm-1':
        BetaHbarOmega = w * CmToEv / (ase.units.kB * T)
        # Change to angular frequency; 2 pi f, f in unit of Hz
        w = 2 * np.pi * (w * SpeedOfLight * 100)
    elif freq_unit.lower() == 'ev':
        BetaHbarOmega = w / (ase.units.kB * T)
        # Change to angular frequency; 2 pi f, f in unit of Hz
        w = 2 * np.pi * (w / PlanckConstant)
    else:
        raise ValueError('Invalid unit of frequency!')

    # the phonon population
    n = 1. / (np.exp(BetaHbarOmega) - 1.)

    return ase.units._hbar / (2 * m * ase.units._amu * w) * (1. + 2 * n) * ase.units.m**2

def load_vibmodes_from_outcar(inf='OUTCAR', exclude_imag=False):
    '''
    Read vibration eigenvectors and eigenvalues from OUTCAR.
    '''

    out = [line for line in open(inf) if line.strip()]
    ln = len(out)
    for line in out:
        if "NIONS =" in line:
            nions = int(line.split()[-1])
            break

    THz_index = []
    for ii in range(ln-1, 0, -1):
        if '2PiTHz' in out[ii]:
            THz_index.append(ii)
        if 'Eigenvectors and eigenvalues of the dynamical matrix' in out[ii]:
            i_index = ii + 2
            break
    j_index = THz_index[0] + nions + 2

    real_freq = [False if 'f/i' in line else True
                 for line in out[i_index:j_index]
                 if '2PiTHz' in line]

    # frequencies in unit of cm-1
    omegas = [line.split()[-4] for line in out[i_index:j_index]
              if '2PiTHz' in line]
    modes = [line.split()[3:6] for line in out[i_index:j_index]
             if ('dx' not in line) and ('2PiTHz' not in line)]

    omegas = np.array(omegas, dtype=float)
    modes = np.array(modes, dtype=float).reshape((-1, nions, 3))

    if exclude_imag:
        omegas = omegas[real_freq]
        modes = modes[real_freq]

    return omegas, modes

def phonon_traj(w, e, p0, q=0, temperature=300,
                dt=1.0, nsw=None, msd='quantum', freq_unit='cm-1'):
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
        freq_unit: unit of the frequency
    '''
    M = p0.get_masses()

    if freq_unit.lower() == 'cm-1':
        T = 1E15 / (w * SpeedOfLight * 100)
    elif freq_unit.lower() == 'ev':
        T = 1E15 * PlanckConstant / w
    else:
        raise ValueError('Invalid unit of frequency!')

    if nsw is None:
        nsw = int(T / dt) + 5
    else:
        nsw = nsw
        dt = T / nsw

    if msd.lower() == 'quantum':
        A = np.sqrt(msd_quantum(w, T=temperature, m=M, freq_unit=freq_unit))
    elif msd.lower() == 'classical':
        A = np.sqrt(msd_classical(w, T=temperature, m=M, freq_unit=freq_unit))
    else:
        A = 1.0 / np.sqrt(M)

    chemical_symbols = p0.get_chemical_symbols()
    for elemnent in set(chemical_symbols):
        ind = chemical_symbols.index(elemnent)
        print("{:4s}: {:.4f} ang".format(elemnent, A[ind]))

    trajs = []
    pos0 = p0.positions.copy()
    ndigit = int(np.log10(nsw)) + 1
    fmt = 'traj_{{:0{}d}}.vasp'.format(ndigit)

    for ii in range(nsw):
        pos1 = pos0 + A[:, None] * e * np.sin(2 * np.pi * ii / nsw)
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
                     help='Select the vibration mode.')
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

    return arg.parse_args(cml)

def main(cml):
    arg = parse_cml_args(cml)

    atoms = read(arg.poscar, format='vasp')
    omegas, modes = load_vibmodes_from_outcar(arg.outcar)
    print("Generation phonon animation for mode {:d} with frequency {:8.4f} cm-1".format(arg.mode, omegas[arg.mode]))

    phonon_traj(
            omegas[arg.mode], modes[arg.mode], atoms,
            temperature=arg.temperature,
            nsw=arg.nsw, dt=arg.dt,
            msd=arg.msd
    )
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
