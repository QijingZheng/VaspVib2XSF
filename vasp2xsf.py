#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from ase.io import read

from vaspvib import (load_vibmodes_from_outcar, participation_ratio,
                     is_localized)

CACHE = 'MODES.npz'


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
                     default=None,
                     help='Select the vibration mode, STARTING FROM 0 (same '
                          'convention as phonon_traj.py). Default: all modes. '
                          'The output file is named after the 1-based mode '
                          'number printed by VASP, so "-m 0" writes '
                          'mode_0001.xsf.')
    arg.add_argument('-s', dest='scale', action='store', type=float,
                     default=1.0,
                     help='Scale factor of the vector field.')
    arg.add_argument('--no-cache', dest='cache', action='store_false',
                     help='Do not read or write {}.'.format(CACHE))

    return arg.parse_args(cml)


def fingerprint(outcar, poscar):
    '''
    Identify the inputs a cache file was built from, so that a stale cache
    left over from another system is never silently reused.
    '''
    return np.array([
        os.path.abspath(outcar), '{:.6f}'.format(os.path.getmtime(outcar)),
        str(os.path.getsize(outcar)),
        os.path.abspath(poscar), '{:.6f}'.format(os.path.getmtime(poscar)),
    ], dtype='U')


def load_displacements(outcar, poscar, masses, use_cache=True):
    '''
    Return the DISPLACEMENT vectors (dynamical-matrix eigenvectors divided by
    sqrt(mass)) together with the frequencies and the participation ratios.
    '''
    fp = fingerprint(outcar, poscar)

    if use_cache and os.path.isfile(CACHE):
        try:
            z = np.load(CACHE, allow_pickle=False)
            if np.array_equal(z['fingerprint'], fp):
                return z['omegas'], z['modes'], z['real_freq'], z['pr']
            print('{} was built from different inputs, re-reading {}.'.format(
                CACHE, outcar))
        except (OSError, ValueError, KeyError):
            print('{} is unreadable, re-reading {}.'.format(CACHE, outcar))

    omegas, modes, real_freq = load_vibmodes_from_outcar(outcar)
    if modes.shape[1] != len(masses):
        raise ValueError(
            'OUTCAR has {} ions but {} has {}!'.format(
                modes.shape[1], poscar, len(masses)))

    # NB: the participation ratio must be evaluated on the MASS-WEIGHTED
    # eigenvectors, i.e. BEFORE the division below.  PR is invariant under an
    # overall rescaling of a mode, but NOT under the per-atom rescaling by
    # 1/sqrt(M_i), which is a different (displacement-based) definition.
    pr = participation_ratio(modes)

    # Eigenvectors after division by SQRT(mass): displacement vector.
    modes = modes / np.sqrt(np.asarray(masses)[None, :, None])

    if use_cache:
        np.savez(CACHE, omegas=omegas, modes=modes, real_freq=real_freq,
                 pr=pr, fingerprint=fp)

    return omegas, modes, real_freq, pr


def write_xsf(imode, atoms, vector, scale=1.0):
    """
    Write the position and vector field in XSF format.
    """

    vector = np.asarray(vector, dtype=float) * scale
    assert vector.shape == atoms.positions.shape
    pos_vec = np.hstack((atoms.positions, vector))
    nions = pos_vec.shape[0]
    chem_symbs = atoms.get_chemical_symbols()
    with open('mode_{:04d}.xsf'.format(imode), 'w') as out:
        line = "CRYSTAL\n"
        line += "PRIMVEC\n"
        line += '\n'.join([
            ' '.join(['%21.16f' % a for a in vec])
            for vec in atoms.cell
        ])
        line += "\nPRIMCOORD\n"
        line += "{:3d} {:d}\n".format(nions, 1)
        line += '\n'.join([
            '{:3s}'.format(chem_symbs[ii]) +
            ' '.join(['%21.16f' % a for a in pos_vec[ii]])
            for ii in range(nions)
        ])

        out.write(line)


def main(cml):
    p = parse_cml_args(cml)

    atoms = read(p.poscar, format='vasp')
    omegas, modes, real_freq, pr = load_displacements(
        p.outcar, p.poscar, atoms.get_masses(), use_cache=p.cache)

    n_mode = len(omegas)
    nions = len(atoms)
    if p.mode is None:
        selected = range(n_mode)
    else:
        if not 0 <= p.mode < n_mode:
            raise SystemExit(
                'Mode index {} out of range; OUTCAR has {} modes, '
                'indexed 0 to {}.'.format(p.mode, n_mode, n_mode - 1))
        selected = [p.mode]

    loc = is_localized(pr, nions)

    print('{:<15s} {:>12s} {:>8s} {:>9s}  {}'.format(
        'file', 'freq [cm-1]', 'PR', 'PR/Nions', 'flags'))
    for ii in selected:
        # the file name follows the 1-based mode number printed by VASP
        write_xsf(ii + 1, atoms, modes[ii], p.scale)
        flags = ' '.join(f for f, on in
                         (('IMAGINARY', not real_freq[ii]), ('LOCALIZED', loc[ii]))
                         if on)
        print('{:<15s} {:12.4f} {:8.2f} {:9.3f}  {}'.format(
            'mode_{:04d}.xsf'.format(ii + 1), omegas[ii],
            pr[ii], pr[ii] / nions, flags))


if __name__ == "__main__":
    main(sys.argv[1:])
