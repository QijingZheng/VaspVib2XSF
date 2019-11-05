#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from ase.io import read


def load_vibmodes_from_outcar(inf='OUTCAR', include_imag=False):
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

    omegas = [line.split()[-4] for line in out[i_index:j_index]
              if '2PiTHz' in line]
    modes = [line.split()[3:6] for line in out[i_index:j_index]
             if ('dx' not in line) and ('2PiTHz' not in line)]

    omegas = np.array(omegas)
    modes = np.array(modes).reshape((-1, nions, 3))

    if include_imag:
        omegas = omegas[real_freq]
        modes = modes[real_freq]

    return omegas, modes


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
                     help='Select the vibration mode, 0 for all modes.')
    arg.add_argument('-s', dest='scale', action='store', type=float,
                     default=1.0,
                     help='Scale factor of the vector field.')

    return arg.parse_args(cml)


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
    if not os.path.isfile('MODES.npy'):
        omegas, modes = load_vibmodes_from_outcar(p.outcar)
        # Eigenvectors after division by SQRT(mass): displacement vector.
        modes /= np.sqrt(atoms.get_masses()[None, :, None])
        np.save('OMEGAS', omegas)
        np.save('MODES', modes)
    else:
        omegas = np.load('OMEGAS.npy')
        modes = np.load('MODES.npy')

    n_mode = len(omegas)
    assert 0 <= p.mode <= n_mode
    if p.mode == 0:
        for ii in range(n_mode):
            write_xsf(ii+1, atoms, modes[ii], p.scale)
    else:
        write_xsf(p.mode, atoms, modes[p.mode - 1], p.scale)


if __name__ == "__main__":
    main(sys.argv[1:])
