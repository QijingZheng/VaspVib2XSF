#!/usr/bin/env python

import numpy as np
import os, ase
from ase.io import read, write
from ase import Atoms

############################################################

def posFromXdatcar(inf='XDATCAR', direct=True, return_vel=False, dt=1.0):
    '''
    extract coordinates from XDATCAR and create POSCAR files.
    
    Input arguments:
        inf:        location of XDATCAR
        direct:     coordinates in fractional or cartesian 
        return_vel: whether to calculate velocity or not
        dt:         time step of MD
    '''

    inp = open(inf).readlines()

    ElementNames = inp[5].split()
    ElementNumbers = np.array([int(x) for x in inp[6].split()], dtype=int)
    cell = np.array([line.split() for line in inp[2:5]], dtype=float)
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
        vel = np.diff(positions, axis=0) / dt
        # periodic boundary condition.
        vel[vel > 0.5] -= 1.0
        vel[vel <-0.5] += 1.0
        vel = np.dot(vel, cell)
    else:
        vel = None

    return geo, pos, vel

def load_vibmodes_from_outcar(inf='OUTCAR', include_imag=False):
    '''
    Read vibration eigenvectors and eigenvalues from OUTCAR. The frequencies are
    returned in units of cm^-1.
    '''

    out = [line for line in open(inf) if line.strip()]
    ln  = len(out)
    for line in out:
        if "NIONS =" in line:
            nions = int(line.split()[-1])
            break

    THz_index = []
    for ii in range(ln-1,0,-1):
        if '2PiTHz' in out[ii]:
            THz_index.append(ii)
        if 'Eigenvectors and eigenvalues' in out[ii]:
            i_index = ii + 2
            break
    j_index = THz_index[0] + nions + 2

    real_freq = [False if 'f/i' in line else True
                 for line in out[i_index:j_index] 
                 if '2PiTHz' in line]

    omegas = [line.split()[ -4] for line in out[i_index:j_index]
              if '2PiTHz' in line]
    modes  = [line.split()[3:6] for line in out[i_index:j_index]
              if ('dx' not in line) and ('2PiTHz' not in line)]

    omegas = np.array(omegas, dtype=float)
    modes  = np.array(modes,  dtype=float).reshape((-1, nions, 3))

    if not include_imag:
        omegas = omegas[real_freq]
        modes  = modes[real_freq]

    return omegas, modes
   
def velocity_drift(x, a, b):
    return a * x + b

############################################################
if __name__ == '__main__':
    # time step used in MD, in femtosecond
    dt = 1.0 
    if not os.path.isfile('E_n.npy'):
        # load the trajectory from an NVE MD
        geo, pa, va = posFromXdatcar('XDATCAR', direct=True,
                                     return_vel=False, dt=1.0)
        M     = geo.get_masses()
        niter = pa.shape[0]
        natom = pa.shape[1]

        # the equilibrium POSCAR
        # p0 = read('POSCAR_eq', format='vasp').get_scaled_positions()
        # p0 = read('knew.vasp').get_scaled_positions()
        geo0 = read('POSCAR')
        p0   = geo0.get_scaled_positions()
        assert np.allclose(M, geo0.get_masses())

        # read the vibration modes from OUTCAR
        w, v = load_vibmodes_from_outcar('OUTCAR')

        # ########################################
        # # subtract the drift in the x and y direction
        # ########################################
        from scipy.optimize import curve_fit
        T = np.arange(niter)
        # drift in x direction
        val_x, err = curve_fit(velocity_drift, T, pa[:,0,0])
        # drift in y direction
        val_y, err = curve_fit(velocity_drift, T, pa[:,0,1])
        pa[:,:,0] -= (T * val_x[0])[:,np.newaxis]
        pa[:,:,1] -= (T * val_y[0])[:,np.newaxis]
        # periodic boundary condition.
        pa[pa > 1.0] -= 1.0
        pa[pa < 0.0] += 1.0
        # np.save('pa.npy', pa)
        # ########################################

        # deviation from the equilibrium positions
        pd = pa - p0[np.newaxis,...]
        # periodic boundary condition.
        pd[pd >=0.5] -= 1.0
        pd[pd <-0.5] += 1.0
        pd = np.dot(pd, geo.cell).reshape((-1, natom * 3))

        # phonon polarization vector multiply the the mass square root
        v_sqrtM = np.sqrt(M[None,:,None]) * v
        # normal mode coordinates
        nc = np.dot(pd, v_sqrtM.reshape((-1, natom * 3)).T) / np.sqrt(natom)
        vc = np.diff(nc, axis=0) / dt

        # save the frequencies to file, in unit of cm^-1
        np.save('omega.npy', w)
        # change vibration frequencies unit to eV, unit conversion values are those used
        # in phonopy
        THzToCm = 33.3564095198152
        CmToEv  = 0.00012398418743309975
        # to 2PiTHz
        w0 = w / THzToCm * 2 * np.pi

        # The energy of each normal mode, according to the "10.1038/ncomms11504" is then
        # calculated by the following formula
        # E_n = 0.5 * ((d nc / dt)**2 + w0**2 * nc**2)
        # E1 = vc**2 * 1E6
        # E2 = w0[np.newaxis,...]**2 * nc[:-1,...]**2
        # En = 0.5 * (E1 + E2) * THzToCm * CmToEv

        E1 = 0.5 * (vc * 1E-10 / 1E-15 * np.sqrt(ase.units._mp))**2
        E2 = 0.5 * (nc[:-1,...] * 1E-10 * np.sqrt(ase.units._mp))**2 * (w[None,:] / THzToCm * 1E12 * 2 * np.pi)**2
        En = (E1 + E2) / ase.units._e

        # np.save('E1.npy', E1)
        # np.save('E2.npy', E2)
        np.save('E_n.npy', En)
    else:
        En = np.load('E_n.npy')
        w  = np.load('omega.npy')

    ############################################################
    import matplotlib as mpl
    # mpl.use('agg')
    mpl.rcParams['axes.unicode_minus'] = False

    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=300)
    fig.set_size_inches(4.0, 2.5)
    ax      = plt.subplot()

    Ntime = En.shape[0]
    Nmode = En.shape[1]
    ModeI = np.arange(Nmode) + 1
    T     = 60
    Nmax  = 5
    ########################################
    # energy_of_mode = En[T-1]
    energy_of_mode = np.average(En, axis=0)
    loc_max_peak   = np.argsort(energy_of_mode)[-Nmax:]

    # ax.vlines(ModeI, ymin=0.0,
    ax.vlines(w, ymin=0.0,
              ymax=energy_of_mode,
              lw=1.0, color='k')

    for pk in loc_max_peak:
        # ax.text(ModeI[pk], energy_of_mode[pk] * 1.01, 'N=%d' % ModeI[pk],
        ax.text(w[pk], energy_of_mode[pk] * 1.01, 'N=%d' % ModeI[pk],
                ha='center', va='bottom',
                fontsize='x-small',
                color='red',
                )

    # ax.set_xlim(0, nsw)
    # ax.set_ylim(0.0, 8.0)

    # ax.set_xlabel('Mode Number',   fontsize='small', labelpad=5)
    ax.set_xlabel('Wavenumber [cm$^{-1}$]', fontsize='small', labelpad=5)
    ax.set_ylabel('Energy [eV]', fontsize='small', labelpad=8)
    ax.tick_params(which='both', labelsize='x-small')

    ########################################
    plt.tight_layout(pad=0.2)
    plt.savefig('kaka.png', dpi=360)
    plt.show()
