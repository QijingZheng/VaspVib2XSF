#!/usr/bin/env python
'''
Shared helpers for reading VASP vibration / phonon data.

This module exists so that phonon_traj.py, vasp2xsf.py and vib_proj.py parse
OUTCAR in exactly the same way.  Each script used to carry its own copy of
load_vibmodes_from_outcar() and the copies had drifted apart: the flag that
controls imaginary modes had different names AND opposite defaults, so the
same OUTCAR yielded different mode indices depending on which script read it.
'''

import warnings

import numpy as np

PlanckConstant = 4.13566733e-15         # [eV s]
SpeedOfLight = 299792458.               # [m/s]

# phonopy's value; 1 THz = 33.3564... cm-1.  Note that
#     1E12 / THzToCm  ==  SpeedOfLight * 100,
# so the two conversions below are the same one written two ways.  They used
# to live in phonon_traj.py and vib_proj.py separately.
THzToCm = 33.3564095198152
CmToEv = PlanckConstant * SpeedOfLight * 100


def cm2rad_per_s(w_cm):
    '''
    Wavenumber [cm^-1] to angular frequency [rad/s].
    '''
    return 2 * np.pi * (np.asarray(w_cm, dtype=float) * SpeedOfLight * 100)


def load_vibmodes_from_outcar(inf='OUTCAR', exclude_imag=False):
    '''
    Read vibration eigenvectors and eigenvalues from OUTCAR.

    Inputs:
        inf: location of the VASP OUTCAR.
        exclude_imag: drop the modes with imaginary frequencies ("f/i" in
                      OUTCAR).

                      THE DEFAULT IS FALSE, i.e. imaginary modes are KEPT, so
                      that the index of a mode in the returned arrays matches
                      its position in the OUTCAR listing.  Set it to True only
                      when that correspondence does not matter, and be aware
                      that the mode indices then shift.

    Returns:
        omegas: (nmodes,) frequencies in cm^-1.  An imaginary frequency is
                returned as its modulus; use "real_freq" to tell them apart.
        modes:  (nmodes, nions, 3) eigenvectors of the dynamical matrix,
                normalized as sum_i |e_i|^2 = 1.  These are MASS-WEIGHTED
                vectors; divide by sqrt(M_i) to get displacement vectors.
        real_freq: (nmodes,) bool array, False for the imaginary ("f/i") modes.
    '''

    out = [line for line in open(inf) if line.strip()]
    ln = len(out)

    nions = None
    for line in out:
        if "NIONS =" in line:
            nions = int(line.split()[-1])
            break
    if nions is None:
        raise ValueError('"NIONS =" not found in {}!'.format(inf))

    THz_index = []
    i_index = None
    for ii in range(ln-1, 0, -1):
        if '2PiTHz' in out[ii]:
            THz_index.append(ii)
        if 'Eigenvectors and eigenvalues of the dynamical matrix' in out[ii]:
            i_index = ii + 2
            break
    if i_index is None or not THz_index:
        raise ValueError(
            'No vibration eigenvectors found in {}! '
            'Did the calculation finish with IBRION = 5/6/7/8?'.format(inf)
        )
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
    real_freq = np.array(real_freq, dtype=bool)

    if exclude_imag:
        omegas = omegas[real_freq]
        modes = modes[real_freq]
        real_freq = real_freq[real_freq]

    return omegas, modes, real_freq


# A mode is flagged as "localized" when its participation ratio falls below
# this fraction of the total number of atoms.  Purely a heuristic for warnings.
LOCALIZATION_THRESHOLD = 0.2


def participation_ratio(modes, masses=None):
    '''
    Participation ratio (PR) of each vibration mode: roughly "how many atoms
    actually take part in this mode".

        p_i = |e_i|^2 / sum_j |e_j|^2          (grouped per ATOM, not per
                                                Cartesian degree of freedom)
        IPR = sum_i p_i^2
        PR  = 1 / IPR

    PR ranges from 1 (the whole mode sits on a single atom) to nions (every
    atom moves equally).  The normalized value PR / nions is the useful
    dimensionless measure: it stays O(1) for a delocalized mode as the
    supercell grows, and falls off as 1/nions for a localized one.

    This is what decides whether the sqrt(Natoms) amplitude scaling in
    phonon_traj.py is appropriate.  That scaling assumes |e_i|^2 ~ 1/nions,
    which holds only for a delocalized mode; applied to a localized mode it
    inflates the displacement without bound as the supercell grows.

    Inputs:
        modes:  (nmodes, nions, 3) eigenvectors, as returned by
                load_vibmodes_from_outcar().
        masses: if given, (nions,) atomic masses, and the PR is computed for
                the DISPLACEMENT pattern e_i / sqrt(M_i) rather than for the
                mass-weighted eigenvector itself.  The displacement-based PR
                gives exactly nions for a rigid translation whatever the mass
                distribution, whereas the mass-weighted PR is biased downward
                when the masses differ a lot.  The mass-weighted form (the
                default) is the one used in the phonon-localization
                literature, and either choice scales the same way with the
                supercell size.

    Returns:
        pr: (nmodes,) participation ratio, in the range [1, nions].

    NB: within a degenerate set of modes the individual eigenvectors are
    basis-dependent, so their individual PRs are not well defined.  Only the
    degenerate subspace as a whole is meaningful there.

    A mode whose vector is identically zero has no PR; NaN is returned for it
    and a warning is issued, rather than letting a bare 0/0 produce a NaN
    silently.  is_localized() reports False for such a mode, since NaN fails
    every comparison.
    '''

    modes = np.asarray(modes, dtype=float)
    if modes.ndim == 2:
        modes = modes[None, ...]

    if masses is not None:
        modes = modes / np.sqrt(np.asarray(masses, dtype=float))[None, :, None]

    # weight of each ATOM in each mode
    p = np.sum(modes**2, axis=2)
    norm = np.sum(p, axis=1, keepdims=True)

    if np.any(norm <= 0):
        warnings.warn(
            '{} mode(s) have a zero eigenvector; their participation ratio '
            'is undefined and is returned as NaN.'.format(int(np.sum(norm <= 0))),
            RuntimeWarning, stacklevel=2)

    p = np.divide(p, norm, out=np.zeros_like(p), where=(norm > 0))
    ipr = np.sum(p**2, axis=1)

    pr = np.full(ipr.shape, np.nan)
    np.divide(1.0, ipr, out=pr, where=(ipr > 0))
    return pr


def inverse_participation_ratio(modes, masses=None):
    '''
    Inverse participation ratio, IPR = 1 / PR.  See participation_ratio().

    IPR is 1/nions for a fully delocalized mode and 1 for a mode confined to a
    single atom, so it grows as a mode becomes more localized.
    '''
    return 1.0 / participation_ratio(modes, masses)


def is_localized(pr, nions, threshold=LOCALIZATION_THRESHOLD):
    """
    Boolean mask, True where PR / nions < threshold.

    Takes the participation ratio rather than the eigenvectors, because every
    caller already has it: recomputing PR from the modes would mean keeping
    the mass-weighted eigenvectors around just for this, and vasp2xsf.py and
    vib_proj.py read PR back from a cache without the modes at all.  Use
        is_localized(participation_ratio(modes), modes.shape[-2])
    when starting from the eigenvectors.

    See participation_ratio() for the caveats; this is a heuristic, not a
    sharp criterion.  A NaN PR (a zero eigenvector) reports False.
    """
    return np.asarray(pr, dtype=float) < threshold * nions
