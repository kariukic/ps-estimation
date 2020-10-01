# Useful MS manipulation functions.
import numpy as np
from casacore.tables import table, maketabdesc, makearrcoldesc

c = 299792458


def read_mset(mset, number_channels):
    mset_table = table(mset, readonly=True)
    uvw = get_uvw(mset_table)
    frequencies = get_channels(mset_table, ls=False)  # [0:number_channels]
    print(
        f"Frequency Channels: {len(frequencies)}, minimum: {min(frequencies)}, maximum: {max(frequencies)}"
    )
    frequencies = frequencies[0:number_channels]
    lmbdas = c / frequencies
    uvw_lmbdas = uvw[:, None, :] / lmbdas[None, :, None]
    visibilities = get_data(mset_table)

    no_of_timestamps = len(set(get_time_stamps(mset_table)))
    print("Number of timestamps in MS: ", no_of_timestamps)

    number_of_baselines = visibilities.shape[0] // no_of_timestamps
    visibilities = visibilities[0:number_of_baselines, 0:number_channels, 0]
    baseline_u = uvw_lmbdas[0:number_of_baselines, :, 0]
    baseline_v = uvw_lmbdas[0:number_of_baselines, :, 1]

    # calculate number of baselines
    number_of_antennas = 130
    while number_of_antennas > 2:
        number_of_antennas -= 1
        if 0.5 * number_of_antennas * (number_of_antennas - 1) == number_of_baselines:
            break
    print("baseline_v shape", baseline_v.shape)
    print("number of antennas", number_of_antennas)
    print("number ofbaselines: ", number_of_baselines)
    print("visibilities shape: ", visibilities.shape)

    mset_table.close()
    return visibilities, frequencies, baseline_u, baseline_v


def get_data(tbl):
    data = tbl.getcol("DATA")
    return data


def get_uvw(tbl):
    uvw = tbl.getcol("UVW")
    return uvw


def get_time_stamps(tbl):
    timestamps = tbl.getcol("TIME")
    return timestamps


def get_phase_center(tbl):
    """
    Grabs the phase centre of the observation in RA and Dec

    Parameters
    ----------
    tbl : casacore table.
        The casacore mset table opened with readonly=False.\n
    Returns
    -------
    float, float.
        RA and Dec in radians.
    """
    ra0, dec0 = tbl.FIELD.getcell("PHASE_DIR", 0)[0]
    return ra0, dec0


def get_channels(tbl, ls=True):
    if ls:
        chans = c / tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    else:
        chans = tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    return chans


def get_ant12(mset):
    ms = table(mset, readonly=False, ack=False)
    antenna1 = ms.getcol("ANTENNA1")
    antenna2 = ms.getcol("ANTENNA2")
    return antenna1, antenna2


def put_col(tbl, col, dat):
    """add data 'dat' to the column 'col'"""
    tbl.putcol(col, dat)


def add_col(tbl, colnme):
    """Add a column 'colnme' to the MS"""
    col_dmi = tbl.getdminfo("DATA")
    col_dmi["NAME"] = colnme
    shape = tbl.getcell("DATA", 0).shape
    tbl.addcols(
        maketabdesc(
            makearrcoldesc(colnme, 0.0 + 0.0j, valuetype="complex", shape=shape)
        ),
        col_dmi,
        addtoparent=True,
    )


def get_lmns(tbl, ra_rad, dec_rad):
    """
    Calculating l, m, n values from ras,decs and phase centre.
    𝑙 = cos 𝛿 * sin Δ𝛼
    𝑚 = sin 𝛿 * cos 𝛿0 − cos 𝛿 * sin 𝛿0 * cos Δ𝛼
    Δ𝛼 = 𝛼 − 𝛼0
    """
    ra0, dec0 = get_phase_center(tbl)

    ra_delta = ra_rad - ra0
    ls = np.cos(dec_rad) * np.sin(ra_delta)
    ms = np.sin(dec_rad) * np.cos(dec0) - np.cos(dec_rad) * np.sin(dec0) * np.cos(
        ra_delta
    )
    ns = np.sqrt(1 - ls ** 2 - ms ** 2) - 1

    return ls, ms, ns


def get_bl_lens(mset):
    """Calculate the baseline length for each DATA row in the measurement set"""
    t = table(mset + "/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()

    tt = table(mset)
    ant1 = tt.getcol("ANTENNA1")
    ant2 = tt.getcol("ANTENNA2")
    tt.close()

    bls = np.zeros(len(ant1))
    for i in range(len(ant1)):
        p = ant1[i]
        q = ant2[i]
        pos1, pos2 = pos[p], pos[q]
        bls[i] = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    return bls


def get_bl_vectors(mset, refant=0):
    """
    Gets the antenna XYZ position coordinates and recalculates them with the reference antenna as the origin.

    Parameters
    ----------
    mset : Measurement set. \n
    refant : int, optional
        The reference antenna ID, by default 0. \n

    Returns
    -------
    XYZ coordinates of each antenna with respect to the reference antenna.
    """
    # First get the positions of each antenna recorded in XYZ coordinates from the MS
    t = table(mset + "/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()

    no_ants = len(pos)
    print("The mset has %s antennas." % (no_ants))

    bls = np.zeros((no_ants, 3))
    for i in range(no_ants):  # calculate and fill bls with distances from the refant
        pos1, pos2 = pos[i], pos[refant]
        bls[i] = np.array([pos2 - pos1])
    return bls
