import os
import logging
import numpy as np
import multiprocessing

from scipy import signal
from scipy.constants import c

from powerbox.tools import angular_average_nd
from powerbox.dft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib import colors
from plottools import colorbar

logger = logging.getLogger("PS_parallel")


def beam_sigma(frequencies, tile_diameter=4):
    "The Gaussian beam width at each frequency"
    epsilon = 0.42  # scaling from airy disk to Gaussian
    return (epsilon * c) / (frequencies * tile_diameter)


def uvplane_grid(uv_max, gridding_resolution=0.5):
    # re_gridding_resolution = 0.5  # lambda
    n_regridded_cells = int(
        np.ceil(uv_max / gridding_resolution)
    )  # int(np.ceil(2 * uv_max / gridding_resolution))
    # n_regridded_cells = 200
    # ensure gridding cells are always odd numbered i.e.
    # +1 because these are bin edges.
    if n_regridded_cells % 2 == 0:
        n_regridded_cells += 1
    else:
        pass

    u_grid = np.linspace(-uv_max, uv_max, n_regridded_cells)
    # u_grid = (u_grid[1:] + u_grid[:-1]) / 2

    return n_regridded_cells, u_grid


def maximum_resolution(baseline_u, baseline_v):
    max_u = np.max(np.abs(baseline_u[:, -1]))
    max_v = np.max(np.abs(baseline_v[:, -1]))
    uv_max = max(max_u, max_v)
    return uv_max


def u_min(u_grid):
    """Minimum of |u| grid"""
    return np.abs(u_grid).min()


def u_edges(u_min, u_max, n_regridded_cells):
    """Edges of |u| bins where |u| = sqrt(u**2+v**2)"""
    return np.linspace(u_min, u_max, n_regridded_cells)  # same as u_grid!


def fourierBeam(centres, u_bl, v_bl, frequency, min_attenuation=1e-10, N=20):
    """
    Find the Fourier Transform of the Gaussian beam

    Parameter
    ---------
    centres : (ngrid)-array
        The centres of the grid.

    u_bl : (n_baselines)-array
        The list of baselines in m.

    v_bl : (n_baselines)-array
        The list of baselines in m.

    frequency: float
        The frequency in Hz.
    """

    a = 1 / (2 * beam_sigma(frequency) ** 2)

    indx_u = np.digitize(u_bl, centres)
    indx_v = np.digitize(v_bl, centres)

    beam = []

    for jj in range(len(u_bl)):
        x, y = np.meshgrid(
            centres[indx_u[jj] - int(N / 2) : indx_u[jj] + int(N / 2)],
            centres[indx_v[jj] - int(N / 2) : indx_v[jj] + int(N / 2)],
            copy=False,
        )
        B = (np.exp(-((x - u_bl[jj]) ** 2 + (y - v_bl[jj]) ** 2) / a)).T
        B[B < min_attenuation] = 0
        beam.append(B)

    indx_u += -int(N / 2)
    indx_v += -int(N / 2)

    indx_u[indx_u < 0] = 0
    indx_v[indx_v < 0] = 0

    return beam, indx_u, indx_v


def grid_visibilities(
    visibilities,
    frequencies,
    baseline_u,
    baseline_v,
    n_grid_cells,
    u_grid,
    N=52,
    kernel_weights_path=None,
):
    """
    Grid a set of visibilities from baselines onto a UV grid.

    Uses Fourier (Gaussian) beam weighting to perform the gridding.

    Parameters
    ----------
    visibilities : complex (n_baselines, n_freq)-array
        The visibilities at each baseline and frequency.
    u_grid : same as the centers

    Returns
    -------
    visgrid : (ngrid, ngrid, n_freq)-array
        The visibility grid, in Jy.
    """
    logger.info("Gridding the visibilities")

    visgrid = np.zeros(
        (n_grid_cells, n_grid_cells, len(frequencies)), dtype=np.complex128
    )

    if kernel_weights_path:
        kernel_weights = np.load(kernel_weights_path)
    else:
        kernel_weights = None

    if kernel_weights is None:
        weights = np.zeros((n_grid_cells, n_grid_cells, len(frequencies)))

    for jj, freq in enumerate(frequencies):
        u_bl = baseline_u[:, jj]  # (self.baselines[:, 0] * freq / c).value
        v_bl = baseline_v[:, jj]  # (self.baselines[:, 1] * freq / c).value

        beam, indx_u, indx_v = fourierBeam(u_grid, u_bl, v_bl, freq, N=N)
        # print("beam shape", beam.shape)
        for kk in range(len(indx_u)):
            visgrid[
                indx_u[kk] : indx_u[kk] + np.shape(beam[kk])[0],
                indx_v[kk] : indx_v[kk] + np.shape(beam[kk])[1],
                jj,
            ] += (
                beam[kk] / np.sum(beam[kk]) * visibilities[kk, jj]
            )

            if kernel_weights is None:

                weights[
                    indx_u[kk] : indx_u[kk] + np.shape(beam[kk])[0],
                    indx_v[kk] : indx_v[kk] + np.shape(beam[kk])[1],
                    jj,
                ] += beam[kk] / np.sum(beam[kk])

    if kernel_weights is None:
        kernel_weights = weights

    visgrid[kernel_weights != 0] /= kernel_weights[kernel_weights != 0]

    return visgrid, kernel_weights


def grid_visibilities_parallel(
    visibilities,
    frequencies,
    baseline_u,
    baseline_v,
    uv_max,
    n_grid_cells,
    u_grid,
    n_obs,
    min_attenuation=1e-10,
    N=52,
    kernel_weights_path=None,
):
    """
    Grid a set of visibilities from baselines onto a UV grid.

    Uses Fourier (Gaussian) beam weighting to perform the gridding.
    Parameters
    ----------
    visibilities : complex (n_baselines, n_freq)-array
        The visibilities at each basline and frequency.

    Returns
    -------
    visgrid : (ngrid, ngrid, n_freq)-array
        The visibility grid, in Jy.
    """

    # Find out the number of frequencies to process per thread
    nfreq = len(frequencies)
    numperthread = int(np.ceil(nfreq / n_obs))
    offset = 0
    nfreqstart = np.zeros(n_obs, dtype=int)
    nfreqend = np.zeros(n_obs, dtype=int)
    infreq = np.zeros(n_obs, dtype=int)
    for i in range(n_obs):
        nfreqstart[i] = offset
        nfreqend[i] = offset + numperthread

        if i == n_obs - 1:
            infreq[i] = nfreq - offset
        else:
            infreq[i] = numperthread

        offset += numperthread

    # Set the last process to the number of frequencies
    nfreqend[-1] = nfreq

    processes = []

    visgrid = np.zeros(
        (n_grid_cells, n_grid_cells, len(frequencies)), dtype=np.complex128
    )

    if kernel_weights_path:
        kernel_weights = np.load(kernel_weights_path)
    else:
        kernel_weights = None

    if kernel_weights is None:
        weights = np.zeros((n_grid_cells, n_grid_cells, len(frequencies)))

    visgrid_buff_real = []
    visgrid_buff_imag = []
    weights_buff = []

    # Lets split this array up into chunks
    for i in range(n_obs):

        visgrid_buff_real.append(
            multiprocessing.RawArray(
                np.sctype2char(visgrid.real),
                visgrid[:, :, nfreqstart[i] : nfreqend[i]].size,
            )
        )
        visgrid_buff_imag.append(
            multiprocessing.RawArray(
                np.sctype2char(visgrid.imag),
                visgrid[:, :, nfreqstart[i] : nfreqend[i]].size,
            )
        )
        visgrid_tmp_real = np.frombuffer(visgrid_buff_real[i])
        visgrid_tmp_real = visgrid[:, :, nfreqstart[i] : nfreqend[i]].real.flatten()

        visgrid_tmp_imag = np.frombuffer(visgrid_buff_imag[i])
        visgrid_tmp_imag = visgrid[:, :, nfreqstart[i] : nfreqend[i]].imag.flatten()

        if kernel_weights is None:
            weights_buff.append(
                multiprocessing.RawArray(
                    np.sctype2char(weights),
                    weights[:, :, nfreqstart[i] : nfreqend[i]].size,
                )
            )
            weights_tmp = np.frombuffer(weights_buff[i])
            weights_tmp = weights[:, :, nfreqstart[i] : nfreqend[i]]
        else:
            weights_buff.append(None)

        processes.append(
            multiprocessing.Process(
                target=_grid_visibilities_buff,
                args=(
                    n_grid_cells,
                    visgrid_buff_real[i],
                    visgrid_buff_imag[i],
                    weights_buff[i],
                    visibilities[:, nfreqstart[i] : nfreqend[i]],
                    frequencies[nfreqstart[i] : nfreqend[i]],
                    baseline_u,
                    baseline_v,
                    u_grid,
                    beam_sigma(frequencies[nfreqstart[i] : nfreqend[i]]),
                    min_attenuation,
                    N,
                ),
            )
        )

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    for i in range(n_obs):

        visgrid[:, :, nfreqstart[i] : nfreqend[i]].real = np.frombuffer(
            visgrid_buff_real[i]
        ).reshape(n_grid_cells, n_grid_cells, nfreqend[i] - nfreqstart[i])
        visgrid[:, :, nfreqstart[i] : nfreqend[i]].imag = np.frombuffer(
            visgrid_buff_imag[i]
        ).reshape(n_grid_cells, n_grid_cells, nfreqend[i] - nfreqstart[i])

        if kernel_weights is None:
            weights[:, :, nfreqstart[i] : nfreqend[i]] = np.frombuffer(
                weights_buff[i]
            ).reshape(n_grid_cells, n_grid_cells, nfreqend[i] - nfreqstart[i])

    if kernel_weights is None:
        kernel_weights = weights

    visgrid[kernel_weights != 0] /= kernel_weights[kernel_weights != 0]

    return visgrid, kernel_weights


def _grid_visibilities_buff(
    n_uv,
    visgrid_buff_real,
    visgrid_buff_imag,
    weights_buff,
    visibilities,
    frequencies,
    baseline_u,
    baseline_v,
    centres,
    sigfreq,
    min_attenuation=1e-10,
    N=52,
):

    logger.info("Gridding the visibilities")

    nfreq = len(frequencies)

    vis_real = np.frombuffer(visgrid_buff_real).reshape(n_uv, n_uv, nfreq)
    vis_imag = np.frombuffer(visgrid_buff_imag).reshape(n_uv, n_uv, nfreq)

    if weights_buff is not None:
        weights = np.frombuffer(weights_buff).reshape(n_uv, n_uv, nfreq)

    for ii in range(nfreq):

        # freq = frequencies[ii]

        u_bl = baseline_u[:, ii]  # (self.baselines[:, 0] * freq / c).value
        v_bl = baseline_v[:, ii]  # (self.baselines[:, 1] * freq / c).value

        a = 1 / (2 * sigfreq[ii] ** 2)

        indx_u = np.digitize(u_bl, centres)
        indx_v = np.digitize(v_bl, centres)

        beam = np.zeros([len(u_bl), N, N])
        xshape = np.zeros(len(u_bl), dtype=int)
        yshape = np.zeros(len(u_bl), dtype=int)

        for jj in range(len(u_bl)):
            x, y = np.meshgrid(
                centres[indx_u[jj] - int(N / 2) : indx_u[jj] + int(N / 2)],
                centres[indx_v[jj] - int(N / 2) : indx_v[jj] + int(N / 2)],
                copy=False,
            )
            B = (np.exp(-((x - u_bl[jj]) ** 2 + (y - v_bl[jj]) ** 2) / a)).T
            B[B < min_attenuation] = 0
            xshape[jj] = B.shape[0]
            yshape[jj] = B.shape[1]
            beam[jj][: xshape[jj], : yshape[jj]] = B

        indx_u += -int(N / 2)
        indx_v += -int(N / 2)

        indx_u[indx_u < 0] = 0
        indx_v[indx_v < 0] = 0

        for kk in range(len(indx_u)):
            vis_real[
                indx_u[kk] : indx_u[kk] + xshape[kk],
                indx_v[kk] : indx_v[kk] + yshape[kk],
                ii,
            ] += (
                beam[kk][: xshape[kk], : yshape[kk]]
                / np.sum(beam[kk][: xshape[kk], : yshape[kk]])
                * visibilities[kk, ii].real
            )
            vis_imag[
                indx_u[kk] : indx_u[kk] + xshape[kk],
                indx_v[kk] : indx_v[kk] + yshape[kk],
                ii,
            ] += (
                beam[kk][: xshape[kk], : yshape[kk]]
                / np.sum(beam[kk][: xshape[kk], : yshape[kk]])
                * visibilities[kk, ii].imag
            )

            if weights_buff is not None:
                weights[
                    indx_u[kk] : indx_u[kk] + xshape[kk],
                    indx_v[kk] : indx_v[kk] + yshape[kk],
                    ii,
                ] += beam[kk][: xshape[kk], : yshape[kk]] / np.sum(
                    beam[kk][: xshape[kk], : yshape[kk]]
                )


def compute_power(
    visibilities,
    kernel_weights,
    frequencies,
    u_grid,
    n_regridded_cells,
    n_obs=1,
    ps_dim=2,
    baselines_type="not_grid_centres",
):
    """
    Compute the 2D power spectrum within the current context.

    Parameters
    ----------
    visibilities : (nbl, nf)-complex-array
        The visibilities of each baseline at each frequency

    Returns
    -------
    power2d : (nperp, npar)-array
        The 2D power spectrum.

    coords : list of 2 arrays
        The first is kperp, and the second is kpar.
    """

    visgrid = visibilities

    # Transform frequency axis
    visgrid, eta_coords = frequency_fft(
        visgrid,
        frequencies,
        ps_dim,
        taper=signal.blackmanharris,
        n_obs=n_obs,
    )  # self.frequency_taper)

    # Get 2D power from gridded vis.
    power2d, uv_bins = get_power(
        visgrid,
        kernel_weights,
        u_grid,
        frequencies,
        eta_coords,
        n_regridded_cells,
        ps_dim=ps_dim,
    )
    return power2d, uv_bins, eta_coords


def get_power(
    gridded_vis,
    kernel_weights,
    u_grid,
    frequencies,
    eta_coords,
    n_regridded_cells,
    ps_dim=2,
):
    """
    Determine the 2D Power Spectrum of the observation.

    Parameters
    ----------

    gridded_vis : complex (ngrid, ngrid, neta)-array
        The gridded visibilities, fourier-transformed along the frequency axis. Units JyHz.

    coords: list of 3 1D arrays.
        The [u,v,eta] co-ordinates corresponding to the gridded fourier visibilities. u and v in 1/rad, and
        eta in 1/Hz.

    Returns
    -------
    PS : float (n_obs, n_eta, bins)-list
        The cylindrical averaged (or 2D) Power Spectrum, with units JyHz**2.
    """
    logger.info("Calculating the power spectrum")
    PS = []
    print(
        "input u_grid, u_grid, eta_coords and weights shape",
        np.array(u_grid).shape,
        np.array(u_grid).shape,
        np.array(eta_coords).shape,
        np.sum(kernel_weights, axis=2).shape,
    )
    for vis in gridded_vis:
        # The 3D power spectrum
        power_3d = np.absolute(vis) ** 2

        if ps_dim == 2:
            P, uv_bins = angular_average_nd(
                field=power_3d,
                coords=[u_grid, u_grid, eta_coords],
                bins=75,  # n_regridded_cells,
                n=ps_dim,
                weights=np.sum(kernel_weights, axis=2),  # weights,
                bin_ave=False,
            )  # [0]

        elif ps_dim == 1:

            P = angular_average_nd(
                field=power_3d,
                coords=[u_grid, u_grid, eta_coords],
                bins=75,  # n_regridded_cells,
                weights=kernel_weights,
                bin_ave=False,
            )  # [0]

        P[np.isnan(P)] = 0
        PS.append(P)

    return np.array(PS), uv_bins


def frequency_fft(vis, freq, dim, taper=signal.blackmanharris, n_obs=1):
    """
    Fourier-transform a gridded visibility along the frequency axis.

    Parameters
    ----------
    vis : complex (ncells, ncells, nfreq)-array
        The gridded visibilities.

    freq : (nfreq)-array
        The linearly-spaced frequencies of the observation.

    taper : callable, optional
        A function which computes a taper function on an nfreq-array. Default is to have no taper. Callable should
        take single argument, N.

    n_obs : int, optional
        Number of observations used to separate the visibilities into different bandwidths.

    Returns
    -------
    ft : (ncells, ncells, nfreq/2)-array
        The fourier-transformed signal, with negative eta removed.

    eta : (nfreq/2)-array
        The eta-coordinates, without negative values.
    """
    ft = []
    eta_coords = []
    W = (freq.max() - freq.min()) / n_obs
    L = int(len(freq) / n_obs)

    for ii in range(n_obs):
        ffteed = fft(
            vis[:, :, ii * L : (ii + 1) * L] * taper(L),
            W,
            axes=(2,),
            a=0,
            b=2 * np.pi,
        )
        ft.append(ffteed[0][:, :, int(L / 2) :])  # return the positive part)
        eta_coords.append(ffteed[1][:, int(L / 2) :])

    dnu = freq[1] - freq[0]
    etaz = fftfreq(int(len(freq)), d=dnu, b=2 * np.pi)
    etaz = np.array(etaz)[len(freq) // 2 :]
    print(etaz.shape, etaz)
    return np.array(ft), np.array(etaz)  # np.array(eta_coords)


def eta(frequencies, n_obs=1):
    "Grid of positive frequency fourier-modes"
    dnu = frequencies[1] - frequencies[0]
    print("dnu", dnu)
    eta = fftfreq(int(len(frequencies) / n_obs), d=dnu, b=2 * np.pi)
    return eta


def power_spectrum_plot(uv_bins, eta_coords, ideal_PS, plot_file_name, axes="k"):
    fontsize = 15
    tickfontsize = 15
    figure = plt.figure(figsize=(6, 8))
    ideal_axes = figure.add_subplot(111)

    ideal_plot = ideal_axes.pcolor(
        uv_bins,
        eta_coords,
        ideal_PS,
        cmap="Spectral_r",
        norm=colors.LogNorm(vmin=10 ** 12, vmax=10 ** 15),
    )

    ideal_axes.set_xscale("log")
    ideal_axes.set_yscale("log")

    if axes == "k":
        x_labeling = r"$ k_{\perp} \, [\mathrm{h}\,\mathrm{Mpc}^{-1}]$"
        y_labeling = r"$k_{\parallel}$ [$h$Mpc$^{-1}$]"
    else:
        x_labeling = r"$ |u |$"
        y_labeling = r"$ \eta $"

    ideal_axes.set_xlabel(x_labeling, fontsize=fontsize)

    ideal_axes.set_ylabel(y_labeling, fontsize=fontsize)

    ideal_axes.tick_params(axis="both", which="major", labelsize=tickfontsize)

    ideal_axes.set_title("stochastic sky, Gaussian beam, BmH taper", fontsize=fontsize)

    # ideal_axes.set_xlim(10**-2.5, 10**-0.5)
    # broken_axes.set_xlim(10**-2.5, 10**-0.5)
    # difference_axes.set_xlim(10**-2.5, 10**-0.5)

    print(uv_bins)

    # ideal_axes.set_xlim(numpy.nanmin(uv_bins), 2 * 1e2)
    ideal_axes.set_xlim(np.nanmin(uv_bins), np.nanmax(uv_bins))

    ideal_cax = colorbar(ideal_plot)
    ideal_cax.ax.tick_params(axis="both", which="major", labelsize=tickfontsize)

    print(plot_file_name)
    figure.savefig(plot_file_name)
    return


def Bestimator(
    visibilities,
    frequencies,
    baseline_u,
    baseline_v,
    n_obs,
    kernel_weights_path,
    power2D_path,
    eta_coords_path,
    ps2d_plot_path,
    verbose=True,
):
    uv_max = maximum_resolution(baseline_u, baseline_v)
    n_grid_cells, u_grid = uvplane_grid(uv_max)
    if verbose:
        print("Gridding Visibilities...")
        print("uv_max:", uv_max)
        print("n_grid_cells:", n_grid_cells)

    if os.path.exists(kernel_weights_path):
        kernel_weights_ = kernel_weights_path
    else:
        kernel_weights_ = None

    # Grid visibilities
    if n_obs == 1:
        gridded_visibilities, kernel_weights = grid_visibilities(
            visibilities,
            frequencies,
            baseline_u,
            baseline_v,
            n_grid_cells,
            u_grid,
            N=52,
            kernel_weights_path=kernel_weights_,
        )
    else:
        gridded_visibilities, kernel_weights = grid_visibilities_parallel(
            visibilities,
            frequencies,
            baseline_u,
            baseline_v,
            uv_max,
            n_grid_cells,
            u_grid,
            n_obs,
            min_attenuation=1e-10,
            N=52,
            kernel_weights_path=kernel_weights_,
        )

    if kernel_weights_ is None:
        np.save(
            kernel_weights_path,
            kernel_weights,
        )

    if verbose:
        print("Computing the 2D power...")
    power2D, uv_bins, eta_coords = compute_power(
        gridded_visibilities,
        kernel_weights,
        frequencies,
        u_grid,
        n_grid_cells,
        n_obs=1,
        ps_dim=2,
    )

    power2D = np.array(np.real(power2D[0])).T  # return the positive part
    print(uv_bins.shape, "uv_bins SHAPE")
    # vmin = np.nanmin(power2D)
    # vmax = np.nanmax(power2D)
    # print(vmin, vmax, "VMINMAX")
    # power2D = np.where(power2D > 0, power2D, np.nan)

    if verbose:
        print("saving 2D power")
    np.save(
        power2D_path,
        power2D,
    )

    np.save(
        eta_coords_path,
        eta_coords,
    )

    power_spectrum_plot(
        uv_bins,
        eta_coords,
        power2D,
        ps2d_plot_path,
        axes="e",
    )
    return
