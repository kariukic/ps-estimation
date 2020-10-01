import numpy
import matplotlib.colors as colors
from matplotlib import pyplot
from scipy import signal

import powerbox

from radiotelescope import beam_width
from plottools import colorbar
from generaltools import from_eta_to_k_par, from_u_to_k_perp, from_jansky_to_milikelvin


def regrid_visibilities(measured_visibilities, baseline_u, baseline_v, u_grid):
    u_shifts = numpy.diff(u_grid) / 2.0

    u_bin_edges = numpy.concatenate(
        (
            numpy.array([u_grid[0] - u_shifts[0]]),
            u_grid[1:] - u_shifts,
            numpy.array([u_grid[-1] + u_shifts[-1]]),
        )
    )

    weights_regrid, u_bins, v__bins = numpy.histogram2d(
        baseline_u, baseline_v, bins=(u_bin_edges, u_bin_edges)
    )

    real_regrid, u_bins, v__bins = numpy.histogram2d(
        baseline_u,
        baseline_v,
        bins=(u_bin_edges, u_bin_edges),
        weights=numpy.real(measured_visibilities),
    )

    imag_regrid, u_bins, v__bins = numpy.histogram2d(
        baseline_u,
        baseline_v,
        bins=(u_bin_edges, u_bin_edges),
        weights=numpy.imag(measured_visibilities),
    )

    regridded_visibilities = real_regrid + 1j * imag_regrid
    normed_regridded_visibilities = numpy.nan_to_num(
        regridded_visibilities / weights_regrid
    )
    return normed_regridded_visibilities, weights_regrid


def regrid_visibilities_gaussian(
    measured_visibilities, baseline_u, baseline_v, u_grid, frequency
):
    u_shifts = numpy.diff(u_grid) / 2.0

    u_bin_edges = numpy.concatenate(
        (
            numpy.array([u_grid[0] - u_shifts[0]]),
            u_grid[1:] - u_shifts,
            numpy.array([u_grid[-1] + u_shifts[-1]]),
        )
    )

    gridded_data = numpy.zeros((len(u_grid), len(u_grid)), dtype=complex)
    gridded_weights = numpy.zeros((len(u_grid), len(u_grid)))

    # calculate the kernel
    kernel_pixel_size = 52
    if kernel_pixel_size % 2 == 0:
        dimension = kernel_pixel_size / 2
    else:
        dimension = (kernel_pixel_size + 1) / 2

    grid_midpoint = int(len(u_grid) / 2)
    kernel_width = beam_width(frequency)
    print(kernel_width)
    kernel_grid = u_grid[
        int(grid_midpoint - dimension) : int(grid_midpoint + dimension + 1)
    ]
    uu, vv = numpy.meshgrid(kernel_grid, kernel_grid)

    kernel = numpy.exp(-(kernel_width ** 2) * (uu ** 2.0 + vv ** 2.0)).flatten()
    kernel_coordinates = numpy.arange(-dimension, dimension + 1, 1, dtype=int)
    kernel_mapx, kernel_mapy = numpy.meshgrid(kernel_coordinates, kernel_coordinates)

    for i in range(len(measured_visibilities)):
        u_index = numpy.digitize(numpy.array(baseline_u[i]), u_bin_edges)
        v_index = numpy.digitize(numpy.array(baseline_v[i]), u_bin_edges)

        kernel_x = kernel_mapx.flatten() + u_index
        kernel_y = kernel_mapy.flatten() + v_index

        # filter indices which are beyond array range
        indices = numpy.where(
            (kernel_x > 0)
            & (kernel_x < len(u_grid))
            & (kernel_y > 0)
            & (kernel_y < len(u_grid))
        )[0]

        # print(indices)
        gridded_data[kernel_x[indices], kernel_y[indices]] += (
            measured_visibilities[i] * kernel[indices]
        )
        gridded_weights[kernel_x[indices], kernel_y[indices]] += kernel[indices]

    normed_gridded_data = numpy.nan_to_num(gridded_data / gridded_weights)

    return normed_gridded_data, gridded_weights


def get_power_spectrum(
    frequency_range,
    baseline_u,
    baseline_v,
    ideal_measured_visibilities,
    power2D_path,
    eta_coords_path,
    plot_file_name,
    gaussian_kernel=False,
    convert_axes_to_k_units=False,
    verbose=True,
):

    # Determine maximum resolution
    max_u = numpy.max(numpy.abs(baseline_u[:, -1]))
    max_v = numpy.max(numpy.abs(baseline_v[:, -1]))
    max_b = max(max_u, max_v)

    re_gridding_resolution = 0.5  # lambda
    n_regridded_cells = 1000  # int(numpy.ceil(2 * max_b / re_gridding_resolution))
    # ensure gridding cells are always odd numbered
    if n_regridded_cells % 2 == 0:
        n_regridded_cells += 1
    else:
        pass

    regridded_uv = numpy.linspace(-max_b, max_b, n_regridded_cells)

    if verbose:
        print("Gridding data for Power Spectrum Estimation")
        # print("uv_max:", max_b)
        # print("n_grid_cells:", n_regridded_cells)
        # print("regridded_uv:", regridded_uv)

    # Create empty_uvf_cubes:
    ideal_regridded_cube = numpy.zeros(
        (n_regridded_cells, n_regridded_cells, len(frequency_range)), dtype=complex
    )

    ideal_regridded_weights = numpy.zeros(
        (n_regridded_cells, n_regridded_cells, len(frequency_range))
    )

    for frequency_index in range(len(frequency_range)):
        if gaussian_kernel:
            (
                ideal_regridded_cube[..., frequency_index],
                ideal_regridded_weights[..., frequency_index],
            ) = regrid_visibilities_gaussian(
                ideal_measured_visibilities[:, frequency_index],
                baseline_u[:, frequency_index],
                baseline_v[:, frequency_index],
                regridded_uv,
                frequency_range[frequency_index],
            )

        else:

            (
                ideal_regridded_cube[..., frequency_index],
                ideal_regridded_weights[..., frequency_index],
            ) = regrid_visibilities(
                ideal_measured_visibilities[:, frequency_index],
                baseline_u[:, frequency_index],
                baseline_v[:, frequency_index],
                regridded_uv,
            )

        pyplot.imshow(numpy.abs(ideal_regridded_weights[..., frequency_index]))
        pyplot.savefig("ideal_regridded_weights.pdf")

    # visibilities have now been re-gridded
    if verbose:
        print("Taking Fourier Transform over frequency and averaging")
    ideal_shifted = numpy.fft.ifftshift(ideal_regridded_cube, axes=2)

    taper = signal.blackmanharris
    ideal_uvn, eta_coords = powerbox.dft.fft(
        ideal_shifted * taper(len(frequency_range)),
        L=numpy.max(frequency_range) - numpy.min(frequency_range),
        axes=(2,),
        a=0,
    )

    # The 3D power spectrum
    power_3d = numpy.absolute(ideal_uvn) ** 2

    print("power_3d.shape", power_3d.shape)
    print("eta_coords", eta_coords)

    # Cylindrical averaging to 2D PS
    ideal_PS, uv_bins = powerbox.tools.angular_average_nd(
        power_3d,
        coords=[regridded_uv, regridded_uv, eta_coords],
        bins=75,
        n=2,
        weights=numpy.sum(ideal_regridded_weights, axis=2),
    )

    selection = int(len(eta_coords[0]) / 2) + 1

    axes = "uvn"
    if convert_axes_to_k_units:
        axes = "k"
        central_frequency = frequency_range[int(len(frequency_range) / 2)]
        eta_coords = from_eta_to_k_par(eta_coords, central_frequency)
        uv_bins = from_u_to_k_perp(uv_bins, central_frequency)

    ideal_PS = ideal_PS[:, selection:]  # return the positive part
    # ideal_PS = from_jansky_to_milikelvin(ideal_PS, frequency_range)

    print(uv_bins.shape, "uv_bins SHAPE")
    print(ideal_PS.shape, "power2D SHAPE")

    if verbose:
        print("saving 2D power")
    numpy.save(
        power2D_path,
        numpy.real(ideal_PS).T,
    )
    numpy.save(
        eta_coords_path,
        eta_coords[0, selection:],
    )

    if verbose:
        print("Making 2D PS Plots")
    power_spectrum_plot(
        uv_bins,
        eta_coords[0, selection:],
        numpy.real(ideal_PS).T,
        plot_file_name,
        axes=axes,
    )
    return


def power_spectrum_plot(uv_bins, eta_coords, ideal_PS, plot_file_name, axes="k"):
    fontsize = 15
    tickfontsize = 15
    figure = pyplot.figure(figsize=(6, 8))
    ideal_axes = figure.add_subplot(111)

    ideal_plot = ideal_axes.pcolor(
        uv_bins,
        eta_coords,
        ideal_PS,
        cmap="Spectral_r",
        norm=colors.LogNorm(vmin=10 ** 12, vmax=10 ** 16),
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
    ideal_axes.set_xlim(numpy.nanmin(uv_bins), numpy.nanmax(uv_bins))

    ideal_cax = colorbar(ideal_plot)
    ideal_cax.ax.tick_params(axis="both", which="major", labelsize=tickfontsize)

    print(plot_file_name)
    figure.savefig(plot_file_name)
    return
