import numpy
import powerbox

import matplotlib
from matplotlib import pyplot
import matplotlib.colors as colors

from plottools import colorbar
from generaltools import symlog_bounds
from radiotelescope import beam_width

"""
This file is going to contain all relevant power spectrum functions, i.e data gridding, (frequency tapering), frequency
fft, angular averaging, plotting

"""

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
    kernel_pixel_size = 51
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

        print(indices)
        gridded_data[kernel_x[indices], kernel_y[indices]] += (
            measured_visibilities[i] * kernel[indices]
        )
        gridded_weights[kernel_x[indices], kernel_y[indices]] += kernel[indices]

    normed_gridded_data = numpy.nan_to_num(gridded_data / gridded_weights)
    return normed_gridded_data, gridded_weights


def get_power_spectrum(
    frequency_range,
    radio_telescope,
    ideal_measured_visibilities,
    broken_measured_visibilities,
    faulty_tile,
    plot_file_name,
    gaussian_kernel=False,
    verbose=False,
):
    baseline_table = radio_telescope.baseline_table

    # Determine maximum resolution
    max_frequency = frequency_range[-1]
    max_u = numpy.max(numpy.abs(baseline_table.u(max_frequency)))
    max_v = numpy.max(numpy.abs(baseline_table.v(max_frequency)))
    max_b = max(max_u, max_v)

    re_gridding_resolution = 0.5  # lambda
    n_regridded_cells = int(numpy.ceil(2 * max_b / re_gridding_resolution))
    # ensure gridding cells are always odd numbered
    if n_regridded_cells % 2 == 0:
        n_regridded_cells += 1
    else:
        pass

    regridded_uv = numpy.linspace(-max_b, max_b, n_regridded_cells)

    if verbose:
        print("Gridding data for Power Spectrum Estimation")

    # Create empty_uvf_cubes:
    ideal_regridded_cube = numpy.zeros(
        (n_regridded_cells, n_regridded_cells, len(frequency_range)), dtype=complex
    )
    broken_regridded_cube = ideal_regridded_cube.copy()

    ideal_regridded_weights = numpy.zeros(
        (n_regridded_cells, n_regridded_cells, len(frequency_range))
    )
    broken_regridded_weights = ideal_regridded_weights.copy()

    for frequency_index in range(len(frequency_range)):
        if gaussian_kernel:
            (
                ideal_regridded_cube[..., frequency_index],
                ideal_regridded_weights[..., frequency_index],
            ) = regrid_visibilities_gaussian(
                ideal_measured_visibilities[:, frequency_index],
                baseline_table.u(frequency_range[frequency_index]),
                baseline_table.v(frequency_range[frequency_index]),
                regridded_uv,
                frequency_range[frequency_index],
            )

            (
                broken_regridded_cube[..., frequency_index],
                broken_regridded_weights[..., frequency_index],
            ) = regrid_visibilities_gaussian(
                broken_measured_visibilities[:, frequency_index],
                baseline_table.u(frequency_range[frequency_index]),
                baseline_table.v(frequency_range[frequency_index]),
                regridded_uv,
                frequency_range[frequency_index],
            )
        else:

            (
                ideal_regridded_cube[..., frequency_index],
                ideal_regridded_weights[..., frequency_index],
            ) = regrid_visibilities(
                ideal_measured_visibilities[:, frequency_index],
                baseline_table.u(frequency_range[frequency_index]),
                baseline_table.v(frequency_range[frequency_index]),
                regridded_uv,
            )

            (
                broken_regridded_cube[..., frequency_index],
                broken_regridded_weights[..., frequency_index],
            ) = regrid_visibilities(
                broken_measured_visibilities[:, frequency_index],
                baseline_table.u(frequency_range[frequency_index]),
                baseline_table.v(frequency_range[frequency_index]),
                regridded_uv,
            )

        pyplot.imshow(numpy.abs(ideal_regridded_weights[..., frequency_index]))
        pyplot.savefig("blaah1.pdf")

    # visibilities have now been re-gridded
    if verbose:
        print("Taking Fourier Transform over frequency and averaging")
    ideal_shifted = numpy.fft.ifftshift(ideal_regridded_cube, axes=2)
    broken_shifted = numpy.fft.ifftshift(broken_regridded_cube, axes=2)

    ideal_uvn, eta_coords = powerbox.dft.fft(
        ideal_shifted,
        L=numpy.max(frequency_range) - numpy.min(frequency_range),
        axes=(2,),
    )
    broken_uvn, eta_coords = powerbox.dft.fft(
        broken_shifted,
        L=numpy.max(frequency_range) - numpy.min(frequency_range),
        axes=(2,),
    )

    ideal_PS, uv_bins = powerbox.tools.angular_average_nd(
        numpy.abs(ideal_uvn) ** 2,
        coords=[regridded_uv, regridded_uv, eta_coords],
        bins=75,
        n=2,
        weights=numpy.sum(ideal_regridded_weights, axis=2),
    )

    broken_PS, uv_bins = powerbox.tools.angular_average_nd(
        numpy.abs(broken_uvn) ** 2,
        coords=[regridded_uv, regridded_uv, eta_coords],
        bins=75,
        n=2,
        weights=numpy.sum(broken_regridded_weights, axis=2),
    )

    diff_PS, uv_bins = powerbox.tools.angular_average_nd(
        numpy.abs(broken_uvn - ideal_uvn) ** 2,
        coords=[regridded_uv, regridded_uv, eta_coords],
        bins=75,
        n=2,
        weights=numpy.sum(broken_regridded_weights, axis=2),
    )

    # diff_PS = (broken_PS - ideal_PS)/ideal_PS
    selection = int(len(eta_coords[0]) / 2) + 1

    if verbose:
        print("Making 2D PS Plots")
    power_spectrum_plot(
        uv_bins,
        eta_coords[0, selection:],
        ideal_PS[:, selection:],
        broken_PS[:, selection:],
        diff_PS[:, selection:],
        plot_file_name,
        faulty_tile,
    )
    return


def power_spectrum_plot(
    uv_bins, eta_coords, ideal_PS, broken_PS, diff_PS, plot_file_name, faulty_tile=-1,
):
    fontsize = 25
    tickfontsize = 20
    figure = pyplot.figure(figsize=(30, 10))
    ideal_axes = figure.add_subplot(131)
    broken_axes = figure.add_subplot(132)
    difference_axes = figure.add_subplot(133)

    ideal_plot = ideal_axes.pcolor(
        uv_bins,
        eta_coords,
        numpy.real(ideal_PS.T),
        cmap="Spectral_r",
        norm=colors.LogNorm(
            vmin=numpy.nanmin(numpy.real(ideal_PS.T)),
            vmax=numpy.nanmax(numpy.real(ideal_PS.T)),
        ),
    )

    broken_plot = broken_axes.pcolor(
        uv_bins,
        eta_coords,
        numpy.real(broken_PS.T),
        cmap="Spectral_r",
        norm=colors.LogNorm(
            vmin=numpy.nanmin(numpy.real(broken_PS.T)),
            vmax=numpy.nanmax(numpy.real(broken_PS.T)),
        ),
    )

    symlog_min, symlog_max, symlog_threshold, symlog_scale = symlog_bounds(
        numpy.real(diff_PS)
    )

    diff_plot = difference_axes.pcolor(
        uv_bins,
        eta_coords,
        numpy.real(diff_PS.T),
        norm=colors.SymLogNorm(
            linthresh=10 ** -5, linscale=symlog_scale, vmin=symlog_min, vmax=symlog_max
        ),
        cmap="coolwarm",
    )

    ideal_axes.set_xscale("log")
    ideal_axes.set_yscale("log")

    broken_axes.set_xscale("log")
    broken_axes.set_yscale("log")

    difference_axes.set_xscale("log")
    difference_axes.set_yscale("log")

    x_labeling = r"$ k_{\perp} \, [\mathrm{h}\,\mathrm{Mpc}^{-1}]$"
    y_labeling = r"$k_{\parallel} $"

    x_labeling = r"$ |u |$"
    y_labeling = r"$ \eta $"

    ideal_axes.set_xlabel(x_labeling, fontsize=fontsize)
    broken_axes.set_xlabel(x_labeling, fontsize=fontsize)
    difference_axes.set_xlabel(x_labeling, fontsize=fontsize)

    ideal_axes.set_ylabel(y_labeling, fontsize=fontsize)

    ideal_axes.tick_params(axis="both", which="major", labelsize=tickfontsize)
    broken_axes.tick_params(axis="both", which="major", labelsize=tickfontsize)
    difference_axes.tick_params(axis="both", which="major", labelsize=tickfontsize)

    figure.suptitle(f"Tile {faulty_tile}")
    ideal_axes.set_title("Ideal Array", fontsize=fontsize)
    broken_axes.set_title("Broken Array", fontsize=fontsize)
    difference_axes.set_title("(Ideal - Broken)/Ideal", fontsize=fontsize)

    # ideal_axes.set_xlim(10**-2.5, 10**-0.5)
    # broken_axes.set_xlim(10**-2.5, 10**-0.5)
    # difference_axes.set_xlim(10**-2.5, 10**-0.5)

    print(uv_bins)

    ideal_axes.set_xlim(numpy.nanmin(uv_bins), 2 * 1e2)
    broken_axes.set_xlim(numpy.nanmin(uv_bins), 2 * 1e2)
    difference_axes.set_xlim(numpy.nanmin(uv_bins), 2 * 1e2)

    ideal_cax = colorbar(ideal_plot)
    broken_cax = colorbar(broken_plot)
    diff_cax = colorbar(diff_plot)
    diff_cax.set_label(r"$[Jy^2]$", fontsize=fontsize)
    ideal_cax.ax.tick_params(axis="both", which="major", labelsize=tickfontsize)
    broken_cax.tick_params(axis="both", which="major", labelsize=tickfontsize)
    diff_cax.ax.tick_params(axis="both", which="major", labelsize=tickfontsize)

    print(plot_file_name)
    figure.savefig(plot_file_name)
    return
