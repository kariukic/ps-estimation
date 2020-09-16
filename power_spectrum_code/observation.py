import numpy
from scipy.constants import c
from scipy import interpolate

import multiprocessing
from functools import partial

import powerbox

from generaltools import from_lm_to_theta_phi
from radiotelescope import ideal_mwa_beam_loader
from radiotelescope import ideal_gaussian_beam


def get_observations(
    source_population,
    baseline_table,
    frequency_range,
    beam_type,
    compute_mode="parallel",
    processes=None,
):
    print(f"Running calculations in {compute_mode}")

    if compute_mode == "parallel":
        ideal_observations = get_observation_MP(
            source_population,
            baseline_table,
            frequency_range,
            beam_type,
            processes=processes,
        )

    else:
        raise ValueError(
            f"compute_mode can be 'parallel', 'serial', or 'high_memory' NOT {compute_mode}"
        )

    print(ideal_observations.shape, " shape")
    return ideal_observations


def get_observation_MP(
    source_population,
    baseline_table,
    frequency_range,
    beam_type,
    processes=4,
):
    # Determine maximum resolution
    max_frequency = frequency_range[-1]
    max_u = numpy.max(numpy.abs(baseline_table.u(max_frequency)))
    max_v = numpy.max(numpy.abs(baseline_table.v(max_frequency)))
    max_b = max(max_u, max_v)
    # sky_resolutions
    min_l = 1.0 / (2 * max_b)

    pool = multiprocessing.Pool(processes=processes)
    iterator = partial(
        get_observation_single_channel,
        source_population,
        baseline_table,
        min_l,
        beam_type,
        frequency_range,
    )
    ideal_observations_list = pool.map(iterator, range(len(frequency_range)))

    ideal_observations = numpy.moveaxis(numpy.array(ideal_observations_list), 0, -1)

    return ideal_observations


def get_observation_single_channel(
    source_population,
    baseline_table,
    min_l,
    beam_type,
    frequency_range,
    frequency_index,
):
    sky_image, l_coordinates = source_population.create_sky_image(
        frequency_channels=frequency_range[frequency_index],
        resolution=min_l,
        oversampling=1,
    )
    ll, mm = numpy.meshgrid(l_coordinates, l_coordinates)

    # Create Beam
    #############################################################################
    if beam_type == "MWA":
        (
            tt,
            pp,
        ) = from_lm_to_theta_phi(ll, mm)
        ideal_beam = ideal_mwa_beam_loader(
            tt, pp, frequency_range[frequency_index], load=False
        )

    elif beam_type == "gaussian":
        ideal_beam = ideal_gaussian_beam(ll, mm, frequency_range[frequency_index])
    else:
        raise ValueError("The only valid option for the beam are 'MWA' or 'gaussian'")

    ideal_measured_visibilities = visibility_extractor(
        baseline_table,
        sky_image,
        frequency_range[frequency_index],
        ideal_beam,
        ideal_beam,
    )

    return ideal_measured_visibilities


def visibility_extractor(
    baseline_table_object,
    sky_image,
    frequency,
    antenna1_response,
    antenna2_response,
    padding_factor=3,
):

    apparent_sky = sky_image * antenna1_response * numpy.conj(antenna2_response)

    padded_sky = numpy.pad(
        apparent_sky, padding_factor * apparent_sky.shape[0], mode="constant"
    )
    shifted_image = numpy.fft.ifftshift(padded_sky, axes=(0, 1))
    visibility_grid, uv_coordinates = powerbox.dft.fft(
        shifted_image, L=2 * (2 * padding_factor + 1), axes=(0, 1)
    )

    measured_visibilities = uv_list_to_baseline_measurements(
        baseline_table_object, frequency, visibility_grid, uv_coordinates
    )

    return measured_visibilities


def uv_list_to_baseline_measurements(
    baseline_table_object, frequency, visibility_grid, uv_grid
):

    u_bin_centers = uv_grid[0]
    v_bin_centers = uv_grid[1]

    baseline_coordinates = numpy.array(
        [baseline_table_object.u(frequency), baseline_table_object.v(frequency)]
    )
    # now we have the bin edges we can start binning our baseline table
    # Create an empty array to store our baseline measurements in
    visibility_data = visibility_grid

    real_component = interpolate.RegularGridInterpolator(
        [u_bin_centers, v_bin_centers], numpy.real(visibility_data)
    )
    imag_component = interpolate.RegularGridInterpolator(
        [u_bin_centers, v_bin_centers], numpy.imag(visibility_data)
    )

    visibilities = real_component(baseline_coordinates.T) + 1j * imag_component(
        baseline_coordinates.T
    )

    return visibilities
