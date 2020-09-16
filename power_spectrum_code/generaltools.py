import numpy
import powerbox
from scipy import interpolate
from scipy.constants import c
from scipy.constants import Boltzmann
from radiotelescope import beam_width
import time


def symlog_bounds(data):
    data_min = numpy.nanmin(data)
    data_max = numpy.nanmax(data)

    if data_min == 0:
        indices = numpy.where(data > 0)[0]
        if len(indices) == 0:
            lower_bound = -0.1
        else:
            print(numpy.nanmin(data[indices]))
            lower_bound = numpy.nanmin(data[indices])
    else:
        lower_bound = data_min

    if data_max == 0:
        indices = numpy.where(data < 0)[0]
        if len(indices) == 0:
            upper_bound = 0.1
        else:
            upper_bound = numpy.nanmax(data[indices])
    else:
        upper_bound = data_max

    ### Figure out what the lintresh is (has to be linear)
    number_orders = int(
        numpy.ceil(
            numpy.log10(numpy.abs(upper_bound)) + numpy.log10(numpy.abs(upper_bound))
        )
    )
    print(number_orders)
    short_end = min(numpy.abs(lower_bound), numpy.abs(upper_bound))
    threshold = 10 ** (2.5 * numpy.log10(short_end) / (number_orders / 10))
    #### Figure out the linscale parameter (has to be in log)
    scale = numpy.log10(upper_bound - lower_bound) / 6

    return lower_bound, upper_bound, threshold, scale


def from_lm_to_theta_phi(ll, mm):
    theta = numpy.arcsin(numpy.sqrt(ll ** 2.0 + mm ** 2.0))
    phi = numpy.arctan(mm / ll)

    # phi is undefined for theta = 0, correct
    index = numpy.where(theta == 0)
    phi[index] = 0
    return theta, phi


def from_eta_to_k_par(eta, nu_observed, H0=70.4, nu_emission=1.42e9):
    # following Morales 2004

    z = redshift(nu_observed, nu_emission)
    hubble_distance = c / H0 * 1e-3  # [Mpc]

    E = E_function(z)
    k_par = eta * 2 * numpy.pi * nu_emission * E / (hubble_distance * (1 + z) ** 2)

    return k_par


def from_u_to_k_perp(u, frequency):
    # following Morales 2004
    distance = comoving_distance(nu_min=frequency)
    k_perp = 2 * numpy.pi * u / distance

    return k_perp


def comoving_distance(nu_min=None, nu_max=None, H0=70.4):

    hubble_distance = c / H0 * 1e-3  # Mpc
    if nu_max is not None:
        z_min = redshift(nu_max)
    else:
        z_min = 0

    z_max = redshift(nu_min)
    z_integration = numpy.linspace(z_min, z_max, 100)
    E = E_function(z_integration)

    d = hubble_distance * numpy.trapz(1 / E, z_integration)

    return d


def E_function(z, Omega_M=0.27, Omega_k=0, Omega_Lambda=0.73):
    E = numpy.sqrt(Omega_M * (1 + z) ** 3 + Omega_k * (1 + z) ** 2 + Omega_Lambda)

    return E


def redshift(nu_observed, nu_emission=1.42e9):

    z = (nu_emission - nu_observed) / nu_observed

    return z


def from_jansky_to_milikelvin(
    measurements_jansky, frequencies, nu_emission=1.42e9, H0=70.4
):
    # following morales & wyithe 2010
    central_frequency = frequencies[int(len(frequencies) / 2)]
    bandwidth = frequencies.max() - frequencies.min()

    z_central = redshift(nu_observed=central_frequency, nu_emission=nu_emission)
    E = E_function(z_central)
    G = H0 * nu_emission * E / (c * (1 + z_central) ** 2) * 1e-3
    x = comoving_distance(nu_min=central_frequency)
    y = comoving_distance(nu_min=frequencies.min(), nu_max=frequencies.max())
    beamwidth = beam_width(central_frequency)

    volume = beamwidth ** 2 * x ** 2 * y
    A_eff = (c / central_frequency) ** 2 / beamwidth

    conversion = (
        (A_eff / (2 * Boltzmann * 1e26)) ** 2 * x ** 4 / G ** 2 / (volume) * 1e6
    )
    temperature = measurements_jansky * conversion

    return temperature


def uv_list_to_baseline_measurements(
    baseline_table_object, frequency, visibility_grid, uv_grid, interpolation="spline"
):

    u_bin_centers = uv_grid[0]
    v_bin_centers = uv_grid[1]

    baseline_coordinates = numpy.array(
        [baseline_table_object.u(frequency), baseline_table_object.v(frequency)]
    )
    # now we have the bin edges we can start binning our baseline table
    # Create an empty array to store our baseline measurements in
    if interpolation == "linear":
        real_component = interpolate.RegularGridInterpolator(
            [u_bin_centers, v_bin_centers], numpy.real(visibility_grid)
        )
        imag_component = interpolate.RegularGridInterpolator(
            [u_bin_centers, v_bin_centers], numpy.imag(visibility_grid)
        )

        visibilities = real_component(baseline_coordinates.T) + 1j * imag_component(
            baseline_coordinates.T
        )
    elif interpolation == "spline":
        real_component = interpolate.RectBivariateSpline(
            u_bin_centers, v_bin_centers, numpy.real(visibility_grid)
        )
        imag_component = interpolate.RectBivariateSpline(
            u_bin_centers, v_bin_centers, numpy.imag(visibility_grid)
        )

        visibilities = real_component.ev(
            baseline_coordinates[0, :], baseline_coordinates[1, :]
        ) + 1j * imag_component.ev(
            baseline_coordinates[0, :], baseline_coordinates[1, :]
        )

    return visibilities


def visibility_extractor(
    baseline_table_object,
    sky_image,
    frequency,
    antenna1_response,
    antenna2_response,
    padding_factor=3,
    interpolation="spline",
    verbose_time=False,
):

    image = sky_image * antenna1_response * numpy.conj(antenna2_response)

    fft_time0 = time.perf_counter()
    visibility_grid, uv_coordinates = powerbox.dft.fft(
        numpy.fft.ifftshift(
            numpy.pad(image, padding_factor * image.shape[0], mode="constant"),
            axes=(0, 1),
        ),
        L=2 * (2 * padding_factor + 1),
        axes=(0, 1),
    )
    fft_time1 = time.perf_counter()
    if verbose_time:
        print(f"\tFFT Time = {fft_time1 - fft_time0}")

    sample_time0 = time.perf_counter()
    measured_visibilities = uv_list_to_baseline_measurements(
        baseline_table_object,
        frequency,
        visibility_grid,
        uv_coordinates,
        interpolation=interpolation,
    )

    sample_time1 = time.perf_counter()
    if verbose_time:
        print(f"\tSample time = {sample_time1 - sample_time0}")
    return measured_visibilities
