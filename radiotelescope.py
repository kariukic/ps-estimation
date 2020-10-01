import numpy
import copy
import os
from scipy.constants import c


class RadioTelescope:
    """
    A radio telescope object that returns the baselines coordinates.
    """

    def __init__(
        self, load=True, path=None, shape=None, frequency_channels=None, verbose=False
    ):

        if verbose:
            print("Creating the radio telescope")
        self.antenna_positions = None
        if shape is not None:
            self.antenna_positions = AntennaPositions(
                False, None, shape, verbose)
        if load:
            self.antenna_positions = AntennaPositions(
                True, path, None, verbose)
        if shape is not None or load:
            self.baseline_table = BaselineTable(
                self.antenna_positions, frequency_channels, verbose
            )
        else:
            self.baseline_table = None
        return


class AntennaPositions:
    def __init__(self, load=True, path=None, shape=None, verbose=False):
        if load:
            if path == None:
                raise ValueError(
                    "Specificy the antenna position path if loading position data"
                )
            else:
                antenna_data = xyz_position_loader(path)

        # if shape is not None:
        #    antenna_data = xyz_position_creator(shape, verbose=verbose)

        if load or shape is not None:
            self.antenna_ids = antenna_data[:, 0]
            self.x_coordinates = antenna_data[:, 1]
            self.y_coordinates = antenna_data[:, 2]
            self.z_coordinates = antenna_data[:, 3]
        else:
            self.antenna_ids = None
            self.x_coordinates = None
            self.y_coordinates = None
            self.z_coordinates = None
        return

    def number_antennas(self):
        return len(self.antenna_ids)


class BaselineTable:
    def __init__(self, position_table=None, frequency_channels=None, verbose=False):
        self.antenna_id1 = None
        self.antenna_id2 = None
        self.u_coordinates = None
        self.v_coordinates = None
        self.w_coordinates = None
        self.reference_frequency = None
        self.number_of_baselines = None
        self.selection = None
        # update all attributes
        if position_table is not None:
            self.baseline_converter(
                position_table, frequency_channels, verbose)
        return

    def baseline_converter(self, position_table, frequency_channels=None, verbose=True):
        if verbose:
            print("")
            print("Converting xyz to uvw-coordinates")

        if frequency_channels is None:
            self.reference_frequency = 150e6
        elif type(frequency_channels) == numpy.ndarray:
            assert (
                min(frequency_channels) > 1e6
            ), "Frequency range is smaller 1 MHz, probably wrong units"
            self.reference_frequency = frequency_channels[0]
        elif numpy.isscalar(frequency_channels):
            assert (
                frequency_channels > 1e6
            ), "Frequency range is smaller 1 MHz, probably wrong units"
            self.reference_frequency = frequency_channels
        else:
            raise ValueError(
                f"frequency_channels should be 'numpy.ndarray', or scalar not type({self.reference_frequency})"
            )

        # calculate the wavelengths of the adjecent channels
        reference_wavelength = c / self.reference_frequency
        # Count the number of antenna
        number_of_antenna = position_table.number_antennas()
        # Calculate the number of possible baselines
        self.number_of_baselines = int(
            0.5 * number_of_antenna * (number_of_antenna - 1.0)
        )

        # Create arrays for the baselines
        # baselines x Antenna1, Antenna2, u, v, w, gain product, phase sum x channels
        antenna_1 = numpy.zeros(self.number_of_baselines)
        antenna_2 = antenna_1.copy()

        u_coordinates = antenna_1.copy()
        v_coordinates = antenna_1.copy()
        w_coordinates = antenna_1.copy()

        if verbose:
            print("")
            print("Number of antenna =", number_of_antenna)
            print("Total number of baselines =", self.number_of_baselines)

        # arbitrary counter to keep track of the baseline table
        k = 0

        for i in range(number_of_antenna):
            for j in range(i + 1, number_of_antenna):
                # save the antenna numbers in the uv table
                antenna_1[k] = position_table.antenna_ids[i]
                antenna_2[k] = position_table.antenna_ids[j]

                # rescale and write uvw to multifrequency baseline table
                u_coordinates[k] = (
                    position_table.x_coordinates[i] -
                    position_table.x_coordinates[j]
                ) / reference_wavelength
                v_coordinates[k] = (
                    position_table.y_coordinates[i] -
                    position_table.y_coordinates[j]
                ) / reference_wavelength
                w_coordinates[k] = (
                    position_table.z_coordinates[i] -
                    position_table.z_coordinates[j]
                ) / reference_wavelength

                k += 1

        self.antenna_id1 = antenna_1
        self.antenna_id2 = antenna_2

        self.u_coordinates = u_coordinates
        self.v_coordinates = v_coordinates
        self.w_coordinates = w_coordinates
        return

    def u(self, frequency=None):
        rescaled_u = rescale_baseline(
            self.u_coordinates, self.reference_frequency, frequency
        )
        selected_rescaled_u = select_baselines(rescaled_u, self.selection)

        return selected_rescaled_u

    def v(self, frequency=None):
        rescaled_v = rescale_baseline(
            self.v_coordinates, self.reference_frequency, frequency
        )
        selected_rescaled_v = select_baselines(rescaled_v, self.selection)

        return selected_rescaled_v

    def w(self, frequency=None):
        rescaled_w = rescale_baseline(
            self.w_coordinates, self.reference_frequency, frequency
        )
        selected_rescaled_w = select_baselines(rescaled_w, self.selection)

        return selected_rescaled_w

    def sub_table(self, baseline_selection_indices):
        subtable = copy.copy(self)
        subtable.selection = baseline_selection_indices
        subtable.number_of_baselines = len(baseline_selection_indices)

        return subtable


def rescale_baseline(baseline_coordinates, reference_frequency, frequency):
    if frequency is None:
        rescaled_coordinates = baseline_coordinates
    elif numpy.isscalar(frequency):
        rescaling_factor = frequency / reference_frequency
        rescaled_coordinates = baseline_coordinates * rescaling_factor
    elif type(frequency) == numpy.ndarray:
        rescaling_factor = frequency / reference_frequency
        coordinate_mesh, rescale_mesh = numpy.meshgrid(
            rescaling_factor, baseline_coordinates
        )
        rescaled_coordinates = coordinate_mesh * rescale_mesh
    else:
        raise ValueError(
            f"frequency should be scalar or numpy.ndarray not {type(frequency)}"
        )

    return rescaled_coordinates


def select_baselines(baseline_coordinates, baseline_selection_indices):
    if baseline_selection_indices is None:
        selected_baseline_coordinates = baseline_coordinates
    else:
        selected_baseline_coordinates = baseline_coordinates[
            baseline_selection_indices, ...
        ]
    return selected_baseline_coordinates


def xyz_position_loader(path):
    antenna_data = numpy.loadtxt(path)
    # Check whether antenna ids are passed are in here
    if antenna_data.shape[1] != 4:
        antenna_ids = numpy.arange(1, antenna_data.shape[0] + 1, 1).reshape(
            (antenna_data.shape[0], 1)
        )
        antenna_data = numpy.hstack((antenna_ids, antenna_data))
    elif antenna_data.shape[1] > 4:
        raise ValueError(
            f"The antenna position file should only contain 4 columns: antenna_id, x, y, z. \n "
            + f"This file contains {antenna_data.shape[1]} columns"
        )

    antenna_data = antenna_data[numpy.argsort(antenna_data[:, 0])]

    return antenna_data


##############################################################################
def beam_width(frequency, diameter=4, epsilon=1):
    sigma = epsilon * c / (frequency * diameter)
    width = numpy.sin(0.5 * sigma)
    return width


def ideal_gaussian_beam(source_l, source_m, nu, diameter=4, epsilon=1):
    sigma = beam_width(nu, diameter, epsilon)

    beam_attenuation = numpy.exp(
        -(source_l ** 2.0 + source_m ** 2.0) / (2 * sigma ** 2)
    )

    return beam_attenuation


def ideal_mwa_beam_loader(theta, phi, frequency, load=True, verbose=False):
    if not load:
        if verbose:
            print("Creating the idealised MWA beam\n")
        ideal_beam = mwa_tile_beam(theta, phi, frequency=frequency)
        if not os.path.exists("beam_maps"):
            print("")
            print("Creating beam map folder locally!")
            os.makedirs("beam_maps")
        numpy.save(f"beam_maps/ideal_beam_map.npy", ideal_beam)
    if load:
        if verbose:
            print("Loading the idealised MWA beam\n")
        ideal_beam = numpy.load(f"beam_maps/ideal_beam_map.npy")

    return ideal_beam


def mwa_tile_beam(
    theta,
    phi,
    target_theta=0,
    target_phi=0,
    frequency=150e6,
    weights=1,
    dipole_type="cross",
    gaussian_width=30 / 180 * numpy.pi,
):
    dipole_sep = 1.1  # meters
    x_offsets = (
        numpy.array(
            [
                -1.5,
                -0.5,
                0.5,
                1.5,
                -1.5,
                -0.5,
                0.5,
                1.5,
                -1.5,
                -0.5,
                0.5,
                1.5,
                -1.5,
                -0.5,
                0.5,
                1.5,
            ],
            dtype=numpy.float32,
        )
        * dipole_sep
    )

    y_offsets = (
        numpy.array(
            [
                1.5,
                1.5,
                1.5,
                1.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -1.5,
                -1.5,
                -1.5,
                -1.5,
            ],
            dtype=numpy.float32,
        )
        * dipole_sep
    )
    z_offsets = numpy.zeros(x_offsets.shape)

    weights += numpy.zeros(x_offsets.shape)

    if dipole_type == "cross":
        dipole_jones_matrix = cross_dipole(theta)
    # elif dipole_type == "gaussian":
    # print(theta_width)
    #    dipole_jones_matrix = gaussian_response(theta, gaussian_width)
    else:
        print("Wrong dipole_type: select cross or gaussian")

    ground_plane_field = electric_field_ground_plane(theta, frequency)
    array_factor = get_array_factor(
        x_offsets,
        y_offsets,
        z_offsets,
        weights,
        theta,
        phi,
        target_theta,
        target_phi,
        frequency,
    )

    tile_response = array_factor * ground_plane_field * dipole_jones_matrix
    tile_response[numpy.isnan(tile_response)] = 0

    if len(theta.shape) > 2:
        beam_normalisation = numpy.add(
            numpy.zeros(tile_response.shape), numpy.amax(
                tile_response, axis=(0, 1))
        )
    else:
        beam_normalisation = numpy.add(
            numpy.zeros(tile_response.shape), numpy.amax(tile_response)
        )
    normalised_response = tile_response / \
        beam_normalisation * numpy.sum(weights) / 16

    return normalised_response


def get_array_factor(
    x, y, z, weights, theta, phi, theta_pointing=0, phi_pointing=0, frequency=150e6
):
    wavelength = c / frequency
    number_dipoles = len(x)

    k_x = (2.0 * numpy.pi / wavelength) * numpy.sin(theta) * numpy.sin(phi)
    k_y = (2.0 * numpy.pi / wavelength) * numpy.sin(theta) * numpy.cos(phi)
    k_z = (2.0 * numpy.pi / wavelength) * numpy.cos(theta)

    k_x0 = (
        (2.0 * numpy.pi / wavelength)
        * numpy.sin(theta_pointing)
        * numpy.sin(phi_pointing)
    )
    k_y0 = (
        (2.0 * numpy.pi / wavelength)
        * numpy.sin(theta_pointing)
        * numpy.cos(phi_pointing)
    )
    k_z0 = (2.0 * numpy.pi / wavelength) * numpy.cos(theta_pointing)
    array_factor_map = numpy.zeros(theta.shape, dtype=complex)

    for i in range(number_dipoles):
        complex_exponent = -1j * (
            (k_x - k_x0) * x[i] + (k_y - k_y0) * y[i] + (k_z - k_z0) * z[i]
        )

        # !This step takes a long time, look into optimisation through vectorisation/clever numpy usage
        dipole_factor = weights[i] * numpy.exp(complex_exponent)

        array_factor_map += dipole_factor

    # filter all NaN
    array_factor_map[numpy.isnan(array_factor_map)] = 0
    array_factor_map = array_factor_map / numpy.sum(weights)

    return array_factor_map


def electric_field_ground_plane(theta, frequency=150e6, height=0.3):
    wavelength = c / frequency
    ground_plane_electric_field = numpy.sin(
        2.0 * numpy.pi * height / wavelength * numpy.cos(theta)
    )
    return ground_plane_electric_field


def cross_dipole(theta):
    response = numpy.cos(theta)
    return response
