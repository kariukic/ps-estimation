import os
import time
import argparse
import numpy as np
from scipy.constants import c

import mset_utils

from power_spectrum_parallel import Bestimator
from power_spectrum_R_modified import get_power_spectrum

from radiotelescope import RadioTelescope
from observation import get_observations
from skymodel import SkyRealisation


def main(
    mset,
    visibilities_path,
    n_channels,
    n_obs,
    beam_type,
    ps_estimator,
    verbose=True,
):
    ##################################################################
    # saving filepaths:
    output_path = os.getcwd() + "/Simulation_Output/"
    input_visibilities = "mset_vis" if mset else "R_simulated_vis"
    if ps_estimator == "B_estimator":
        project_name = input_visibilities + "/" + "B_estimator/"
    else:
        project_name = input_visibilities + "/" + "R_estimator/"
    if not os.path.exists(output_path + project_name):
        os.makedirs(output_path + project_name)

    suffix = "%schannels_%sbands_%s" % (n_channels, n_obs, ps_estimator)
    kernel_weights_path = output_path + project_name + suffix + "_kernel_weights.npy"
    power2D_path = output_path + project_name + suffix + "_2D_power.npy"
    eta_coords_path = output_path + project_name + suffix + "_eta_coords.npy"
    ps2d_plot_path = output_path + project_name + suffix + "_2D_power.pdf"
    #####################################################################
    ################################################################
    # Getting the input visibilities
    if mset:
        if verbose:
            print("Reading MS...")
        visibilities, frequencies, baseline_u, baseline_v = mset_utils.read_mset(
            mset, n_channels
        )
    else:
        xyz_coords_path = "MWA_Compact_Coordinates.txt"  # change if need be
        telescope = RadioTelescope(load=True, path=xyz_coords_path, verbose=verbose)
        baseline_table = telescope.baseline_table
        frequencies = np.linspace(135, 165, n_channels) * 1e6
        sky_param = "random"
        beam_type = "gaussian"
        mode = "parallel"
        processes = 6
        if visibilities_path:
            if verbose:
                print("loading visibilities from disk")
            visibilities = np.load(visibilities_path)
            assert (
                visibilities.shape[1] == n_channels
            ), "Your input n_channels should be same as loaded visibilities n_channels"
        else:
            if verbose:
                print("Building up an MWA telescope observation from scratch!!")
            source_population = SkyRealisation(sky_type=sky_param, verbose=verbose)
            if verbose:
                print("Generating visibility measurements for each frequency")
            visibilities = get_observations(
                source_population,
                baseline_table,
                frequencies,
                beam_type,
                compute_mode=mode,
                processes=processes,
            )

        baseline_u = baseline_table.u()[:, None] * c / frequencies
        baseline_v = baseline_table.v()[:, None] * c / frequencies

    if ps_estimator == "R_estimator":
        if verbose:
            print("Running R estimator")
        get_power_spectrum(
            frequencies,
            baseline_u,
            baseline_v,
            visibilities,
            power2D_path,
            eta_coords_path,
            ps2d_plot_path,
            gaussian_kernel=True,
            convert_axes_to_k_units=False,
            verbose=verbose,
        )
    elif ps_estimator == "B_estimator":
        if verbose:
            print("Running B  estimator")
        Bestimator(
            visibilities,
            frequencies,
            baseline_u,
            baseline_v,
            n_obs,
            kernel_weights_path,
            power2D_path,
            eta_coords_path,
            ps2d_plot_path,
        )

    file = open(
        output_path + project_name + suffix + "_simulation_params.log",
        "w",
    )
    file.write(f"Frequency range: {np.min(frequencies)} - {np.max(frequencies)} MHz \n")
    file.write(f"No. of channels: {n_channels} \n")
    file.write(f"PS Estimator: {ps_estimator} \n")
    file.write(f"Measurement set: {mset} \n")
    if not mset:
        file.write(f"Sky Parameters: {sky_param} \n")
        file.write(f"Beam model:{beam_type} \n")
        file.write(f"Position File: {xyz_coords_path}")
    file.close()


if __name__ == "__main__":
    start = time.process_time()
    parser = argparse.ArgumentParser(description="Ideal visibilities Simulation Set Up")
    parser.add_argument("-mset", required=False, help="Template measurement set")
    parser.add_argument("-vis_path", required=False, help="Template measurement set")
    parser.add_argument("-number_channels", action="store", default=128, type=int)
    parser.add_argument("-n_obs", action="store", default=1, type=int)
    parser.add_argument(
        "-estimator",
        choices=["B_estimator", "R_estimator"],
        required=True,
        help="Template measurement set",
    )
    parser.add_argument("-beam", action="store", default="gaussian", type=str)
    parser.add_argument("-verbose", action="store_true", default=True)
    args = parser.parse_args()
    main(
        args.mset,
        args.vis_path,
        args.number_channels,
        args.n_obs,
        args.beam,
        args.estimator,
        args.verbose,
    )
    end = time.process_time()
