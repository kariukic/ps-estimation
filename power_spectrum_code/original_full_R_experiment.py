import os
import time
import argparse
import numpy

from radiotelescope import RadioTelescope
from observation import get_observations
from power_spectrum_R import get_power_spectrum
from skymodel import SkyRealisation


def main(beam_type="gaussian", n_channels=100, verbose=True):
    print(
        "Beam type: ", beam_type, "No. of channels: ", n_channels, "Verbose: ", verbose
    )
    output_path = "/astro/mwaeor/kchege/power_spectrum_code/Simulation_Output/ps_from_raw_array_vis_R/"
    prefix = str(n_channels)
    suffix = "_sky_ideal_vis_BmH_taper_k_units_"

    path = "MWA_Compact_Coordinates.txt"
    frequency_range = numpy.linspace(135, 165, n_channels) * 1e6
    sky_param = "random"
    mode = "parallel"
    processes = 15
    # calibrate = True
    # beam_type = "gaussian"
    plot_file_name = "MWA_ideal_random_sky_%s_chans_PS.pdf" % (n_channels)

    telescope = RadioTelescope(load=True, path=path, verbose=verbose)
    baseline_table = telescope.baseline_table
    source_population = SkyRealisation(sky_type=sky_param, verbose=verbose)

    ####################################################################################################################
    if verbose:
        print("Generating visibility measurements for each frequency")
    ideal_measured_visibilities = get_observations(
        source_population,
        baseline_table,
        frequency_range,
        beam_type,
        compute_mode=mode,
        processes=processes,
    )
    #####################################################################################################################

    # save simulated data:
    project_name = prefix + beam_type + sky_param + suffix

    if not os.path.exists(output_path + project_name):
        print
        ""
        print
        "!!!Warning: Creating output folder at output destination!"
        os.makedirs(output_path + project_name)

        numpy.save(
            output_path + project_name + "/" + "ideal" + "_simulated_data",
            ideal_measured_visibilities,
        )

    file = open(output_path + project_name + "/" + "simulation_parameters.log", "w")
    file.write(
        f"Frequency range: {numpy.min(frequency_range)} - {numpy.max(frequency_range)} MHz \n"
    )
    file.write(f"Sky Parameters: {sky_param} \n")
    file.write(f"Beam model:{beam_type} \n")
    file.write(f"Position File: {path}")
    file.close()

    get_power_spectrum(
        frequency_range,
        telescope,
        ideal_measured_visibilities,
        output_path + project_name + "/" + plot_file_name,
        gaussian_kernel=True,
        convert_axes_to_k_units=False,
        verbose=True,
    )

    return


if __name__ == "__main__":
    start = time.process_time()
    parser = argparse.ArgumentParser(description="ideal visibilities Simulation Set Up")
    parser.add_argument("-beam", action="store", default="gaussian", type=str)
    parser.add_argument("-number_channels", action="store", default=100, type=int)
    parser.add_argument("-verbose", action="store_true", default=True)
    args = parser.parse_args()
    main(
        args.beam,
        args.number_channels,
        args.verbose,
    )
    end = time.process_time()
    print("Total time is", end - start)
