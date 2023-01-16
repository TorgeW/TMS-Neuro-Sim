import math
import multiprocessing
import time
from typing import Tuple

import numpy as np
from tqdm import tqdm

from tmsneurosim.nrn.cells.neuron_cell import NeuronCell
from tmsneurosim.nrn.simulation.simulation import WaveformType
from tmsneurosim.nrn.simulation.threshold_uniform_gradient_simulation import ThresholdUniformGradientSimulation

progress_counter = multiprocessing.Array('i', [])


def simulate_gradient_threshold_map(neuron_cell: NeuronCell, waveform_type: WaveformType, theta_step, phi_step,
                                    e_gradient_min_percentage=None, e_gradient_max_percentage=None,
                                    number_of_e_gradients=None,
                                    theta_gradient_min_degree=None, theta_gradient_max_degree=None,
                                    number_of_theta_degree=None,
                                    phi_gradient_min_degree=None, phi_gradient_max_degree=None,
                                    number_of_phi_degree=None,
                                    process_count: int = multiprocessing.cpu_count() // 2):
    """
    Simulates a gradient threshold map multi threaded.
    :param neuron_cell: The neuron cell to create the gradient threshold map from.
    :param waveform_type: The waveform type to use.
    :param theta_step: The step size of the polar angle in degree.
    :param phi_step: The step size of the azimuthal angle in degree.
    :param e_gradient_min_percentage: The minimum relative magnitude delta in percent.
    :param e_gradient_max_percentage: The maximum relative magnitude delta in percent.
    :param number_of_e_gradients: The number of relative magnitude delta steps
    :param theta_gradient_min_degree: The minimum polar angle delta in degree.
    :param theta_gradient_max_degree: The maximum polar angle delta in degree.
    :param number_of_theta_degree: The number of polar angle delta steps
    :param phi_gradient_min_degree: The minimum azimuthal angle delta in degree.
    :param phi_gradient_max_degree: The maximum azimuthal angle delta in degree.
    :param number_of_phi_degree: The number of azimuthal angle delta steps.
    :param process_count: The number of processes to use.
    :return: The threshold results and the E-field parameter per result.
    """
    if e_gradient_min_percentage is None or e_gradient_max_percentage is None or number_of_e_gradients is None:
        e_gradient_min_percentage = 0
        e_gradient_max_percentage = 0
        number_of_e_gradients = 1

    if phi_gradient_min_degree is None or phi_gradient_max_degree is None or number_of_phi_degree is None:
        phi_gradient_min_degree = 0
        phi_gradient_max_degree = 0
        number_of_phi_degree = 1

    if theta_gradient_min_degree is None or theta_gradient_max_degree is None or number_of_theta_degree is None:
        theta_gradient_min_degree = 0
        theta_gradient_max_degree = 0
        number_of_theta_degree = 1

    e_field_list = []
    for theta_gradient in np.linspace(theta_gradient_min_degree, theta_gradient_max_degree, number_of_theta_degree):
        for phi_gradient in np.linspace(phi_gradient_min_degree, phi_gradient_max_degree, number_of_phi_degree):
            for e_magnitude_gradient in np.linspace(e_gradient_min_percentage, e_gradient_max_percentage,
                                                    number_of_e_gradients):
                for theta in range(0, 180 + theta_step, theta_step):
                    for phi in range(0, 360, phi_step):
                        e_field_list.append((theta, phi, e_magnitude_gradient, phi_gradient, theta_gradient))

    chunk_size = math.ceil(len(e_field_list) / process_count)
    parameters_per_process = [e_field_list[i:i + chunk_size] for i in range(0, len(e_field_list), chunk_size)]

    global progress_counter
    progress_counter = multiprocessing.Array('i', [-1] * process_count)

    results = []
    with multiprocessing.Pool(processes=process_count) as pool:
        sims = []
        for i, e_field_sub_list in enumerate(parameters_per_process):
            sims.append((
                pool.apply_async(
                    simulate_uniform_threshold,
                    (neuron_cell, waveform_type, e_field_sub_list, i)
                )))
        while sum(progress_counter) < 0:
            time.sleep(1)

        previous_progress = 0
        all_done = False
        with tqdm(total=len(e_field_list), unit='sims',
                  desc=f'{neuron_cell.__class__.__name__}_{neuron_cell.morphology_id}') as pbar:
            while not all_done:
                progress = sum(progress_counter)
                if previous_progress != progress:
                    pbar.update(progress - previous_progress)
                    previous_progress = progress
                time.sleep(1)
                all_done = True
                for s in sims:
                    if not s.ready():
                        all_done = False
                        break

        for s in sims:
            results += s.get()
        pool.close()
        pool.join()
    return results, e_field_list


def simulate_uniform_threshold(neuron_cell: NeuronCell, waveform_type: WaveformType,
                               e_field_list: [Tuple[float, float, float, float]], index=None):
    """
    Simulation of a number of different artificial E-fields.
    :param neuron_cell: The cell to use for the simulation.
    :param waveform_type: The waveform to use for the simulation.
    :param e_field_list: The list of different E-field parameters to use.
    :param index: The precess index.
    :return: Threshold results from the different artificial E-fields.
    """
    neuron_cell.load()
    if index is not None:
        progress_counter[index] = 0
    results = []
    simulation = ThresholdUniformGradientSimulation(neuron_cell, waveform_type)
    simulation.attach()

    for i, (theta, phi, relative_e_gradient_per_mm, phi_gradient_per_mm, theta_gradient_per_mm) in enumerate(
            e_field_list):
        simulation.apply_e_field(theta, phi, relative_e_gradient_per_mm, phi_gradient_per_mm, theta_gradient_per_mm)
        results.append(simulation.find_threshold_factor())
        if index is not None:
            progress_counter[index] += 1

    simulation.detach()
    return results
