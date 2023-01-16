import gc
import logging
import math
import multiprocessing
import time
from typing import Tuple, List

import numpy as np
import simnibs
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from tmsneurosim.cortical_layer import CorticalLayer
from tmsneurosim.nrn.cells import NeuronCell
from tmsneurosim.nrn.simulation.e_field_simulation import EFieldSimulation
from tmsneurosim.nrn.simulation.simulation import WaveformType

WHITE_MATTER_SURFACE = 1001
GRAY_MATTER_SURFACE = 1002

progress_counter = multiprocessing.Array('i', [])


def simulate_combined_threshold_layer(layer: CorticalLayer, cells: [NeuronCell], waveform_type: WaveformType,
                                      rotation_count: int, rotation_step: int, initial_rotation: int = None,
                                      process_count: int = multiprocessing.cpu_count() // 2) -> simnibs.Msh:
    """
    Simulates the threshold of each neuron at each simulation element with all azimuthal rotations.
    :param layer: The cortical layer to place the neurons on.
    :param cells: The neurons to be used for the simulations.
    :param waveform_type: The waveform type to be used for the simulation.
    :param rotation_count: The number of azimuthal rotation per cell.
    :param rotation_step: The step width of the azimuthal rotations per cell.
    :param initial_rotation: The initial azimuthal rotation.
    :param process_count: The number of process to be used.
    :return: The layer with the results in the form of element fields.
    """

    process_count = min(process_count, len(layer.elements))

    if initial_rotation is None:
        azimuthal_rotation = np.random.rand(len(layer.elements)) * 360
    else:
        azimuthal_rotation = np.full(len(layer.elements), initial_rotation)

    layer_results = []
    layer_meta_info = []
    for cell in cells:
        for i in range(rotation_count):
            layer_results.append(simulate_threshold_layer(layer, cell, waveform_type,
                                                          azimuthal_rotation + i * rotation_step,
                                                          i, process_count))
            layer_meta_info.append((cell, i))

    layer.add_selected_elements_field(azimuthal_rotation, 'Initial_Rotation')
    layer.add_selected_elements_field(layer.get_smoothed_normals(), 'Normal')

    for layer_result, (cell, i) in zip(layer_results, layer_meta_info):
        layer.add_selected_elements_field(np.array(layer_result[0]),
                                          f'{cell.__class__.__name__}_{cell.morphology_id}__{i}')
        layer.add_selected_elements_field(np.array(layer_result[1]),
                                          f'{cell.__class__.__name__}_{cell.morphology_id}_{i}_tags')

    if len(layer_results) > 1:
        layer_thresholds = np.array([x[0] for x in layer_results])
        layer.add_selected_elements_field(np.nanmedian(layer_thresholds, axis=0), 'Median_Threshold')
        layer.add_selected_elements_field(np.nanmean(layer_thresholds, axis=0), 'Mean_Threshold')
        layer.add_nearest_interpolation_field(np.nanmedian(layer_thresholds, axis=0), 'Median_Threshold_Visual')
        layer.add_nearest_interpolation_field(np.nanmean(layer_thresholds, axis=0), 'Mean_Threshold_Visual')

    return layer.surface


def simulate_threshold_layer(layer: CorticalLayer, cell: NeuronCell, waveform_type: WaveformType,
                             azimuthal_rotations: NDArray[np.float64], rotation_count,
                             process_count: int) -> Tuple[List[float], List[int]]:
    """
    Simulates the threshold for one neuron at each simulation element with all azimuthal rotations.
    :param layer: The cortical layer to place the neurons on.
    :param cell: The neuron to be used for the simulations.
    :param waveform_type: The waveform type to be used for the simulation.
    :param azimuthal_rotations: A list of azimuthal rotations for each simulation position on the layer.
    :param rotation_count: The rotation index.
    :param process_count: The number of processes to be used
    :return:
    """
    layer_triangle_normals = layer.get_smoothed_normals()
    layer_triangle_center = layer.surface.elements_baricenters().value[layer.elements]

    chunk_size = math.ceil(len(layer.elements) / process_count)
    directions_per_process = [layer_triangle_normals[i:i + chunk_size]
                              for i in range(0, len(layer_triangle_normals), chunk_size)]
    positions_per_process = [layer_triangle_center[i:i + chunk_size]
                             for i in range(0, len(layer_triangle_center), chunk_size)]
    azimuthal_rotations_per_process = [azimuthal_rotations[i:i + chunk_size]
                                       for i in range(0, len(azimuthal_rotations), chunk_size)]

    global progress_counter
    progress_counter = multiprocessing.Array('i', [-1] * process_count)

    logging.info(f'Started {cell.__class__.__name__}_{cell.morphology_id} ({rotation_count})')
    thresholds = []
    tags = []
    with multiprocessing.Pool(processes=process_count) as pool:
        sims = []
        for i, (directions, positions, azimuthal_rotations) in enumerate(zip(directions_per_process,
                                                                             positions_per_process,
                                                                             azimuthal_rotations_per_process)):
            sims.append(
                pool.apply_async(
                    simulate_cell_thresholds,
                    (cell, waveform_type, directions, positions, azimuthal_rotations, layer, i)
                ))
        while sum(progress_counter) < 0:
            time.sleep(1)
        previous_progress = 0
        all_done = False
        with tqdm(total=len(layer.elements), unit='sims',
                  desc=f'{cell.__class__.__name__}_{cell.morphology_id} ({rotation_count})') as pbar:
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
            progress = sum(progress_counter)
            pbar.update(progress - previous_progress)

        for s in sims:
            thresholds_and_tags = s.get()
            thresholds += thresholds_and_tags[0]
            tags += thresholds_and_tags[1]
        pool.close()
        pool.join()

    logging.info(f'Finished {cell.__class__.__name__}_{cell.morphology_id} ({rotation_count})')

    return thresholds, tags


def simulate_cell_thresholds(cell: NeuronCell, waveform_type: WaveformType, directions: [NDArray[np.float64]],
                             positions: [NDArray[np.float64]], azimuthal_rotations: NDArray[np.float64],
                             layer: CorticalLayer, index) -> Tuple[List[float], List[int]]:
    """
    Simulates the threshold for a neuron in a number of positions, directions and azimuthal rotations.
    :param cell: The neuron to be used by the simulation.
    :param waveform_type: The waveform type to be used by the simulation.
    :param directions: A list of neuron directions.
    :param positions: A list of positions on the layer for neurons to be placed.
    :param azimuthal_rotations: A list of neuron azimuthal rotations.
    :param layer: The cortical layer.
    :param index: The index of the process.
    :return: Return thresholds for each set of parameters. Also returns tissue tags of the tissues the
    simulated cell ranged into.
    """
    cell.load()
    simulation = EFieldSimulation(cell, waveform_type)
    simulation.attach()
    progress_counter[index] = 0
    segment_points_mm = cell.get_segment_coordinates() * 0.001
    thresholds = []
    tags = []
    for i, (direction, position, azimuthal_rotation) in enumerate(zip(directions, positions, azimuthal_rotations)):
        threshold_and_tag = simulate_cell_threshold(cell, simulation, segment_points_mm,
                                                    direction, position, azimuthal_rotation, layer)
        thresholds.append(threshold_and_tag[0])
        tags.append(threshold_and_tag[1])
        progress_counter[index] += 1
        gc.collect()
    simulation.detach()
    return thresholds, tags


def simulate_cell_threshold(cell: NeuronCell, simulation, segment_points_mm,
                            direction: NDArray[np.float64], position: NDArray[np.float64], azimuthal_rotation: float,
                            layer: CorticalLayer) -> Tuple[float, int]:
    """
    Simulates the threshold of one neuron at a specific position with a direction and azimuthal rotation.
    :param cell: The neuron to be used in the simulation.
    :param simulation: The simulation instance to be used.
    :param segment_points_mm: The segment coordinates in mm.
    :param direction: The direction of the neuron.
    :param position: The position of the neuron.
    :param azimuthal_rotation: The azimuthal rotation of the neuron.
    :param layer: The cortical layer.
    :return: The threshold of the specified simulation.
    """
    if np.any(np.isnan(direction)) or np.any(np.isnan(position)):
        return np.nan

    azimuthal_rotation = Rotation.from_euler('z', azimuthal_rotation, degrees=True)
    rotation = rotation_from_vectors(cell.direction, direction)

    transformed_cell_coordinates = np.array(segment_points_mm) - segment_points_mm[0]
    transformed_cell_coordinates = rotation.apply(azimuthal_rotation.apply(transformed_cell_coordinates)) + position

    e_field_at_cell, tetrahedron_tags = layer.interpolate_scattered_cashed(transformed_cell_coordinates, 'E',
                                                                           out_fill=np.nan, get_tetrahedron_tags=True)
    if np.isnan(e_field_at_cell).any():
        tetrahedron_tags = np.append(tetrahedron_tags, [0])
    # e_field_at_cell_2 = layer.volumetric_mesh.field['E'].interpolate_scattered(transformed_cell_coordinates, out_fill=0)
    e_field_at_cell = np.nan_to_num(e_field_at_cell, nan=0)
    transformed_e_field = azimuthal_rotation.inv().apply(rotation.inv().apply(e_field_at_cell))

    simulation.apply_e_field(transformed_e_field)
    return simulation.find_threshold_factor(), int(''.join(map(str, np.unique(tetrahedron_tags)))[::-1])


def rotation_from_vectors(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> Rotation:
    """ Creates a rotation instance that rotates vector 1 into the direction of vector 2

    :param vec1: The original direction
    :param vec2: The final direction
    :return: A rotation instance that rotates vector 1 into the direction of vector 2
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return Rotation.identity()
    vec1_normalized = (vec1 / np.linalg.norm(vec1)).reshape(3)
    vec2_normalized = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(vec1_normalized, vec2_normalized)
    rotation = Rotation.identity()
    if any(v):
        c = np.dot(vec1_normalized, vec2_normalized)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = Rotation.from_matrix(np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)))
    return rotation
