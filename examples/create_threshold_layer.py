import datetime
import os
import pathlib
import sys
import time

import simnibs

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from tmsneurosim.cortical_layer import CorticalLayer
from tmsneurosim.nrn import cells
from tmsneurosim.simulate_threshold_layer import simulate_combined_threshold_layer
from tmsneurosim.nrn.simulation.simulation import WaveformType

if __name__ == '__main__':
    example_path = os.path.dirname(os.path.abspath(__file__))
    output_path = pathlib.Path(example_path).joinpath('output')
    sim_mesh = simnibs.read_msh(str(pathlib.Path(example_path).parent.joinpath('data/example.msh')))

    roi = [-50, -18, -33, 1, 3.2, 53]
    layer_depth = [0.06, 0.4, 0.55, 0.65, 0.85]
    layer_cells = [cells.L23_PC_cADpyr(1), cells.L23_PC_cADpyr(2), cells.L23_PC_cADpyr(3),
                   cells.L23_PC_cADpyr(4), cells.L23_PC_cADpyr(5)]
    rotation_count = 6
    timeStart = time.perf_counter()
    layer = CorticalLayer(sim_mesh, roi, layer_depth[1], 1)
    layer.add_e_field_at_triangle_centers_field()
    result_median_layer = simulate_combined_threshold_layer(layer, layer_cells, WaveformType.MONOPHASIC, rotation_count)
    result_median_layer.write(
        str(output_path.joinpath(f'layer_{int(layer_depth[1] * 100)}_{rotation_count}_test.msh').resolve()))
    print(f'Calculated {len(layer_cells) * rotation_count} '
          f'threshold(s) per layer element in {datetime.timedelta(seconds=time.perf_counter() - timeStart)}')
