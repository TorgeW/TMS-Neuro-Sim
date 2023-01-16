import datetime
import os
import pathlib
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from tmsneurosim.simulate_threshold_map import simulate_gradient_threshold_map
from tmsneurosim.nrn import cells
from tmsneurosim.nrn.simulation.simulation import WaveformType

if __name__ == '__main__':
    example_path = os.path.dirname(os.path.abspath(__file__))
    output_path = pathlib.Path(example_path).joinpath('output')

    for n1 in [cells.L6_TPC_L4_cADpyr(morph_id) for morph_id in range(1, 6)]:
        timeStart = time.perf_counter()
        theta_step = 15
        phi_step = 10
        threshold_map, e_field_list = \
            simulate_gradient_threshold_map(n1, WaveformType.MONOPHASIC, theta_step, phi_step, process_count=6)
        with open(output_path.joinpath(f'{n1.__class__.__name__}_{n1.morphology_id}_{theta_step}_{phi_step}.csv'),
                  'w') as f:
            for item, (theta, phi, magnitude_gradient, phi_gradient, theta_gradient) in zip(threshold_map,
                                                                                            e_field_list):
                f.write(f'{theta},{phi},{item}\n')
        print(
            f'Calculated {len(threshold_map)} thresholds in {datetime.timedelta(seconds=time.perf_counter() - timeStart)}')
