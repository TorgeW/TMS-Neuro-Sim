import pathlib
import subprocess

import neuron

from ..nrn import __file__

if not neuron.load_mechanisms(str(pathlib.Path(__file__).parent.joinpath('mechanisms').resolve()),
                              warn_if_already_loaded=False):
    print('NEURON compile mechanisms (Only on first load)')
    n = subprocess.Popen(['nrnivmodl'], cwd=pathlib.Path(__file__).parent.joinpath('mechanisms'),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n.wait()
    neuron.load_mechanisms(str(pathlib.Path(__file__).parent.joinpath('mechanisms').resolve()))
    print('NEURON mechanisms loaded')

neuron.h.load_file('stdrun.hoc')
