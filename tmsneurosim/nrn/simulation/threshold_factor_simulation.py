import numpy as np
from neuron import h

from tmsneurosim.nrn.cells import NeuronCell
from tmsneurosim.nrn.simulation.simulation import Simulation, WaveformType


class ThresholdFactorSimulation(Simulation):
    """
    A NEURON simulation with the functionality to run simulations and to find a threshold factor that is the minimum
    factor to trigger an action potential.
    """

    def __init__(self, neuron_cell: NeuronCell, waveform_type: WaveformType):
        super().__init__(neuron_cell, waveform_type)

    def simulate(self, stimulation_amplitude: float):
        """
        Executes a NEURON simulation with the submitted amplitude as the scaling factor for the E-field.
        :param stimulation_amplitude:
        :return:
        """
        if self.init_state is None:
            h.celsius = self.simulation_temperature
            h.dt = self.simulation_time_step
            h.tstop = self.simulation_duration
            h.finitialize(self.INITIAL_VOLTAGE)
            self.init_state = h.SaveState()
            self.init_state.save()
        else:
            self.init_state.restore()

        waveform_vector = h.Vector(self.waveform * stimulation_amplitude)
        waveform_time_vector = h.Vector(self.waveform_time)

        waveform_vector.play(h._ref_stim_xtra, waveform_time_vector, 1)

        h.run()

    def find_threshold_factor(self):
        """
        Searches for the minimal threshold factor to trigger an action potential in the simulated neuron.
        :return: The threshold factor that was found.
        """
        if not self.attached:
            raise ValueError('Simulation is not attached')
        if not self.neuron_cell.loaded:
            raise ValueError('Neuron cell is not loaded')
        low = 0
        high = 1e5
        amplitude = 100
        epsilon = 1e-8 + 5e-2

        while low <= 0 or high >= 1e5:
            self.simulate(amplitude)
            if self._action_potentials.size() >= 3:
                high = amplitude
                amplitude = amplitude / 2
            else:
                low = amplitude
                amplitude = amplitude * 2
            if low > high:
                return np.inf
            if high < 0.000001:
                return 0

        amplitude = (high + low) / 2

        while high - low > epsilon:
            self.simulate(amplitude)
            if self._action_potentials.size() >= 3:
                high = amplitude
            else:
                low = amplitude
            amplitude = (high + low) / 2
        return high
