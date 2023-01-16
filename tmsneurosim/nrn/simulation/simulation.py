import pathlib
from enum import Enum

import numpy as np
from neuron import h
from scipy.io import loadmat

import tmsneurosim
from tmsneurosim.nrn.cells import NeuronCell


class WaveformType(Enum):
    MONOPHASIC = 1
    BIPHASIC = 2


class Simulation:
    """ Wrapper to set up, modify and execute a NEURON simulation of a single cell.

     Attributes:
            neuron_cell (NeuronCell): The cell that is supposed to be simulated
            stimulation_delay (float): Initial delay before the activation waveform is applied in s
            simulation_temperature (float): Temperature for the simulation in degree Celsius
            simulation_time_step (float): The time step used for the simulation in ms.
            simulation_duration (float): The duration of the simulation in ms.
            waveform ([float]): The amplitude values of the waveform used.
            waveform_time ([float]): The time values of the waveform used.
    """
    INITIAL_VOLTAGE = -70

    def __init__(self, neuron_cell: NeuronCell, waveform_type: WaveformType):
        """
        Initializes the simulation with the transmitted neuron cell and waveform type.
        :param neuron_cell: The neuron that is supposed to be used in the NEURON simulation.
        :param waveform_type: The waveform type that is supposed to be used in the NEURON simulation
        """
        self.neuron_cell = neuron_cell
        self._action_potentials = h.Vector()
        self._action_potentials_recording_ids = h.Vector()

        self.stimulation_delay = 0.005
        self.simulation_temperature = 37
        self.simulation_time_step = 0.005
        self.simulation_duration = 1.0
        self.waveform, self.waveform_time = self._load_waveform(waveform_type)

        self.init_handler = None
        self.init_state = None
        self.attached = False

    def attach(self):
        """
        Attaches spike recording to the neuron and connects the simulation initialization
        methode to the global NEURON space.
        """
        self._init_spike_recording()
        self.init_handler = h.FInitializeHandler(2, self._post_finitialize)

        self.attached = True

    def detach(self):
        """ Removes the spike recording from the neuron and disconnects the initialization methode.
        """
        for net in self.netcons:
            net.record()
        self.netcons.clear()
        del self.init_handler
        self.init_handler = None

        self.attached = False

    def _post_finitialize(self):
        """ Initialization methode to unsure a steady state before the actual simulation is started.
        """
        temp_dt = h.dt

        h.t = -1e11
        h.dt = 1e9

        while h.t < - h.dt:
            h.fadvance()

        h.dt = temp_dt
        h.t = 0
        h.fcurrent()
        h.frecord_init()

    def _init_spike_recording(self):
        """Initializes spike recording for every segment of the neuron.
        """
        self.netcons = []
        for i, section in enumerate(self.neuron_cell.all):
            for segment in section:
                recording_netcon = h.NetCon(segment._ref_v, None, sec=section)
                recording_netcon.threshold = 0
                recording_netcon.record(self._action_potentials, self._action_potentials_recording_ids, i)
                self.netcons.append(recording_netcon)

    def _load_waveform(self, waveform_type: WaveformType):
        """Loads the submitted waveform and modifies it to fit the simulation settings.
        """
        tms_waves = loadmat(
            str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath('coil_recordings/TMSwaves.mat').absolute()))

        recorded_time = tms_waves['tm'].ravel()
        recorded_e_field_magnitude = tms_waves['Erec_m']

        if waveform_type is WaveformType.BIPHASIC:
            recorded_e_field_magnitude = tms_waves['Erec_b']

        sample_factor = int(self.simulation_time_step / np.mean(np.diff(recorded_time)))
        if sample_factor < 1:
            sample_factor = 1

        simulation_time = recorded_time[::sample_factor]
        simulation_e_field_magnitude = np.append(recorded_e_field_magnitude[::sample_factor], 0)

        if self.stimulation_delay >= self.simulation_time_step:
            simulation_time = np.concatenate(
                (np.array([0, self.stimulation_delay - self.simulation_time_step]),
                 simulation_time + self.stimulation_delay))
            simulation_e_field_magnitude = np.concatenate(
                (np.array([0, 0]), np.append(recorded_e_field_magnitude[::sample_factor], 0)))

        simulation_time = np.append(np.concatenate(
            (simulation_time, np.arange(simulation_time[-1] + self.simulation_time_step, self.simulation_duration,
                                        self.simulation_time_step))),
            self.simulation_duration)

        if len(simulation_time) > len(simulation_e_field_magnitude):
            simulation_e_field_magnitude = np.pad(simulation_e_field_magnitude,
                                                  (0, len(simulation_time) - len(simulation_e_field_magnitude)),
                                                  constant_values=(0, 0))
        else:
            simulation_e_field_magnitude = simulation_e_field_magnitude[:len(simulation_time)]

        return simulation_e_field_magnitude, simulation_time
