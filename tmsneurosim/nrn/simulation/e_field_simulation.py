from tmsneurosim.nrn.cells import NeuronCell
from tmsneurosim.nrn.simulation.simulation import WaveformType
from tmsneurosim.nrn.simulation.threshold_factor_simulation import ThresholdFactorSimulation


class EFieldSimulation(ThresholdFactorSimulation):
    """
    A threshold factor NEURON simulation that has functionality to apply an E-field at each segment.
    """

    def __init__(self, neuron_cell: NeuronCell, waveform_type: WaveformType, e_field_at_segments=None):
        super().__init__(neuron_cell, waveform_type)
        if e_field_at_segments is not None:
            self.apply_e_field(e_field_at_segments)

    def apply_e_field(self, e_field_at_segments):
        """
        Applies the submitted E-field vectors at each segment by calculating the quasi potentials from them.
        :param e_field_at_segments: E-field vectors for each segment.
        """
        self.init_state = None
        i = 0
        for section in self.neuron_cell.all:
            for segment in section:
                segment.Ex_xtra = e_field_at_segments[i][0]
                segment.Ey_xtra = e_field_at_segments[i][1]
                segment.Ez_xtra = e_field_at_segments[i][2]
                i += 1

        current_section = self.neuron_cell.soma[0]
        current_section.es_xtra = 0
        section_stack = list(current_section.children())
        parent_stack = [current_section(1)] * len(section_stack)

        while len(section_stack) > 0:
            current_section = section_stack.pop()
            segments = list(current_section)
            parent_segment = parent_stack.pop()

            for segment in segments:
                segment.es_xtra = parent_segment.es_xtra - 0.5 * 1e-3 * \
                                  ((parent_segment.Ex_xtra + segment.Ex_xtra) * (segment.x_xtra - parent_segment.x_xtra)
                                   + (parent_segment.Ey_xtra + segment.Ey_xtra) * (
                                           segment.y_xtra - parent_segment.y_xtra)
                                   + (parent_segment.Ez_xtra + segment.Ez_xtra) * (
                                           segment.z_xtra - parent_segment.z_xtra))
                parent_segment = segment

            children = list(current_section.children())
            section_stack += children
            parent_stack += [parent_segment] * len(children)
