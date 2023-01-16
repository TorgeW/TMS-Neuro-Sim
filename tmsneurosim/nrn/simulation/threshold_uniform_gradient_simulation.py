import math

import numpy as np

from tmsneurosim.nrn.simulation.threshold_factor_simulation import ThresholdFactorSimulation


class ThresholdUniformGradientSimulation(ThresholdFactorSimulation):
    """
    A threshold factor NEURON simulation that has functionality to apply an artificial parameter based E-field at each
    segment.
    """

    def apply_e_field(self, e_field_theta, e_field_phi, relative_gradient_per_mm, phi_gradient_per_mm,
                      theta_gradient_per_mm):
        """
        Applies an artificial parameter based E-field at each segment by calculating the quasi potentials from it.
        :param e_field_theta: The polar angle of the E-field in relation to the somatodentritic axes of the neuron.
        :param e_field_phi: The azimuthal angle of the E-field in relation to the
        :param relative_gradient_per_mm: The relative change of the E-field magnitude over the course of the
        somatodentritic axes of the neuron.
        :param phi_gradient_per_mm: The relative change of the E-field azimuthal angle over the course of the
        somatodentritic axes of the neuron.
        :param theta_gradient_per_mm: The relative change of the E-field polar angle over the course of the
        somatodentritic axes of the neuron.
        """
        cell_segment_coordinates = self.neuron_cell.get_segment_coordinates()

        soma_position_z = self.neuron_cell.soma[0](0.5).z_xtra
        min_z = np.amin(cell_segment_coordinates[:, 2])
        max_z = np.amax(cell_segment_coordinates[:, 2])
        soma_distance_to_min = math.fabs(soma_position_z - min_z)
        soma_distance_to_max = math.fabs(soma_position_z - max_z)

        magnitude_at_max = 1 + soma_distance_to_max / 1000 * ((-relative_gradient_per_mm) / 100)
        magnitude_at_min = 1 + soma_distance_to_min / 1000 * (relative_gradient_per_mm / 100)

        phi_at_max = soma_distance_to_max / 1000 * phi_gradient_per_mm
        phi_at_min = soma_distance_to_min / 1000 * (-phi_gradient_per_mm)

        theta_at_max = soma_distance_to_max / 1000 * theta_gradient_per_mm
        theta_at_min = soma_distance_to_min / 1000 * (-theta_gradient_per_mm)

        i = 0
        for section in self.neuron_cell.all:
            for segment in section:
                gradient_phi = np.interp(cell_segment_coordinates[i][2],
                                         [min_z, soma_position_z, max_z],
                                         [phi_at_min, 0, phi_at_max])
                phi = math.radians(e_field_phi + gradient_phi)

                gradient_theta = np.interp(cell_segment_coordinates[i][2],
                                           [min_z, soma_position_z, max_z],
                                           [theta_at_min, 0, theta_at_max])
                theta = math.radians(e_field_theta + gradient_theta)
                e_field_direction = np.array(
                    [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])

                gradient_magnitude = max(np.interp(cell_segment_coordinates[i][2],
                                                   [min_z, soma_position_z, max_z],
                                                   [magnitude_at_min, 1, magnitude_at_max]), 0)

                e_field_at_segment = e_field_direction * gradient_magnitude
                segment.Ex_xtra = e_field_at_segment[0]
                segment.Ey_xtra = e_field_at_segment[1]
                segment.Ez_xtra = e_field_at_segment[2]
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
