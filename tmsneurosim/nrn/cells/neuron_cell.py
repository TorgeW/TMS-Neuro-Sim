import math
import math
import pathlib
import typing

import nrn
import numpy as np
from matplotlib import pyplot, pyplot as plt
from matplotlib.lines import Line2D
from neuron import h
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.spatial.transform import Rotation

from tmsneurosim.nrn import cells
from tmsneurosim.nrn.cells.cell_modification_parameters.cell_modification_parameters import CellModificationParameters, \
    AxonModificationMode
from tmsneurosim.nrn.cells.cell_modification_parameters.maxh_modification_parameters import MaxhModificationParameters


class NeuronCell:
    """Holds information as well as sections of a specific NEURON cell.

    Attributes:
        all ([nrn.Section]): Contains all sections that belong to this cell
        axon ([nrn.Section]): Contains all axonal sections that belong to this cell
        myelin ([nrn.Section]): Contains all myelinated axonal sections that belong to this cell (Subset of axon)
        node ([nrn.Section]): Contains all axonal node sections that belong to this cell (Subset of axon)
        unmyelin ([nrn.Section]): Contains all unmyelinated axonal sections that belong to this cell (Subset of axon)
        dend ([nrn.Section]): Contains all dendrite sections that belong to this cell
        soma ([nrn.Section]): Contains all somatic sections that belong to this cell
        apic ([nrn.Section]): Contains all apical sections that belong to this cell
        direction (NDArray[float]): The main cell direction pointing upwards from the soma
        variation_seed (int): The seed for the random number generator to create a varied cell
        morphology_path (str): Path to the morphology file (.asc) of the cell
        modification_parameters (CellModificationParameters):
            Parameters for cell modification like scaling and myelination
        loaded (bool): True if cell is currently loaded into the NEURON global space, False otherwise
    """

    MIN_PRE_MYELIN_AXON_LENGTH = 50
    NODE_LENGTH = 1

    def __init__(self, morphology_path: str, morphology_id: int,
                 modification_parameters: CellModificationParameters = None,
                 variation_seed=None):
        """Initializes but not loads a neuron cell. To load the cell into NEURON global space call load().

        :param morphology_path: Path to the morphology file (.asc) of the cell
        :param modification_parameters: (Optional) Parameters for cell modification like scaling and myelination
        :param variation_seed: (Optional) If not none this seed is used to vary the cell on load
        """
        self.all = []
        self.axon = []
        self.myelin = []
        self.node = []
        self.unmyelin = []
        self.dend = []  # basal
        self.soma = []
        self.apic = []

        self.direction = np.array([0, 1, 0])

        if modification_parameters is None:
            modification_parameters = MaxhModificationParameters()

        self.variation_seed = variation_seed

        self.morphology_path = morphology_path
        self.morphology_id = morphology_id
        self.modification_parameters = modification_parameters
        self.loaded = False

    def apply_biophysics(self) -> None:
        """ Applies biophysical properties to the sections of the cell
        """
        pass

    def load(self, modification_parameters: CellModificationParameters = None) -> None:
        """ Loads the sections of this cell into the global NEURON space. If the cell is already loaded it is reloaded.

        :param modification_parameters: (optional) Parameters for cell modification like scaling and myelination
        """
        if self.loaded:
            self.unload()

        if modification_parameters is not None:
            self.modification_parameters = modification_parameters

        h.load_file("import3d.hoc")
        nl = h.Import3d_Neurolucida3()
        nl.quiet = 1
        nl.input(self.morphology_path)
        import_neuron = h.Import3d_GUI(nl, 0)
        import_neuron.instantiate(self)

        if self.variation_seed is not None:
            self._variegate(self.variation_seed)

        self._set_segment_count(self.all, 40)

        self.apply_biophysics()

        self._scale_soma_area(self.modification_parameters.soma_area_scaling_factor)
        self._scale_section_point_diameters(self.modification_parameters.apic_diameter_scaling_factor, self.apic)
        self._scale_section_point_diameters(self.modification_parameters.dend_diameter_scaling_factor, self.dend)
        self._scale_section_length(self.modification_parameters.dend_length_scaling_factor, self.dend)
        self._scale_section_point_diameters(self.modification_parameters.axon_diameter_scaling_factor, self.axon)

        self._rotate_x_90()

        if self.modification_parameters.axon_modification_mode == AxonModificationMode.MYELINATED_AXON:
            self._myelinate_axon()

        self.disable_main_axon_terminal()

        self.loaded = True

    def _variegate(self, variation_seed=None) -> None:
        """Varies the current cell.

        :param variation_seed: (Optional) seed for the random number generator
        """
        rng = np.random.RandomState(variation_seed)
        for section in self.axon + self.dend + self.apic:
            section.L *= rng.normal(1, .2)
            h.define_shape()

            if len(section.children()) > 1:
                i = section.n3d() - 1
                bif_pt = np.array([section.x3d(i), section.y3d(i), section.z3d(i)])

                for child in section.children():
                    x = [[s.x3d(i), s.y3d(i), s.z3d(i)] for s in child.subtree() for i in range(s.n3d())]
                    principal_comp = svd((np.array(x) - bif_pt).T)[0][:, 0]  # principal component direction

                    angle = rng.normal(0, np.pi / 18)
                    for subsec in child.subtree():
                        old_x = np.array([[subsec.x3d(i), subsec.y3d(i), subsec.z3d(i)] for i in range(subsec.n3d())])
                        new_x = Rotation.from_rotvec(angle * principal_comp).apply(old_x - bif_pt) + bif_pt

                        for i in range(subsec.n3d()):
                            subsec.pt3dchange(i, new_x[i, 0], new_x[i, 1], new_x[i, 2], subsec.diam3d(i))

    @staticmethod
    def _set_segment_count(section_list: [nrn.Section], chunk_size: float) -> None:
        """ Sets the segment count of each section in section_list so that each segment has a maximum length of
        chunk_size. For every chunk_size, 2 segments are added.

        :param section_list: The list of sections whose segment number should be updated
        :param chunk_size: The chunk size in microns
        """
        for section in section_list:
            section.nseg = 1 + 2 * int(section.L / chunk_size)

    def unload(self) -> None:
        """Unloads the sections of this cell from the NEURON global space
        """
        for section in self.all:
            h.delete_section(sec=section)
        self.all.clear()
        self.axon.clear()
        self.dend.clear()
        self.soma.clear()
        self.apic.clear()
        self.myelin.clear()
        self.node.clear()
        self.unmyelin.clear()
        self.direction = np.array([0, 1, 0])
        self.loaded = False

    @staticmethod
    def _scale_section_point_diameters(scale_factor: float, sections: [nrn.Section]) -> None:
        """Scales the point diameters of the sections by the scale_factor.

        :param scale_factor: Scale factor for the point diameters
        :param sections: The sections whose point diameters are supposed to be scaled
        """
        if not np.isclose(scale_factor, 1):
            for section in sections:
                for point_index in range(section.n3d()):
                    section.pt3dchange(point_index, section.diam3d(point_index) * scale_factor)

    @staticmethod
    def _scale_section_length(scale_factor: float, sections: [nrn.Section]) -> None:
        """Scales the length of the sections by the scale_factor.

        :param scale_factor: Scale factor for the section length
        :param sections: The sections whose length is supposed to be scaled
        """
        if not np.isclose(scale_factor, 1):
            for section in sections:
                section.L = section.L * scale_factor

    def _scale_soma_area(self, scale_factor: float) -> None:
        """ Scales the area of the soma by the scale_factor.

        :param scale_factor: Scale factor for the soma area
        """
        if not np.isclose(scale_factor, 1):
            self.soma[0](0.5).diam = scale_factor * self.soma[0](0.5).area() / (math.pi * self.soma[0].L)

    def _rotate_x_90(self) -> None:
        """ Rotates the cell by 90 degree around the x-axis
        """
        for section in self.all:
            for i in range(section.n3d()):
                section.pt3dchange(i, section.x3d(i), -section.z3d(i), section.y3d(i), section.diam3d(i))
        self._setup_xtra()
        self.direction = np.array([self.direction[0], -self.direction[2], self.direction[1]])

    def _myelinate_axon(self) -> None:
        """ Replaces the axon sections with myelin, node and unmyelin sections.
        """
        if self.axon[0].L >= self.MIN_PRE_MYELIN_AXON_LENGTH + self.modification_parameters.min_myelin_length:
            self._seperate_pre_myelin_axon()

        axon_without_root = self.axon[1:]

        axon_section_myelin_counts, axon_section_myelin_section_count, unmyelin_section_count \
            = self._calculate_myelin_section_counts(axon_without_root)

        myelin = [h.Section(name=f'Myelin[{n}]', cell=self) for n in range(axon_section_myelin_section_count)]
        nodes = [h.Section(name=f'Node[{n}]', cell=self) for n in range(axon_section_myelin_section_count)]
        unmyelin = [h.Section(name=f'Unmyelin[{n}]', cell=self) for n in range(unmyelin_section_count)]

        myelinated_index = 0
        node_index = 0
        unmyelinated_index = 0
        for i, axon_section in enumerate(axon_without_root):
            axon_section_children = list(axon_section.children())
            axon_section_parent = axon_section.trueparentseg().sec
            axon_section_myelin_section_count = axon_section_myelin_counts[i]
            if axon_section_myelin_section_count == 0:
                unmyelin[unmyelinated_index].connect(axon_section_parent(1), 0)
                for point_index in range(axon_section.n3d()):
                    unmyelin[unmyelinated_index].pt3dadd(axon_section.x3d(point_index), axon_section.y3d(point_index),
                                                         axon_section.z3d(point_index),
                                                         axon_section.diam3d(point_index))
                for axon_section_child in axon_section_children:
                    h.disconnect(sec=axon_section_child)
                    axon_section_child.connect(unmyelin[unmyelinated_index](1), 0)
                unmyelinated_index += 1
            else:
                axon_section.pt3dchange(0, axon_section_parent.x3d(axon_section_parent.n3d() - 1),
                                        axon_section_parent.y3d(axon_section_parent.n3d() - 1),
                                        axon_section_parent.z3d(axon_section_parent.n3d() - 1),
                                        axon_section_parent.diam3d(axon_section_parent.n3d() - 1))

                myelin_length = (axon_section.L - axon_section_myelin_section_count
                                 * self.NODE_LENGTH) / axon_section_myelin_section_count
                current_new_section = axon_section_parent
                current_section_start = 0
                point_index = 0
                for myelin_section_index in range(axon_section_myelin_section_count * 2):
                    previous_section = current_new_section
                    if myelinated_index != node_index:
                        current_new_section = nodes[node_index]
                        node_index += 1
                        current_new_section_length = self.NODE_LENGTH
                    else:
                        current_new_section = myelin[myelinated_index]
                        myelinated_index += 1
                        current_new_section_length = myelin_length
                    current_new_section.pt3dadd(previous_section.x3d(previous_section.n3d() - 1),
                                                previous_section.y3d(previous_section.n3d() - 1),
                                                previous_section.z3d(previous_section.n3d() - 1),
                                                previous_section.diam3d(previous_section.n3d() - 1))
                    current_new_section.connect(previous_section(1), 0)

                    while point_index < axon_section.n3d() and axon_section.arc3d(
                            point_index) - current_section_start < current_new_section_length:
                        current_new_section.pt3dadd(axon_section.x3d(point_index), axon_section.y3d(point_index),
                                                    axon_section.z3d(point_index), axon_section.diam3d(point_index))
                        point_index += 1

                    if point_index < axon_section.n3d():
                        length_left = current_new_section_length - current_new_section.L
                        current_new_section.pt3dadd(*self._interpolate_point_between(
                            length_left, current_new_section, current_new_section.n3d() - 1, axon_section, point_index))

                    current_section_start += current_new_section.L

                for axon_section_child in axon_section_children:
                    h.disconnect(sec=axon_section_child)
                    axon_section_child.connect(nodes[node_index - 1](1), 0)

        for axon_section in axon_without_root:
            self.all.remove(axon_section)
            self.axon.remove(axon_section)
            h.delete_section(sec=axon_section)

        for myelin_section in myelin:
            self.all.append(myelin_section)
            self.axon.append(myelin_section)
            self.myelin.append(myelin_section)
        for node_section in nodes:
            self.all.append(node_section)
            self.axon.append(node_section)
            self.node.append(node_section)
        for unmyelin_section in unmyelin:
            self.all.append(unmyelin_section)
            self.axon.append(unmyelin_section)
            self.unmyelin.append(unmyelin_section)

        for myelin_section in self.myelin:
            for point_index in range(myelin_section.n3d()):
                diam = myelin_section.diam3d(point_index)
                g_ratio = math.pow(diam, 3) * 0.6425
                g_ratio += math.pow(diam, 2) * -1.778
                g_ratio += diam * 1.749
                g_ratio += 0.1518
                g_ratio = max(min(g_ratio, 0.8), 0.4)
                myelin_section.pt3dchange(point_index, diam * (1 / g_ratio))

        self._apply_myelin_biophysics()
        self._set_segment_count(self.axon, 40)
        self._setup_xtra()

    def _calculate_myelin_section_counts(self, axon_without_root: [nrn.Section]) -> ([int], int, int):
        """Calculates how many myelin, node, unmyelin sections fit in each axon_without_root section.

        :param axon_without_root: The sections which length is used to determin the amount of myelin,
            node and unmyelin sections fit into them
        :return: A tuple containing a list of myelin/node section count for each axon_without_root section,
            the total count of myelin/node sections and the total count of unmyelin sections
        """
        axon_section_myelin_section_count = 0
        unmyelin_section_count = 0
        axon_section_myelin_counts = []
        for axon_section in axon_without_root:
            myelin_length = axon_section(0).diam * 100
            axon_section_myelin_count = 0
            if axon_section(1).type_xtra == 2 or axon_section(1).type_xtra == 5:
                myelin_length = axon_section(0).diam * 70
            if axon_section(0).diam >= self.modification_parameters.min_myelin_diameter:
                axon_section_myelin_count += int(axon_section.L / (myelin_length + self.NODE_LENGTH))
                if axon_section_myelin_count == 0:
                    axon_section_myelin_count += int(
                        axon_section.L / (self.modification_parameters.min_myelin_length + self.NODE_LENGTH))
            axon_section_myelin_section_count += axon_section_myelin_count
            if axon_section_myelin_count == 0:
                unmyelin_section_count += 1
            axon_section_myelin_counts.append(axon_section_myelin_count)
        return axon_section_myelin_counts, axon_section_myelin_section_count, unmyelin_section_count

    def _seperate_pre_myelin_axon(self) -> None:
        """ Splits the initial axon section into two axon sections, reducing the size of the initial axon section
        to MIN_PRE_MYELIN_AXON_LENGTH.
        """
        new_axon_section = h.Section(name=f'axon[0_1]', cell=self)
        new_axon_section.insert("xtra")
        new_axon_section.type_xtra = self.axon[0].type_xtra
        point_index = 0
        for i in range(self.axon[0].n3d()):
            if self.axon[0].arc3d(point_index) > self.MIN_PRE_MYELIN_AXON_LENGTH:
                new_axon_section.pt3dadd(self.axon[0].x3d(point_index), self.axon[0].y3d(point_index),
                                         self.axon[0].z3d(point_index), self.axon[0].diam3d(point_index))
                self.axon[0].pt3dremove(point_index)
            else:
                point_index += 1
        length_left = self.MIN_PRE_MYELIN_AXON_LENGTH - self.axon[0].L
        self.axon[0].pt3dadd(*self._interpolate_point_between(
            length_left, self.axon[0], self.axon[0].n3d() - 1, new_axon_section, 0))
        new_axon_section.pt3dinsert(0, self.axon[0].x3d(self.axon[0].n3d() - 1),
                                    self.axon[0].y3d(self.axon[0].n3d() - 1),
                                    self.axon[0].z3d(self.axon[0].n3d() - 1),
                                    self.axon[0].diam3d(self.axon[0].n3d() - 1))
        axon_section_children = list(self.axon[0].children())
        for axon_section_child in axon_section_children:
            h.disconnect(sec=axon_section_child)
            axon_section_child.connect(new_axon_section(1), 0)
        new_axon_section.connect(self.axon[0](1), 0)
        self.axon.insert(1, new_axon_section)
        self.all.append(new_axon_section)

    @staticmethod
    def _interpolate_point_between(distance: float, section_end: nrn.Section, section_end_index: int,
                                   section_start: nrn.Section,
                                   section_start_index: int) -> (float, float, float, float):
        """ Interpolates the point between the section_end point with index section_end_index and the
        section_start point with index section_start_index. The point is distance away from section_end point with index
        section_end_index.

        :param distance: The distance the point is away from section_end point with index section_end_index
        :param section_end: The section containing the first point for the interpolation
        :param section_end_index: The index to the first point of the interpolation in section_end
        :param section_start: The section containing the second point for the interpolation
        :param section_start_index: The index to the first point of the interpolation in section_end
        :return: A tuple containing the x,y,z coordinates as well as the diameter of the interpolated point
        """
        x_displacement = section_start.x3d(section_start_index) - section_end.x3d(section_end_index)
        y_displacement = section_start.y3d(section_start_index) - section_end.y3d(section_end_index)
        z_displacement = section_start.z3d(section_start_index) - section_end.z3d(section_end_index)
        point_distance = math.sqrt(x_displacement ** 2 + y_displacement ** 2 + z_displacement ** 2)

        return section_end.x3d(section_end_index) + distance * x_displacement / point_distance, \
               section_end.y3d(section_end_index) + distance * y_displacement / point_distance, \
               section_end.z3d(section_end_index) + distance * z_displacement / point_distance, \
               section_end.diam3d(section_end_index)

    def _apply_myelin_biophysics(self) -> None:
        """ Applies biophysical properties to the myelin, node and unmyelin sections of the cell.
        Also changes capacitance of the sima, apic and dend sections.
        """
        self.apply_biophysics()
        for soma_section in self.soma:
            soma_section.cm = 1
        for apic_section in self.apic:
            apic_section.cm = 2
        for dend_section in self.dend:
            dend_section.cm = 2

        node_sections_mechanisms = {'pas', 'SKv3_1', 'Nap_Et2', 'K_Pst', 'K_Tst', 'NaTa_t'}
        for node_section in self.node:
            for mechanism in node_section.psection()['density_mechs']:
                if mechanism not in node_sections_mechanisms:
                    node_section.uninsert(mechanism)
            for segment in node_section:
                segment.gNaTa_tbar_NaTa_t = segment.gNaTa_tbar_NaTa_t * 2

        myelin_sections_mechanisms = {'pas'}
        for myelin_section in self.myelin:
            for mechanism in myelin_section.psection()['density_mechs']:
                if mechanism not in myelin_sections_mechanisms:
                    myelin_section.uninsert(mechanism)
            myelin_section.cm = 0.02
            myelin_section.g_pas = 1 / 1.125e6

        unmyelin_sections_mechanisms = {'pas', 'SKv3_1', 'Nap_Et2', 'K_Pst', 'K_Tst', 'NaTa_t'}
        for unmyelin_section in self.unmyelin:
            for mechanism in unmyelin_section.psection()['density_mechs']:
                if mechanism not in unmyelin_sections_mechanisms:
                    unmyelin_section.uninsert(mechanism)

    def disable_main_axon_terminal(self):
        """Disables the main axon terminal of the neuron.
        """
        if isinstance(self, cells.L5_TTPC2_cADpyr):
            dont_disable_axon_ids = [5, 12521, 12530, 12535, 12570, 12587, 12593, 12603]
            if self.morphology_id not in dont_disable_axon_ids:
                self.disable_main_axon_terminal_thickest()
        if isinstance(self, cells.L6_TPC_L4_cADpyr):
            disable_furthest_axon_ids = [2, 5]
            if self.morphology_id in disable_furthest_axon_ids:
                self.disable_main_axon_furthest()
            else:
                self.disable_main_axon_terminal_thickest()

    def disable_main_axon_terminal_thickest(self):
        """Finds the main axon based on the thickest child of each axon section and disables it.
        """
        axon_section = self.axon[0]
        while True:
            axon_children = list(axon_section.children())
            if len(axon_children) > 0:
                axon_section = axon_children[0]
                for children in axon_children:
                    if children(0).diam > axon_section(0).diam:
                        axon_section = children
            else:
                break
        axon_section(1).diam = 1000

    def disable_main_axon_furthest(self):
        """Finds the main axon terminal based on the axon terminal furthest down from the soma and disables it.
        """
        main_axon_xy_radius = 500

        min_axon_section = self.axon[0]
        min_axon_point_index = 0
        for section in self.axon + self.myelin + self.unmyelin + self.node:
            for point_index in range(section.n3d()):
                if section.z3d(point_index) < min_axon_section.z3d(min_axon_point_index):
                    axon_xy_distance = np.linalg.norm(np.array([self.soma[0](0).x_xtra, self.soma[0](0).y_xtra] -
                                                               np.array([section.x3d(point_index),
                                                                         section.y3d(point_index)])))
                    if axon_xy_distance < main_axon_xy_radius:
                        min_axon_section = section
                        min_axon_point_index = point_index
        min_axon_section(1).diam = 1000

    def _setup_xtra(self) -> None:
        """ Sets up xtra_coordinates, xtra_type, xtra_order and xtra_pointer variables.
        """
        for sec in self.all:
            sec.insert('xtra')
            sec.insert('extracellular')
        self._assign_xtra_coordinates()
        self._assign_xtra_type()
        self._assign_xtra_order()
        self._assign_xtra_pointer()

    def _assign_xtra_coordinates(self) -> None:
        """ Interpolates the coordinates of the segment centers from the section points. Calculates the direction
        vectors to the parent section. Applies these values to the xtra variables.
        """
        for section in self.all:
            if section.has_membrane('xtra'):
                segments = list(section)
                segment_geometries = np.zeros((section.n3d(), 4))
                for i in range(section.n3d()):
                    segment_geometries[i, 0] = section.x3d(i)
                    segment_geometries[i, 1] = section.y3d(i)
                    segment_geometries[i, 2] = section.z3d(i)
                    segment_geometries[i, 3] = section.arc3d(i)

                segment_geometries[:, 3] /= segment_geometries[-1, 3]

                segment_coordinates = np.array([segment.x for segment in segments])

                x_interpolated = np.interp(segment_coordinates, segment_geometries[:, 3], segment_geometries[:, 0])
                y_interpolated = np.interp(segment_coordinates, segment_geometries[:, 3], segment_geometries[:, 1])
                z_interpolated = np.interp(segment_coordinates, segment_geometries[:, 3], segment_geometries[:, 2])

                segments[0].x_xtra = x_interpolated[0]
                segments[0].y_xtra = y_interpolated[0]
                segments[0].z_xtra = z_interpolated[0]

                if section.nseg == 1:
                    segments[0].Dx_xtra = section.x3d(section.n3d() - 1) - section.x3d(0)
                    segments[0].Dy_xtra = section.y3d(section.n3d() - 1) - section.y3d(0)
                    segments[0].Dz_xtra = section.z3d(section.n3d() - 1) - section.z3d(0)
                else:
                    segments[0].Dx_xtra = segments[0].x_xtra - section.x3d(0)
                    segments[0].Dy_xtra = segments[0].y_xtra - section.y3d(0)
                    segments[0].Dz_xtra = segments[0].z_xtra - section.z3d(0)

                for previous_segment, segment, x_interpolation, y_interpolation, z_interpolation \
                        in zip(segments, segments[1:], x_interpolated[1:], y_interpolated[1:], z_interpolated[1:]):
                    segment.x_xtra = x_interpolation
                    segment.y_xtra = y_interpolation
                    segment.z_xtra = z_interpolation

                    segment.Dx_xtra = segment.x_xtra - previous_segment.x_xtra
                    segment.Dy_xtra = segment.y_xtra - previous_segment.y_xtra
                    segment.Dz_xtra = segment.z_xtra - previous_segment.z_xtra

    def _assign_xtra_type(self) -> None:
        """ Assigns type of segments by the following rules:
        1 = soma (no parent)
        2 = termination from non-bifurcation (parent is a 1,3, or 6) OR far from bifurcation
        (parent is 4 or 7, L > length constant) ** contains termination point (1) **
        3 = intermediate from non-bifurcation (parent is 1,3, or 6)
        4 = parent side of bifurcation (child is a 5,6, or 7) (parent is 3 OR far from bifurcation (4 or 7))
        ** contains parent side bifurcation point (1)**
        5 = child termination from bifurcation (parent is a 1 or 4 and L < length constant)
        ** contains termination point (1)**
        6 = child side of bifurcation (parent is a 1 or 4) **contains child side bifurcation point (0)**
        7 = parent bifurcation and child of bifurcation (parent is a 1 or 4, child is a 5,6, or 7)
        ** contains parent side bifurcation point (1) **
        """
        # Function dosn't work as it seams but matches the original functionality
        for section in self.all:
            if section.trueparentseg() is None:
                section.type_xtra = 1
            else:
                parent_children = list(section.trueparentseg().sec.children())
                children = list(section.children())
                section_length = section.L
                section_lambda = section.L
                if section.has_membrane("pas"):
                    section_lambda = 50 * math.sqrt(section(1).diam / (section(1).g_pas * section.Ra))
                for segment in section:
                    segment.type_xtra = 3
                if len(parent_children) == 1:
                    if len(children) == 0:
                        section(1).type_xtra = 2
                    elif len(children) >= 2:
                        section(1).type_xtra = 4
                else:
                    section(0).type_xtra = 6
                    if len(children) == 0:
                        section(1).type_xtra = 5 if section_length <= section_lambda else 2
                    elif len(children) >= 2:
                        section(1).type_xtra = 7 if section_length <= section_lambda else 4

    def _assign_xtra_pointer(self) -> None:
        """ Connects xtra mechanism to extracellular mechanism
        """
        for section in self.all:
            if section.has_membrane('xtra') and section.has_membrane('extracellular'):
                for segment in section:
                    h.setpointer(segment._ref_e_extracellular, 'ex', segment.xtra)

    def _assign_xtra_order(self) -> None:
        """ Assigns the branch order to each section.
        """
        soma = self.all[0]
        cell_stack = [soma]
        depth_stack = [0]
        while len(cell_stack) > 0:
            section = cell_stack.pop()
            depth = depth_stack.pop()
            section.order_xtra = depth
            number_children = len(list(section.children()))
            cell_stack += (list(section.children()))
            if number_children > 1:
                depth += 1
            depth_stack += [depth] * number_children

    def get_segment_coordinates(self) -> NDArray[float]:
        """ Returns a list of all segment center coordinates.

        :return: A list of all segment center coordinates
        """
        segment_positions = []
        for section in self.all:
            if section.has_membrane('xtra'):
                for segment in section:
                    segment_positions.append([segment.x_xtra, segment.y_xtra, segment.z_xtra])

        return np.array(segment_positions)

    def display_neuron(self) -> None:
        """ Displays the NEURON cell in the neuron displaying tool. (Blocking)
        """
        # noinspection PyUnresolvedReferences
        from neuron import gui
        ps = h.PlotShape(True)
        nodes = h.SectionList()
        myelin = h.SectionList()
        apical = h.SectionList()
        dendrites = h.SectionList()
        ps.exec_menu("Show Diam")
        ps.rotate(1, 0, 0, math.pi / 2, 0, 0)
        ps.exec_menu("View = plot")
        for axon_sec in self.axon:
            if 'Node' in axon_sec.name():
                nodes.append(axon_sec)
            elif 'Myelin' in axon_sec.name():
                myelin.append(axon_sec)
        for sec in self.apic:
            apical.append(sec)
        for sec in self.dend:
            dendrites.append(sec)
        ps.color_list(nodes, 1)
        ps.color_list(myelin, 2)
        ps.color_list(apical, 3)
        ps.color_list(dendrites, 4)
        while True:
            pass

    def display_neuron_pyplot(self) -> None:
        """ Displays the cell (No diameter)
        """
        ps = h.PlotShape(False)
        ps.variable('es_xtra')
        ps.plot(pyplot)
        pyplot.show()
        self._setup_xtra()

    def display_2d_matplotlib(self, path=None, include_diameter=False, rotated_90=False,
                              plot_grid=True, background_transparent=False, y_range=None) -> None:
        """
        Displays the cell in 2d using matplotlib.
        :param path: Path to save the plot to. If None: shows plot blocking.
        :param include_diameter: If True, the diameter will be displayed. If False, all sections will have the
        same diameter
        :param rotated_90: If True, rotates the neuron by 90 degrees azimuthal.
        :param plot_grid: If False, the grid will not be plotted.
        :param background_transparent: If True, the background will be plotted transparent.
        :param y_range: The y range to use for the plot. If None: takes the max and min y coordinates as the range.
        """
        plt.figure(figsize=(10, 10), dpi=200)
        plt.grid()
        plt.gca().set_aspect('equal')

        color = {'soma': [0, 0, 0, 1], 'axon': [0.8, 0, 0, 1], 'Myelin': [0.8, 0.6, 0, 1], 'Node': [0, 0, 0.8, 1],
                 'Unmyelin': [0.8, 0, 0, 1], 'dend': [0.25, 0.75, 0, 1], 'apic': [0.75, 0, 0.75, 1]}

        if rotated_90:
            z_sorted_sections = sorted(self.all, key=lambda sec: sec(0.5).x_xtra)
        else:
            z_sorted_sections = sorted(self.all, key=lambda sec: sec(0.5).y_xtra)

        for i, section in enumerate(z_sorted_sections):
            x_coordinates = []
            y_coordinates = []
            diameter = []
            section_type = section.name().split('.')[-1].split('[')[0]

            if section_type == 'soma':
                if rotated_90:
                    if include_diameter:
                        soma_circle = plt.Circle((self.soma[0](0.5).y_xtra, self.soma[0](0.5).z_xtra),
                                                 self.soma[0].L / 2,
                                                 color=color[section_type], zorder=i)
                    else:
                        soma_circle = plt.Circle((self.soma[0](0.5).y_xtra, self.soma[0](0.5).z_xtra),
                                                 30,
                                                 color=color[section_type], zorder=i)
                else:
                    if include_diameter:
                        soma_circle = plt.Circle((self.soma[0](0.5).x_xtra, self.soma[0](0.5).z_xtra),
                                                 self.soma[0].L / 2,
                                                 color=color[section_type], zorder=i)
                    else:
                        soma_circle = plt.Circle((self.soma[0](0.5).x_xtra, self.soma[0](0.5).z_xtra),
                                                 30,
                                                 color=color[section_type], zorder=i)
                plt.gca().add_patch(soma_circle)
                continue

            for point_index in range(section.n3d()):
                if rotated_90:
                    x_coordinates.append(section.y3d(point_index))
                else:
                    x_coordinates.append(section.x3d(point_index))
                y_coordinates.append(section.z3d(point_index))
                diameter.append(section.diam3d(point_index))
            if include_diameter:
                for point_index in range(section.n3d() - 1):
                    ortogonal = np.array([x_coordinates[point_index + 1], y_coordinates[point_index + 1]]) - np.array(
                        [x_coordinates[point_index], y_coordinates[point_index]])
                    ortogonal = np.array([ortogonal[1], -ortogonal[0]])
                    if np.linalg.norm(ortogonal) == 0:
                        ortogonal = np.array([0, 1])
                    ortogonal = ortogonal / np.linalg.norm(ortogonal)
                    s1 = np.array([x_coordinates[point_index], y_coordinates[point_index]]) + ortogonal * (
                            diameter[point_index] / 2)
                    s2 = np.array([x_coordinates[point_index], y_coordinates[point_index]]) + ortogonal * -(
                            diameter[point_index] / 2)
                    e1 = np.array([x_coordinates[point_index + 1], y_coordinates[point_index + 1]]) + ortogonal * (
                            diameter[point_index + 1] / 2)
                    e2 = np.array([x_coordinates[point_index + 1], y_coordinates[point_index + 1]]) + ortogonal * -(
                            diameter[point_index + 1] / 2)
                    polygon = plt.Polygon([s1, s2, e2, e1], True, facecolor=color[section_type], zorder=i)
                    plt.gca().add_patch(polygon)
            else:
                plt.plot(x_coordinates, y_coordinates, c=color[section.name().split('.')[-1].split('[')[0]], lw=3)
        legend_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color['soma'], markersize=10),
                        Line2D([0], [0], color=color['Myelin'], lw=2),
                        Line2D([0], [0], color=color['Node'], lw=2),
                        Line2D([0], [0], color=color['Unmyelin'], lw=2),
                        Line2D([0], [0], color=color['dend'], lw=2),
                        Line2D([0], [0], color=color['apic'], lw=2)]

        if plot_grid:
            plt.legend(legend_lines, ['soma', 'myelinated axon', 'node of ranvier',
                                      'unmyelinated axon', 'basal dendrite', 'apical dendrite'])
        else:
            plt.axis('off')
            plt.grid(b=None)

        coords = self.get_segment_coordinates()
        if y_range is not None:
            plt.ylim(y_range)
        else:
            plt.ylim([np.min(coords[:, 2]) - 100, np.max(coords[:, 2]) + 100])
        plt.xlim([np.min(coords[:, 0]) - 100, np.max(coords[:, 0]) + 100])

        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            if background_transparent:
                plt.savefig(path, dpi=1000, bbox_inches='tight', transparent="True", pad_inches=0)
            else:
                plt.savefig(path, dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close()

    def write_stl(self, path: typing.Union[str, pathlib.Path]) -> None:
        """ Writes the sections of the cell as truncated cones to a stl ascii file.

        :param path: Path to write the stl file to
        """
        path = pathlib.Path(path)
        triangles_all = {'soma': [], 'axon': [], 'Myelin': [], 'Node': [], 'Unmyelin': [], 'dend': [], 'apic': []}
        for section in self.all:
            triangles = triangles_all[section.name().split('.')[-1].split('[')[0]]
            for point_index in range(section.n3d() - 1):
                if np.allclose([section.x3d(point_index), section.y3d(point_index), section.z3d(point_index),
                                section.diam3d(point_index)],
                               [section.x3d(point_index + 1), section.y3d(point_index + 1),
                                section.z3d(point_index + 1), section.diam3d(point_index + 1)]):
                    continue
                radius1 = section.diam3d(point_index) / 2
                radius2 = section.diam3d(point_index + 1) / 2
                center1 = np.array([section.x3d(point_index), section.y3d(point_index), section.z3d(point_index)])
                center2 = np.array(
                    [section.x3d(point_index + 1), section.y3d(point_index + 1), section.z3d(point_index + 1)])

                direction = center2 - center1
                direction = direction / (np.linalg.norm(direction) + 1e-16)
                direction_max = np.argmax(direction)
                direction_min = np.argmin(direction)
                if direction_min == direction_max:
                    direction_max += 1
                direction_middle = 3 - direction_max - direction_min
                direction_r = np.copy(direction)
                direction_r[direction_min] = 0
                direction_r[direction_max] = direction[direction_middle]
                direction_r[direction_middle] = -direction[direction_max]
                direction_r = direction_r / (np.linalg.norm(direction_r) + 1e-16)

                r_x1 = direction_r * radius1
                r_x2 = direction_r * radius2
                r_y1 = np.cross(direction, direction_r) * radius1
                r_y2 = np.cross(direction, direction_r) * radius2

                delta_theta = 2.0 * math.pi / 30
                v2 = center1 + r_x1
                v3 = center2 + r_x2

                for n in range(1, 31):
                    theta = n * delta_theta
                    v1 = v2
                    v4 = v3
                    v2 = center1 + math.cos(theta) * r_x1 + math.sin(theta) * r_y1
                    v3 = center2 + math.cos(theta) * r_x2 + math.sin(theta) * r_y2
                    triangles.append([v1, v2, v3])
                    triangles.append([v1, v3, v4])
                    triangles.append([v2, v1, center1])
                    triangles.append([v4, v3, center2])

        with open(pathlib.Path(path), 'w') as f:
            for triangles in triangles_all:
                if len(triangles_all[triangles]) == 0:
                    continue
                f.write(f'solid {triangles}\n')
                for triangle in triangles_all[triangles]:
                    f.write(f'facet normal 0 0 0\n')
                    f.write(f'\touter loop\n')
                    f.write(f'\t\tvertex {triangle[0][0]} {triangle[0][1]} {triangle[0][2]}\n')
                    f.write(f'\t\tvertex {triangle[1][0]} {triangle[1][1]} {triangle[1][2]}\n')
                    f.write(f'\t\tvertex {triangle[2][0]} {triangle[2][1]} {triangle[2][2]}\n')
                    f.write(f'\tendloop\n')
                    f.write(f'endfacet\n')
                f.write(f'endsolid {triangles}\n')
