# noinspection PyUnresolvedReferences
import math
# noinspection PyUnresolvedReferences
import pathlib

# noinspection PyUnresolvedReferences
from neuron import h

# noinspection PyUnresolvedReferences
import tmsneurosim
from tmsneurosim.nrn.cells.cell_modification_parameters.cell_modification_parameters import CellModificationParameters
from tmsneurosim.nrn.cells.neuron_cell import NeuronCell


class L23_PC_cADpyr(NeuronCell):
    def __init__(self, morphology_id, modification_parameters: CellModificationParameters = None, variation_seed=None):
        super().__init__(str(list(pathlib.Path(str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath(
            'cells/cells_hoc/L23_PC_cADpyr/morphology/').resolve())).joinpath(f'{morphology_id}').iterdir())[0]),
                         morphology_id, modification_parameters, variation_seed)

    @staticmethod
    def get_morphology_ids():
        return sorted([int(f.name) for f in pathlib.Path(str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath(
            'cells/cells_hoc/L23_PC_cADpyr/morphology/').resolve())).iterdir() if f.is_dir()])

    def apply_biophysics(self):
        for all_section in self.all:
            all_section.insert('pas')
            all_section.e_pas = -75
            all_section.Ra = 100
            all_section.cm = 1
            all_section.g_pas = 3e-5

        for axon_section in self.axon:
            axon_section.insert('Ca_HVA')
            axon_section.insert('SKv3_1')
            axon_section.insert('SK_E2')
            axon_section.insert('CaDynamics_E2')
            axon_section.insert('Nap_Et2')
            axon_section.insert('K_Pst')
            axon_section.insert('K_Tst')
            axon_section.insert('Ca_LVAst')
            axon_section.insert('NaTa_t')
            axon_section.ena = 50
            axon_section.ek = -85

        for soma_section in self.soma:
            soma_section.insert('Ca_HVA')
            soma_section.insert('SKv3_1')
            soma_section.insert('SK_E2')
            soma_section.insert('Ca_LVAst')
            soma_section.insert('Ih')
            soma_section.insert('NaTs2_t')
            soma_section.insert('CaDynamics_E2')
            soma_section.ena = 50
            soma_section.ek = -85

        for apic_section in self.apic:
            apic_section.insert('Im')
            apic_section.insert('NaTs2_t')
            apic_section.insert('SKv3_1')
            apic_section.insert('Ih')
            apic_section.ena = 50
            apic_section.ek = -85
            apic_section.cm = 2

        for dend_section in self.dend:
            dend_section.insert('Ih')
            dend_section.cm = 2

        for dend_section in self.dend:
            for dend_segment in dend_section.allseg():
                dend_segment_distance = h.distance(self.soma[0](0), dend_segment)
                dend_segment.gIhbar_Ih = (0.0 * dend_segment_distance + 1.0) * 0.000080
        for apic_section in self.apic:
            for apic_segment in apic_section.allseg():
                apic_segment_distance = h.distance(self.soma[0](0), apic_segment)
                apic_segment.gNaTs2_tbar_NaTs2_t = (0.0 * apic_segment_distance + 1.0) * 0.012009
                apic_segment.gSKv3_1bar_SKv3_1 = (0.0 * apic_segment_distance + 1.0) * 0.000513
                apic_segment.gIhbar_Ih = (-0.869600 + 2.087000 * math.exp(
                    (apic_segment_distance - 0.000000) * 0.003100)) * 0.000080
                apic_segment.gImbar_Im = (0.0 * apic_segment_distance + 1.0) * 0.000740
        for axon_section in self.axon:
            for axon_segment in axon_section.allseg():
                axon_segment_distance = h.distance(self.soma[0](0), axon_segment)
                axon_segment.gNaTa_tbar_NaTa_t = (0.0 * axon_segment_distance + 1.0) * 3.429725
                axon_segment.gK_Tstbar_K_Tst = (0.0 * axon_segment_distance + 1.0) * 0.001035
                axon_segment.gamma_CaDynamics_E2 = (0.0 * axon_segment_distance + 1.0) * 0.016713
                axon_segment.gNap_Et2bar_Nap_Et2 = (0.0 * axon_segment_distance + 1.0) * 0.009803
                axon_segment.gSK_E2bar_SK_E2 = (0.0 * axon_segment_distance + 1.0) * 0.008085
                axon_segment.gCa_HVAbar_Ca_HVA = (0.0 * axon_segment_distance + 1.0) * 0.000306
                axon_segment.gK_Pstbar_K_Pst = (0.0 * axon_segment_distance + 1.0) * 0.959296
                axon_segment.gSKv3_1bar_SKv3_1 = (0.0 * axon_segment_distance + 1.0) * 0.094971
                axon_segment.decay_CaDynamics_E2 = (0.0 * axon_segment_distance + 1.0) * 384.114655
                axon_segment.gCa_LVAstbar_Ca_LVAst = (0.0 * axon_segment_distance + 1.0) * 0.000050
        for soma_section in self.soma:
            for soma_segment in soma_section.allseg():
                soma_segment_distance = h.distance(self.soma[0](0), soma_segment)
                soma_segment.gamma_CaDynamics_E2 = (0.0 * soma_segment_distance + 1.0) * 0.000533
                soma_segment.gSKv3_1bar_SKv3_1 = (0.0 * soma_segment_distance + 1.0) * 0.102517
                soma_segment.gSK_E2bar_SK_E2 = (0.0 * soma_segment_distance + 1.0) * 0.099433
                soma_segment.gCa_HVAbar_Ca_HVA = (0.0 * soma_segment_distance + 1.0) * 0.000374
                soma_segment.gNaTs2_tbar_NaTs2_t = (0.0 * soma_segment_distance + 1.0) * 0.926705
                soma_segment.gIhbar_Ih = (0.0 * soma_segment_distance + 1.0) * 0.000080
                soma_segment.decay_CaDynamics_E2 = (0.0 * soma_segment_distance + 1.0) * 342.544232
                soma_segment.gCa_LVAstbar_Ca_LVAst = (0.0 * soma_segment_distance + 1.0) * 0.000778
