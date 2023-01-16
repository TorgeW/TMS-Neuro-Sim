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


class L4_LBC_cACint(NeuronCell):
    def __init__(self, morphology_id, modification_parameters: CellModificationParameters = None, variation_seed=None):
        super().__init__(str(list(pathlib.Path(str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath(
            'cells/cells_hoc/L4_LBC_cACint/morphology/').resolve())).joinpath(f'{morphology_id}').iterdir())[0]),
                         morphology_id, modification_parameters, variation_seed)

    @staticmethod
    def get_morphology_ids():
        return sorted([int(f.name) for f in pathlib.Path(str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath(
            'cells/cells_hoc/L4_LBC_cACint/morphology/').resolve())).iterdir() if f.is_dir()])

    def apply_biophysics(self):
        for all_section in self.all:
            all_section.insert('pas')
            all_section.Ra = 100.0
            all_section.cm = 1.0
            all_section.e_pas = -75.300257

        for axon_section in self.axon:
            axon_section.insert('SKv3_1')
            axon_section.insert('Ca')
            axon_section.insert('SK_E2')
            axon_section.insert('CaDynamics_E2')
            axon_section.insert('Nap_Et2')
            axon_section.insert('Im')
            axon_section.insert('K_Pst')
            axon_section.insert('K_Tst')
            axon_section.insert('Ca_LVAst')
            axon_section.insert('NaTa_t')
            axon_section.ena = 50
            axon_section.ek = -85

        for soma_section in self.soma:
            soma_section.insert('NaTs2_t')
            soma_section.insert('SKv3_1')
            soma_section.insert('Ca')
            soma_section.insert('SK_E2')
            soma_section.insert('Ca_LVAst')
            soma_section.insert('Nap_Et2')
            soma_section.insert('Im')
            soma_section.insert('K_Pst')
            soma_section.insert('K_Tst')
            soma_section.insert('CaDynamics_E2')
            soma_section.ena = 50
            soma_section.ek = -85

        for apic_section in self.apic:
            apic_section.insert('NaTs2_t')
            apic_section.insert('SKv3_1')
            apic_section.insert('Nap_Et2')
            apic_section.insert('Ih')
            apic_section.insert('Im')
            apic_section.insert('K_Pst')
            apic_section.insert('K_Tst')
            apic_section.ena = 50
            apic_section.ek = -85

        for dend_section in self.dend:
            dend_section.insert('NaTs2_t')
            dend_section.insert('SKv3_1')
            dend_section.insert('Nap_Et2')
            dend_section.insert('Ih')
            dend_section.insert('Im')
            dend_section.insert('K_Pst')
            dend_section.insert('K_Tst')
            dend_section.ena = 50
            dend_section.ek = -85

        for dend_section in self.dend:
            for dend_segment in dend_section.allseg():
                dend_segment_distance = h.distance(self.soma[0](0), dend_segment)
                dend_segment.gK_Tstbar_K_Tst = (0.0 * dend_segment_distance + 1.0) * 0.007375
                dend_segment.gSKv3_1bar_SKv3_1 = (0.0 * dend_segment_distance + 1.0) * 0.000041
                dend_segment.gNap_Et2bar_Nap_Et2 = (0.0 * dend_segment_distance + 1.0) * 0.000000
                dend_segment.gNaTs2_tbar_NaTs2_t = (0.0 * dend_segment_distance + 1.0) * 0.001373
                dend_segment.gIhbar_Ih = (-0.869600 + 2.087000 * math.exp(
                    (dend_segment_distance - 0.000000) * 0.003000)) * 0.000023
                dend_segment.e_pas = (0.0 * dend_segment_distance + 1.0) * -63.118492
                dend_segment.g_pas = (0.0 * dend_segment_distance + 1.0) * 0.000001
                dend_segment.gImbar_Im = (0.0 * dend_segment_distance + 1.0) * 0.000014
        for apic_section in self.apic:
            for apic_segment in apic_section.allseg():
                apic_segment_distance = h.distance(self.soma[0](0), apic_segment)
                apic_segment.gK_Tstbar_K_Tst = (0.0 * apic_segment_distance + 1.0) * 0.007375
                apic_segment.gSKv3_1bar_SKv3_1 = (0.0 * apic_segment_distance + 1.0) * 0.000041
                apic_segment.gNap_Et2bar_Nap_Et2 = (0.0 * apic_segment_distance + 1.0) * 0.000000
                apic_segment.gNaTs2_tbar_NaTs2_t = (0.0 * apic_segment_distance + 1.0) * 0.001373
                apic_segment.gIhbar_Ih = (-0.869600 + 2.087000 * math.exp(
                    (apic_segment_distance - 0.000000) * 0.003000)) * 0.000023
                apic_segment.e_pas = (0.0 * apic_segment_distance + 1.0) * -63.118492
                apic_segment.g_pas = (0.0 * apic_segment_distance + 1.0) * 0.000001
                apic_segment.gImbar_Im = (0.0 * apic_segment_distance + 1.0) * 0.000014
        for axon_section in self.axon:
            for axon_segment in axon_section.allseg():
                axon_segment_distance = h.distance(self.soma[0](0), axon_segment)
                axon_segment.gNaTa_tbar_NaTa_t = (0.0 * axon_segment_distance + 1.0) * 3.993125
                axon_segment.gK_Tstbar_K_Tst = (0.0 * axon_segment_distance + 1.0) * 0.000000
                axon_segment.gamma_CaDynamics_E2 = (0.0 * axon_segment_distance + 1.0) * 0.000503
                axon_segment.gNap_Et2bar_Nap_Et2 = (0.0 * axon_segment_distance + 1.0) * 0.000000
                axon_segment.gCa_LVAstbar_Ca_LVAst = (0.0 * axon_segment_distance + 1.0) * 0.009986
                axon_segment.gSK_E2bar_SK_E2 = (0.0 * axon_segment_distance + 1.0) * 0.005611
                axon_segment.gK_Pstbar_K_Pst = (0.0 * axon_segment_distance + 1.0) * 0.068460
                axon_segment.gSKv3_1bar_SKv3_1 = (0.0 * axon_segment_distance + 1.0) * 0.517764
                axon_segment.decay_CaDynamics_E2 = (0.0 * axon_segment_distance + 1.0) * 573.007045
                axon_segment.e_pas = (0.0 * axon_segment_distance + 1.0) * -64.601696
                axon_segment.g_pas = (0.0 * axon_segment_distance + 1.0) * 0.000063
                axon_segment.gImbar_Im = (0.0 * axon_segment_distance + 1.0) * 0.000345
                axon_segment.gCabar_Ca = (0.0 * axon_segment_distance + 1.0) * 0.000501
        for soma_section in self.soma:
            for soma_segment in soma_section.allseg():
                soma_segment_distance = h.distance(self.soma[0](0), soma_segment)
                soma_segment.gK_Tstbar_K_Tst = (0.0 * soma_segment_distance + 1.0) * 0.077616
                soma_segment.gamma_CaDynamics_E2 = (0.0 * soma_segment_distance + 1.0) * 0.000814
                soma_segment.gNap_Et2bar_Nap_Et2 = (0.0 * soma_segment_distance + 1.0) * 0.000001
                soma_segment.gCa_LVAstbar_Ca_LVAst = (0.0 * soma_segment_distance + 1.0) * 0.009728
                soma_segment.gSK_E2bar_SK_E2 = (0.0 * soma_segment_distance + 1.0) * 0.002099
                soma_segment.gK_Pstbar_K_Pst = (0.0 * soma_segment_distance + 1.0) * 0.000114
                soma_segment.gSKv3_1bar_SKv3_1 = (0.0 * soma_segment_distance + 1.0) * 0.260872
                soma_segment.decay_CaDynamics_E2 = (0.0 * soma_segment_distance + 1.0) * 967.678789
                soma_segment.e_pas = (0.0 * soma_segment_distance + 1.0) * -69.781406
                soma_segment.g_pas = (0.0 * soma_segment_distance + 1.0) * 0.000020
                soma_segment.gImbar_Im = (0.0 * soma_segment_distance + 1.0) * 0.000032
                soma_segment.gNaTs2_tbar_NaTs2_t = (0.0 * soma_segment_distance + 1.0) * 0.395881
                soma_segment.gCabar_Ca = (0.0 * soma_segment_distance + 1.0) * 0.000028
