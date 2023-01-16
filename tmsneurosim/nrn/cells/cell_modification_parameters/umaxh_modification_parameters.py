from dataclasses import dataclass

from tmsneurosim.nrn.cells.cell_modification_parameters.cell_modification_parameters import CellModificationParameters, \
    AxonModificationMode


@dataclass
class UmaxHModificationParameters(CellModificationParameters):
    """
    Parameters to create a neuron of an adult, human with unmyelinated axon using L2/3 PC rat BB: human Eyal 2018
    """

    axon_modification_mode: AxonModificationMode = AxonModificationMode.KEEP_AXON
    soma_area_scaling_factor: float = 2.453
    axon_diameter_scaling_factor: float = 2.453
    main_axon_diameter_scaling_factor: float = 1
    apic_diameter_scaling_factor: float = 1.876
    dend_diameter_scaling_factor: float = 1.946
    dend_length_scaling_factor: float = 1.17
