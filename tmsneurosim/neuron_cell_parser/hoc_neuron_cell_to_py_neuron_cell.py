import pathlib
import re
from collections import defaultdict

import tmsneurosim.nrn

compartment_names = {
    'all': 'all',
    'axonal': 'axon',
    'basal': 'dend',
    'somatic': 'soma',
    'apical': 'apic'
}


def convert_hoc_to_python_cell(cell_type: str):
    """
    Converts a hoc based neuron model to a Python based neuron model.
    The Python based model is a subclass of NeuronCell.
    :param cell_type: The name of the neuron model to convert
    :return: The file name and the class name of the new Python neuron model
    """
    path = nrn_path.joinpath(f'cells/cells_hoc/{cell_type}').absolute()
    output = nrn_path.joinpath(f'cells/{cell_type.lower().replace("-", "_")}.py')
    morphology_path = f'str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath(\'cells/cells_hoc/{cell_type}/morphology/\').resolve())'

    biophysics_path = pathlib.Path(path).joinpath('biophysics.hoc')
    template_file_path = nrn_path.joinpath('cells/neuron_cell_template.py')
    with open(biophysics_path, 'r') as f:
        biophysics = f.read().replace("\t", "")
    section_types = set(re.findall('forsec \\$o1\\.([^\\s]*)', biophysics))
    apply_biophysics = ''
    for section_type in section_types:
        apply_biophysics += f'for {compartment_names[section_type]}_section in self.{compartment_names[section_type]}:\n'
        section_code_segments = re.findall(section_type + ' {([^}]*)}', biophysics)
        section_attributes = []
        for section_code in section_code_segments:
            section_attributes.extend(section_code.split('\n'))
        for section_attribute in section_attributes:
            if 'insert' in section_attribute:
                mechanism = section_attribute.replace('insert ', '')
                apply_biophysics += f'    {compartment_names[section_type]}_section.insert(\'{mechanism}\')\n'
            elif '=' in section_attribute:
                attribute = section_attribute.split(' = ')
                apply_biophysics += f'    {compartment_names[section_type]}_section.{attribute[0]} = {attribute[1]}\n'
        apply_biophysics += '\n'
    segment_attributes = re.findall('distribute\\(\\$o1\\.([^,]*),"([^"]*)","([^"]*)"', biophysics)
    d = defaultdict(list)
    for k, *v in segment_attributes:
        d[k].append(v)
    segment_attributes = list(d.items())
    for segment in segment_attributes:
        apply_biophysics += f'for {compartment_names[segment[0]]}_section in self.{compartment_names[segment[0]]}:\n'
        apply_biophysics += f'    for {compartment_names[segment[0]]}_segment in {compartment_names[segment[0]]}_section.allseg():\n'
        distance_term = f'{compartment_names[segment[0]]}_segment_distance'
        apply_biophysics += f'        {distance_term} = h.distance(self.soma[0](0), {compartment_names[segment[0]]}_segment)\n'
        for segment_attribute in segment[1]:
            segment_attribute[1] = segment_attribute[1].replace('exp', 'math.exp')
            apply_biophysics += f'        {compartment_names[segment[0]]}_segment.{segment_attribute[0]} = {segment_attribute[1].replace("%g", distance_term)}\n'
    apply_biophysics = '    ' + apply_biophysics.replace('\n', '\n    ')
    apply_biophysics = '\ndef apply_biophysics(self):\n' + apply_biophysics
    apply_biophysics = apply_biophysics.replace('\n', '\n    ')
    with open(output, 'w') as f, open(template_file_path, 'r') as template:
        template_cell = template.read()
        template_cell = template_cell.replace('NeuronCellTemplate', f'{cell_type.replace("-", "_")}')
        template_cell = template_cell.replace('"morphology_path"', f'{morphology_path}')

        f.write(template_cell)
        f.write(apply_biophysics)
        f.write('\n')
    return f'{cell_type.lower().replace("-", "_")}', cell_type.replace("-", "_")


if __name__ == '__main__':
    nrn_path = pathlib.Path(tmsneurosim.nrn.__file__).parent
    neuron_cell_names = [folder_paths.name for folder_paths
                         in sorted(pathlib.Path(nrn_path.joinpath(f'cells/cells_hoc').absolute()).iterdir())
                         if folder_paths.is_dir()]

    python_cell_names = []
    for neuron_cell_name in neuron_cell_names:
        python_cell_names.append(convert_hoc_to_python_cell(neuron_cell_name))

    with open(nrn_path.joinpath(f'cells/__init__.py'), 'w') as f:
        for file_name, class_name in python_cell_names:
            f.write(f'from .{file_name} import {class_name}\n')
        f.write(f'from .neuron_cell import NeuronCell\n')
