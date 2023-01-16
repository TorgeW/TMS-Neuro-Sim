import logging
import math

import numpy as np
import simnibs
import vtk
from scipy.interpolate import LinearNDInterpolator
from simnibs.mesh_tools import mesh_io
from simnibs.segmentation.marching_cube import marching_cubes_lewiner
from tqdm import tqdm

from tmsneurosim.nrn.cells import NeuronCell

WHITE_MATTER = 1
GRAY_MATTER = 2
CSF = 3
SKULL = 4
SKIN = 5
EYE = 6
WHITE_MATTER_SURFACE = 1001
GRAY_MATTER_SURFACE = 1002
CEREBROSPINAL_FLUID_SURFACE = 1003
SKULL_SURFACE = 1004
SKIN_SURFACE = 1005

POINTS_PER_MM = 3


def crop_box(mesh: simnibs.Msh, roi: [float], keep_elements: bool = False) -> simnibs.Msh:
    """ Returns the cropped mesh with all points that are inside the region of interest

    :param keep_elements: If True, keeps elements with at least one point in roi, else removes them
    :param mesh: The mesh that is supposed to be cropped
    :param roi: The region of interest which the mesh should be cropped to ([x-min, x-max, y-min, y-max, z-min, z-max])
    :return: The cropped mesh
    """
    node_keep_indexes = np.where(np.all(np.logical_and(
        mesh.nodes.node_coord <= [roi[1], roi[3], roi[5]],
        [roi[0], roi[2], roi[4]] <= mesh.nodes.node_coord), axis=1))[0] + 1

    if keep_elements:
        return mesh.crop_mesh(nodes=node_keep_indexes)
    else:
        elements_to_keep = \
            np.where(np.all(np.isin(mesh.elm.node_number_list, node_keep_indexes).reshape(-1, 4), axis=1))[0] + 1
        return mesh.crop_mesh(elements=elements_to_keep)


def create_roi_from_points(points, offset=0):
    """
    Creates a cubical region of interest array from a point clod and an offset.
    :param points: The point cloud that is supossed to be included inside the roi.
    :param offset: The offset of how much bigger the roi is supossed to be.
    :return: An array in the form of [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    return [points[:, 0].min() - offset, points[:, 0].max() + offset,
            points[:, 1].min() - offset, points[:, 1].max() + offset,
            points[:, 2].min() - offset, points[:, 2].max() + offset]


class CorticalLayer:
    """ Representation of a cortical layer.
    """

    def __init__(self, volumetric_mesh: simnibs.Msh, roi: [float] = None,
                 depth: float = None, elements_per_square_mm: float = None,
                 path=None,
                 surface: simnibs.Msh = None):
        """
        Creates a new instance of a cortical layer.
        :param volumetric_mesh: The E-field simulation mesh to create the cortical layer from.
        It is also used to get the E-field data from it
        :param roi: The region of interest inside the volumetric mesh.
        :param depth: The normalized depth of the cortical layer.
        :param elements_per_square_mm: The amount of simulation targets per mm²
        :param path: A path to load a cortical layer .msh file from.
        :param surface: A allready loaded .msh surface.
        """
        if path is not None:
            self.surface = simnibs.read_msh(path)
            roi = create_roi_from_points(self.surface.nodes.node_coord, 5)
            self.volumetric_mesh = crop_box(volumetric_mesh, roi, True)
        elif surface is not None:
            self.surface = surface
            roi = create_roi_from_points(self.surface.nodes.node_coord, 5)
            self.volumetric_mesh = crop_box(volumetric_mesh, roi, True)
        elif roi is not None and depth is not None:
            self.volumetric_mesh = crop_box(volumetric_mesh, roi + np.tile([-5, 5], 3), True)
            self.surface = None
            self.generate_layer(depth, roi)
        else:
            raise ValueError('At least one set of optional parameters must be assigned')

        if elements_per_square_mm is None:
            self.elements = np.array(range(self.surface.elm.nr))
        else:
            logging.info('Started element selection ...')
            self.elements = self.get_evenly_spaced_element_subset(elements_per_square_mm)
            logging.info('Finished element selection')
            logging.info(f'Selected {len(self.elements)}/{self.surface.elm.nr} elements'
                         f'({len(self.elements) / np.sum(self.surface.elements_volumes_and_areas().value)} elm/mm²)')

        self.node_fields = dict()
        for elm_field in self.volumetric_mesh.elmdata:
            for tag in np.unique(self.volumetric_mesh.elm.tag1[self.volumetric_mesh.elm.tetrahedra - 1]):
                self.node_fields[(tag, elm_field.field_name)] = \
                    self.volumetric_mesh.crop_mesh(tags=[tag]).field[elm_field.field_name].elm_data2node_data()

    def generate_layer(self, depth, roi):
        """
        Generate a cortical layer based on the volumetric mesh, the depth and the region of interest.
        :param depth:  The normalized depth of the cortical layer in the gray matter.
        :param roi: The region of interest to create the layer in.
        """
        outer_roi = [roi[0] - 5, roi[1] + 5, roi[2] - 5, roi[3] + 5, roi[4] - 5, roi[5] + 5]

        grid_x = np.linspace(outer_roi[0], outer_roi[1],
                             int(math.fabs(outer_roi[0] - outer_roi[1]) * POINTS_PER_MM))
        grid_y = np.linspace(outer_roi[2], outer_roi[3],
                             int(math.fabs(outer_roi[2] - outer_roi[3]) * POINTS_PER_MM))
        grid_z = np.linspace(outer_roi[4], outer_roi[5],
                             int(math.fabs(outer_roi[4] - outer_roi[5]) * POINTS_PER_MM))
        grid_points = np.stack(np.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), axis=-1).reshape(-1, 3)

        logging.info('Started tetrahedron search ...')
        point_tissue_tag = self.volumetric_mesh.find_tetrahedron_with_points(grid_points, compute_baricentric=False)
        point_indexes_in_volume = np.where(point_tissue_tag != -1)[0]
        point_tissue_tag[point_indexes_in_volume] = self.volumetric_mesh.elm.tag1[
            point_tissue_tag[point_indexes_in_volume] - 1]
        points_outside = np.where((point_tissue_tag != WHITE_MATTER) & (point_tissue_tag != GRAY_MATTER))[0]
        points_inside_gray_matter = np.where(point_tissue_tag == GRAY_MATTER)[0]

        roi_points = np.stack(np.meshgrid(outer_roi[0:2], outer_roi[2:4], outer_roi[4:6], indexing='ij'),
                              axis=-1).reshape(-1, 3)
        roi_tissue_tag = self.volumetric_mesh.elm.tag1[
            self.volumetric_mesh.find_tetrahedron_with_points(roi_points, compute_baricentric=False)]
        roi_tissue_tag[roi_tissue_tag != 1] = 0
        logging.info('Tetrahedron search done')

        logging.info('Started interpolation ...')
        wm_surface = self.volumetric_mesh.crop_mesh(WHITE_MATTER_SURFACE).nodes.node_coord
        gm_surface = self.volumetric_mesh.crop_mesh(GRAY_MATTER_SURFACE).nodes.node_coord

        data = [1] * len(wm_surface)
        data += [0] * len(gm_surface)
        data += roi_tissue_tag.tolist()

        data_points = np.concatenate((wm_surface, gm_surface, roi_points), axis=0)

        interpolation = LinearNDInterpolator(data_points, data, fill_value=-1)
        gray_matter_interpolation = interpolation(grid_points[points_inside_gray_matter])
        logging.info("Interpolation done")

        logging.info('Started surface creation ...')
        volume_data = np.empty((len(grid_x), len(grid_y), len(grid_z)))
        volume_data.fill(1)
        outside_x = (points_outside / (len(grid_z) * len(grid_y))).astype(int)
        outside_y = ((points_outside / len(grid_z)) % len(grid_y)).astype(int)
        outside_z = (points_outside % len(grid_z)).astype(int)
        volume_data[(outside_x, outside_y, outside_z)] = 0
        inside_gray_matter_x = (points_inside_gray_matter / (len(grid_z) * len(grid_y))).astype(int)
        inside_gray_matter_y = ((points_inside_gray_matter / len(grid_z)) % len(grid_y)).astype(int)
        inside_gray_matter_z = (points_inside_gray_matter % len(grid_z)).astype(int)
        volume_data[(inside_gray_matter_x, inside_gray_matter_y, inside_gray_matter_z)] = gray_matter_interpolation
        vertices, faces, _, _ = marching_cubes_lewiner(volume_data, level=depth,
                                                       spacing=tuple(np.array(
                                                           [grid_x[1] - grid_x[0], grid_y[1] - grid_y[0],
                                                            grid_z[1] - grid_z[0]], dtype='float32')),
                                                       step_size=1, allow_degenerate=False)
        self.surface = mesh_io.Msh(mesh_io.Nodes(vertices), mesh_io.Elements(faces + 1))
        # self.surface.smooth_surfaces(30)
        self.remove_unconnected_surfaces()
        self.surface.nodes.node_coord = self.surface.nodes.node_coord + [(outer_roi[0]), (outer_roi[2]), (outer_roi[4])]
        logging.info('Surface creation done')

        self.surface = crop_box(self.surface, roi, keep_elements=True)
        self.remove_unconnected_surfaces()
        self.surface.fix_surface_orientation()
        logging.info(f'Created layer ({np.sum(self.surface.elements_volumes_and_areas().value)} mm²) '
                     f'with {self.surface.elm.nr} elements '
                     f'({self.surface.elm.nr / np.sum(self.surface.elements_volumes_and_areas().value)} elm/mm²)')

    def remove_unconnected_surfaces(self):
        """ Removes unconnected surface patches and only keeps the largest surface.
        """
        surfaces = self.surface.elm.connected_components()
        surfaces.sort(key=len)
        self.surface = self.surface.crop_mesh(elements=surfaces[-1])

    def add_e_field_at_triangle_centers_field(self):
        """ Adds the E-field vector and the E-field magnitude as separate element fields.
        """
        centers = self.surface.elements_baricenters().value
        e_field_at_centers = self.interpolate_scattered_cashed(centers, 'E', 0)
        self.surface.add_element_field(e_field_at_centers, 'E-Field')
        self.surface.add_element_field(np.array([np.linalg.norm(v) for v in e_field_at_centers]), 'E-Field_Magnitude')

    def add_e_field_gradient_selected_elements_field(self, above, below, prefix=''):
        """
        Adds the E-field magnitude delta between above and below the cell for selected simulation targets.
        :param above: How far above to evaluate the E-field magnitude in mm.
        :param below: How far below to evaluate the E-field magnitude in mm.
        :param prefix: The prefix to be used on the name of the element field.
        """
        normals = self.get_smoothed_normals()
        centers = self.surface.elements_baricenters().value[self.elements]

        np.nan_to_num(normals, nan=-1, copy=False)
        np.nan_to_num(centers, nan=-1, copy=False)

        point_above = centers + normals * above
        point_below = centers + (normals * -1) * below
        cell_height = above + below

        e_field_vector_above = self.volumetric_mesh.field['E'].interpolate_scattered(point_above, out_fill=0)
        e_field_vector_below = self.volumetric_mesh.field['E'].interpolate_scattered(point_below, out_fill=0)

        e_field_above = np.array([np.linalg.norm(v) for v in e_field_vector_above])
        e_field_below = np.array([np.linalg.norm(v) for v in e_field_vector_below])

        e_field_gradient = e_field_below - e_field_above
        e_field_gradient_per_mm = e_field_gradient / cell_height

        self.add_selected_elements_field(e_field_gradient, f'{prefix}_E-Field_Gradient')
        self.add_selected_elements_field(e_field_gradient_per_mm, f'{prefix}_Gradient_per_mm')

    def add_position_tag_field(self, prefix='', cell_height=1, above=None, below=None, cell: NeuronCell = None):
        """
        Adds tissue tags for the top and the bottom at each cell at each triangle position of the cortical layer.
        :param prefix: Prefix for the name of the element field.
        :param cell_height: (optional) Height of the cell in mm.
        :param above: (optional) Distance to the top of the cell in mm.
        :param below: (optional) Distance to the bottom of the cell in mm.
        :param cell: (optional) A neuron to evaluate the height from.
        """
        normals = self.surface.triangle_normals().value
        centers = self.surface.elements_baricenters().value

        np.nan_to_num(normals, nan=-1, copy=False)
        np.nan_to_num(centers, nan=-1, copy=False)

        if above is not None and below is not None:
            point_above = centers + normals * above
            point_below = centers + (normals * -1) * below
        elif cell is not None:
            cell.load()

            cell_segment_coordinates = cell.get_segment_coordinates()
            soma_position_z = cell.soma[0](0.5).z_xtra
            min_z = np.amin(cell_segment_coordinates[:, 2])
            max_z = np.amax(cell_segment_coordinates[:, 2])
            soma_distance_to_min = math.fabs(soma_position_z - min_z) / 1000
            soma_distance_to_max = math.fabs(soma_position_z - max_z) / 1000
            point_above = centers + normals * soma_distance_to_max
            point_below = centers + (normals * -1) * soma_distance_to_min

            cell.unload()
        else:
            point_above = centers + normals * (cell_height / 2)
            point_below = centers + (normals * -1) * (cell_height / 2)

        tags_above = self.volumetric_mesh.elm.tag1[
            self.volumetric_mesh.find_tetrahedron_with_points(point_above, compute_baricentric=False)]
        tags_below = self.volumetric_mesh.elm.tag1[
            self.volumetric_mesh.find_tetrahedron_with_points(point_below, compute_baricentric=False)]

        if np.allclose(point_above, point_below):
            self.surface.add_element_field(tags_above, f'{prefix}_Tag')
        else:
            self.surface.add_element_field(tags_above, f'{prefix}_Tag_Bottom')
            self.surface.add_element_field(tags_below, f'{prefix}_Tag_Top')

    def add_e_field_gradient_between_wm_gm_field(self):
        """
        Adds the E-field magnitude delta between wm surface and gm surface.
        """
        centers = self.surface.elements_baricenters().value

        gray_matter_surface: simnibs.Msh = self.volumetric_mesh.crop_mesh(GRAY_MATTER_SURFACE)
        white_matter_surface: simnibs.Msh = self.volumetric_mesh.crop_mesh(WHITE_MATTER_SURFACE)

        _, closest_gm_indexes, gm_distance = gray_matter_surface.find_closest_element(centers,
                                                                                      return_index=True,
                                                                                      return_distance=True)
        _, closest_wm_indexes, wm_distance = white_matter_surface.find_closest_element(centers,
                                                                                       return_index=True,
                                                                                       return_distance=True)

        gm_centers = gray_matter_surface.elements_baricenters()[closest_gm_indexes] - \
                     gray_matter_surface.triangle_normals()[closest_gm_indexes] * 0.001
        wm_centers = white_matter_surface.elements_baricenters()[closest_wm_indexes] + \
                     white_matter_surface.triangle_normals()[closest_wm_indexes] * 0.001

        e_field_above = self.volumetric_mesh.field['E'].interpolate_scattered(gm_centers, out_fill=0)
        e_field_below = self.volumetric_mesh.field['E'].interpolate_scattered(wm_centers, out_fill=0)

        e_field_above_mag = np.array([np.linalg.norm(v) for v in e_field_above])[:, None].flatten()
        e_field_below_mag = np.array([np.linalg.norm(v) for v in e_field_below])[:, None].flatten()

        e_field_gradient = e_field_below_mag - e_field_above_mag
        gray_matter_thickness = gm_distance + wm_distance
        distance_offset = gm_distance / gray_matter_thickness
        e_field_gradient_per_mm = e_field_gradient / gray_matter_thickness

        self.surface.add_element_field(e_field_gradient, 'E-Field_Gradient')
        self.surface.add_element_field(e_field_gradient_per_mm, 'Gradient_per_mm')
        self.surface.add_element_field(gray_matter_thickness, 'Thickness')
        self.surface.add_element_field(distance_offset, 'Mid-layer_Offset')

    def make_vtkpolydata(self, pts, tris):
        """
        Creates a vtk polydata object from vertex positions and a triangle list.
        :param pts: Vertex positions.
        :param tris: Triangle list.
        """
        # prepare vertices
        pts_vtk = vtk.vtkPoints()
        pts_vtk.SetNumberOfPoints(pts.shape[0])
        for i in range(pts.shape[0]):
            pts_vtk.SetPoint(i, pts[i][0], pts[i][1], pts[i][2])

        # prepare triangles
        tris_vtk = vtk.vtkCellArray()
        for tri in tris:
            tris_vtk.InsertNextCell(3)
            for v in tri:
                tris_vtk.InsertCellPoint(v)

        # prepare GM polygonal surface
        surf_vtk = vtk.vtkPolyData()
        surf_vtk.SetPoints(pts_vtk)
        surf_vtk.SetPolys(tris_vtk)

        return surf_vtk

    def add_delta_e_mag_between_wm_gm_field(self):
        """
        Adds the E-field magnitude delta between wm surface and gm surface based on ray casting.
        """
        gray_matter_surface: simnibs.Msh = self.volumetric_mesh.crop_mesh(GRAY_MATTER_SURFACE)
        white_matter_surface: simnibs.Msh = self.volumetric_mesh.crop_mesh(WHITE_MATTER_SURFACE)

        gm_nodes = gray_matter_surface.nodes.node_coord
        gm_tris = gray_matter_surface.elm.node_number_list
        wm_nodes = white_matter_surface.nodes.node_coord
        wm_tris = white_matter_surface.elm.node_number_list

        gm_surf_vtk = self.make_vtkpolydata(gm_nodes, gm_tris[:, :3] - 1)
        wm_surf_vtk = self.make_vtkpolydata(wm_nodes, wm_tris[:, :3] - 1)

        gm_intersector = vtk.vtkOBBTree()
        gm_intersector.SetDataSet(gm_surf_vtk)
        gm_intersector.BuildLocator()

        wm_intersector = vtk.vtkOBBTree()
        wm_intersector.SetDataSet(wm_surf_vtk)
        wm_intersector.BuildLocator()

        intersectors = [gm_intersector, wm_intersector]
        normal_sign = [1, -1]

        layer_gm_wm_info = dict()
        intersec_pts = [[], []]
        layer_centers = self.surface.elements_baricenters().value
        layer_normals = self.surface.triangle_normals(smooth=30).value

        for pt, normal in zip(layer_centers, layer_normals):
            for idx in range(len(intersectors)):
                intersection_pts = vtk.vtkPoints()
                intersected_tris = vtk.vtkIdList()

                intersector = intersectors[idx]

                intersector.IntersectWithLine(pt, normal_sign[idx] * 100 * normal + pt, intersection_pts,
                                              intersected_tris)
                if intersection_pts.GetNumberOfPoints() > 0:
                    intersec_pts[idx].append(intersection_pts.GetPoint(0))
                else:
                    intersec_pts[idx].append([np.iinfo(np.uint16).max] * 3)

        gm_intersec_pts = np.array(intersec_pts[0])
        wm_intersec_pts = np.array(intersec_pts[1])

        # Check the distances of the found points on the GM/WM surface to the layer nodes.
        # If the distance is too high, raycasting failed (e.g. no intersection found or too far away).
        # In this case, we determine the nearest neighbor for these nodes.
        layer_gm_distance = np.linalg.norm(layer_centers - gm_intersec_pts, ord=2, axis=1)
        layer_wm_distance = np.linalg.norm(layer_centers - wm_intersec_pts, ord=2, axis=1)

        raycast_error_nodes_gm = np.argwhere(layer_gm_distance > 2)  # mm
        raycast_error_nodes_wm = np.argwhere(layer_wm_distance > 2)  # mm

        closest_gm_nodes, _ = gray_matter_surface.nodes.find_closest_node(layer_centers[raycast_error_nodes_gm],
                                                                          return_index=True)
        closest_wm_nodes, _ = white_matter_surface.nodes.find_closest_node(layer_centers[raycast_error_nodes_wm],
                                                                           return_index=True)

        gm_intersec_pts[raycast_error_nodes_gm] = closest_gm_nodes
        wm_intersec_pts[raycast_error_nodes_wm] = closest_wm_nodes

        layer_gm_distance = np.linalg.norm(layer_centers - gm_intersec_pts, ord=2, axis=1)
        layer_wm_distance = np.linalg.norm(layer_centers - wm_intersec_pts, ord=2, axis=1)

        center_to_gm_vecs = gm_intersec_pts - layer_centers
        center_to_wm_vecs = wm_intersec_pts - layer_centers

        associated_gray_matter_points = gm_intersec_pts - \
                                        np.multiply(
                                            # multiply unit-normals by individual gm thickness, then scale to 20%
                                            center_to_gm_vecs,
                                            layer_gm_distance[:, np.newaxis]
                                        ) * 0.1
        associated_white_matter_points = wm_intersec_pts - \
                                         np.multiply(
                                             # multiply unit-normals by individual wm thickness, then scale to 20%
                                             center_to_wm_vecs,
                                             layer_wm_distance[:, np.newaxis]
                                         ) * 0.1

        e_field_above = self.volumetric_mesh.field['E'].interpolate_scattered(associated_gray_matter_points, out_fill=0)
        e_field_below = self.volumetric_mesh.field['E'].interpolate_scattered(associated_white_matter_points,
                                                                              out_fill=0)

        e_field_above_mag = np.array([np.linalg.norm(v) for v in e_field_above])[:, None].flatten()
        e_field_below_mag = np.array([np.linalg.norm(v) for v in e_field_below])[:, None].flatten()

        e_field_gradient = e_field_below_mag - e_field_above_mag
        gray_matter_thickness = layer_gm_distance + layer_wm_distance
        distance_offset = layer_gm_distance / gray_matter_thickness
        e_field_gradient_per_mm = e_field_gradient / gray_matter_thickness

        self.surface.add_element_field(e_field_gradient, 'E-Field_Gradient')
        self.surface.add_element_field(e_field_gradient_per_mm, 'Gradient_per_mm')
        self.surface.add_element_field(gray_matter_thickness, 'Thickness')
        self.surface.add_element_field(distance_offset, 'Mid-layer_Offset')

    def get_evenly_spaced_element_subset(self, elements_per_square_mm):
        """
        Creates an evenly spaced subset of layer triangles. Selects the sumbitted number of elements per mm².
        :param elements_per_square_mm: Number of elements to select per mm².
        :return: Array of selected element indexes.
        """
        centers = self.surface.elements_baricenters().value
        min_distance_square = (1 / elements_per_square_mm) * math.sqrt(2) / 2
        selected_elements = np.array([0])
        for element_index in tqdm(range(self.surface.elm.nr)):
            selected_elements_centers = centers[selected_elements]
            element_center = centers[element_index]
            distances_square = (selected_elements_centers[:, 0] - element_center[0]) ** 2 + \
                               (selected_elements_centers[:, 1] - element_center[1]) ** 2 + \
                               (selected_elements_centers[:, 2] - element_center[2]) ** 2
            if distances_square.min() > min_distance_square:
                selected_elements = np.append(selected_elements, element_index)

        return np.array(selected_elements)

    def get_smoothed_normals(self):
        """
        Calculates smoothed normals on the layer.
        :return: Smoothed normal vectors.
        """
        return self.surface.triangle_normals(smooth=30).value[self.elements]

    def add_nearest_interpolation_field(self, values, name):
        """
        Adds an element field of selected element values by nearest neighbor interpolation.
        :param values: The values to interpolate over the layer.
        :param name: The name of the element field.
        """
        centers = self.surface.elements_baricenters().value
        element_values = np.array(values)
        nearest_interpolation_values = np.zeros(self.surface.elm.nr)
        for element_index in tqdm(range(self.surface.elm.nr)):
            elements_centers = centers[self.elements]
            element_center = centers[element_index]
            distances_square = (elements_centers[:, 0] - element_center[0]) ** 2 + \
                               (elements_centers[:, 1] - element_center[1]) ** 2 + \
                               (elements_centers[:, 2] - element_center[2]) ** 2
            nearest_interpolation_values[element_index] = element_values[np.argmin(distances_square)]
        self.surface.add_element_field(nearest_interpolation_values, name)

    def add_selected_elements_field(self, values, name):
        """
        Adds an element field of selected elements by filling not selected values with nan.
        :param values: Values to use for the element field.
        :param name: The name of the element field.
        """
        if len(values.shape) > 1:
            element_field = np.full((self.surface.elm.nr, values.shape[1]), np.nan)
        else:
            element_field = np.full(self.surface.elm.nr, np.nan)
        element_field[self.elements] = values
        self.surface.add_element_field(element_field, name)

    def interpolate_scattered_cashed(self, points, elm_field: str, out_fill=np.nan, get_tetrahedron_tags=False):
        """ $SimNIBS$"""

        data_field = self.volumetric_mesh.field[elm_field]
        if len(data_field.value.shape) > 1:
            f = np.zeros((points.shape[0], data_field.nr_comp), data_field.value.dtype)
        else:
            f = np.zeros((points.shape[0],), data_field.value.dtype)

        th_with_points, bar = self.volumetric_mesh.find_tetrahedron_with_points(points, compute_baricentric=True)

        inside = th_with_points != -1
        sorted_tag = np.array([])

        # if any points are inside
        if np.any(inside):

            # get the indices of True elements in 'inside'
            where_inside = np.where(inside)[0]

            # get the 'elm_number' of the tetrahedra in 'msh' which contain 'points' in 'points' order
            # assert where_inside.shape == th.shape
            th = th_with_points[where_inside]

            # get sorted unique elements of `th`
            sorted_th, arg_th, arg_inv = np.unique(th, return_index=True, return_inverse=True)

            # get the 'tag1' from 'msh' for every element in 'th' in 'points' order
            sorted_tag = self.volumetric_mesh.elm.tag1[sorted_th - 1]

            for t in np.unique(sorted_tag):
                # find the elements in 'sorted_tag' which equals to 't'
                is_t = sorted_tag == t

                # 'msh_tag' contains only the tetrahedra with 'elm_number == th_with_t'
                msh_tag = self.volumetric_mesh.crop_mesh(tags=t)

                # convert the 'elmdata' to 'nodedata'
                # nd = msh_tag.elmdata[0].elm_data2node_data()
                # nd = msh_tag.field[f'{elm_field}_node']

                nd = self.node_fields[(t, elm_field)]

                # 'msh_with_t' is sorted because 'elm_number' is always sorted
                msh_with_t = self.volumetric_mesh.elm.elm_number[self.volumetric_mesh.elm.tag1 == t]

                # use 'is_t' to select the indices of the tetrahedra inside and with 'tag1 == t'
                where_inside_with_t = where_inside[is_t[arg_inv]]

                # the 'elm_number' of elements in 'msh'. These elements contain points and 'tag1 == t'
                th_with_t = th_with_points[where_inside_with_t]

                # get the indices of elements in 'msh_tag'. The 'elm_number' of the same elements are 'th_with_t' in 'msh'. 'idx' starts from 0, not 1.
                idx = np.searchsorted(msh_with_t, th_with_t)

                if where_inside_with_t.size and len(nd.value.shape) == 1:
                    f[where_inside_with_t] = np.einsum('ik, ik -> i',
                                                       nd[msh_tag.elm[idx + 1]],
                                                       bar[where_inside_with_t])
                elif where_inside_with_t.size:
                    f[where_inside_with_t] = np.einsum('ikj, ik -> ij',
                                                       nd[msh_tag.elm[idx + 1]],
                                                       bar[where_inside_with_t])

        # Finally, fill in the unassigned values
        if np.any(~inside):
            f[~inside] = out_fill

        f = np.squeeze(f)
        if get_tetrahedron_tags:
            return f, sorted_tag
        else:
            return f
