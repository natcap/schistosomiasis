"""Preprocess data for Schistosomiasis.

This script clips and reprojects data needed for the
Schisto model.
"""
import argparse
import logging
import os
import sys

import numpy

import pygeoprocessing
from pygeoprocessing import geoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)

logging.getLogger('taskgraph').setLevel('DEBUG')

### Set up areas of interest and a projected spatial reference.
_SENEGAL_EPSG = 32628 # WGS 84 / UTM zone 28N
_SENEGAL_SRS = osr.SpatialReference()
_SENEGAL_SRS.ImportFromEPSG(_SENEGAL_EPSG)

_TANZANIA_EPSG = 21037 # Arc 1960 / UTM zonea 37S 
_TANZANIA_SRS = osr.SpatialReference()
_TANZANIA_SRS.ImportFromEPSG(_TANZANIA_EPSG)

_CIV_EPSG = 2043 # Abidjan 1987 / UTM zone 29N
_CIV_SRS = osr.SpatialReference()
_CIV_SRS.ImportFromEPSG(_CIV_EPSG)

# A dictionary that defines parameters for an area
LOCATION_MAP = {
    'sen': {
        'srs': _SENEGAL_SRS, 'dir_name': 'Senegal', 'epsg': _SENEGAL_EPSG,
        'aoi_dirs': ['sen_aoi', 'SEN_lsib_0']},
    'civ': {
        'srs': _CIV_SRS, 'dir_name': 'CIV', 'epsg': _CIV_EPSG,
        'aoi_dirs': ['civ_aoi', 'civ_ocha_lvl0']},
    'tza': {
        'srs': _TANZANIA_SRS, 'dir_name':'Tanzania', 'epsg': _TANZANIA_EPSG,
        'aoi_dirs': ['tan_aoi', 'TZA_lsib_0']},
    }

# nodata values to use throughout the script
UINT32_NODATA = int(numpy.iinfo(numpy.uint32).max)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255


def define_nodata(raster_path, output_path, nodata_value):
    """Copies a raster and sets the output nodata value."""

    def _identity_op(array):

        return array

    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], _identity_op, output_path, raster_info['datatype'],
        nodata_value)

def _gdal_warp(target_path, input_path, dstSRS, xRes=None, yRes=None):
    """Taskgraph wrapper for gdal warp.

    Reproject ``input_path`` given a spatial reference.

    Args:
        target_path (string): The path to save the reprojected raster.
        input_path (string): The path to a raster to reproject. 
        dstSRS (string): A gdal OSR spatial reference to project to.
        xRes (int): An optional pixel size to scale to
        yRes (int): An optional pixel size to scale to

    Returns:
        Nothing   
    """
    kwargs={
        'destNameOrDestDS': target_path,
        'srcDSOrSrcDSTab': input_path,
        'dstSRS': dstSRS,
        'xRes': xRes,
        'yRes': yRes,
        }
    gdal.Warp(**kwargs)

def _pop_density_to_count(pop_density_path, target_path):
    """Convert population density to population count.

    Args:
        pop_density_path (string): The path to a population density raster
        target_raster_path (string): The path to where the target raster with
            converted units should be saved

    Returns:
        Nothing
    """
    population_raster_info = pygeoprocessing.get_raster_info(pop_density_path)
    pop_pixel_area = abs(numpy.multiply(*population_raster_info['pixel_size']))

    kwargs={
        'op': lambda x: x*pop_pixel_area,
        'rasters': [pop_density_path],
        'target_path': target_path,
        'target_nodata': -1,
    }

    pygeoprocessing.raster_map(**kwargs)

def _convert_to_from_density(source_raster_path, target_raster_path,
        direction='to_density'):
    """Convert a raster to/from counts/pixel and counts/unit area.

    Args:
        source_raster_path (string): The path to a raster containing units that
            need to be converted.
        target_raster_path (string): The path to where the target raster with
            converted units should be stored.
        direction='to_density' (string): The direction of unit conversion. If
            'to_density', then the units of ``source_raster_path`` must be in
            counts per pixel and will be converted to counts per square meter.
            If 'from_density', then the units of ``source_raster_path`` must be
            in counts per square meter and will be converted to counts per
            pixel.

    Returns:
        ``None``
    """
    LOGGER.info(f'Converting {direction} {source_raster_path} --> '
                f'{target_raster_path}')
    source_raster_info = pygeoprocessing.get_raster_info(source_raster_path)
    source_nodata = source_raster_info['nodata'][0]

    # Calculate the per-pixel area based on the latitude.
    _, miny, _, maxy = source_raster_info['bounding_box']
    pixel_area_in_m2_by_latitude = (
        geoprocessing._create_latitude_m2_area_column(
            miny, maxy, source_raster_info['raster_size'][1]))

    def _convert(array, pixel_area):
        out_array = numpy.full(array.shape, FLOAT32_NODATA, dtype=numpy.float32)

        valid_mask = slice(None)
        if source_nodata is not None:
            valid_mask = ~numpy.isclose(array, source_nodata)

        if direction == 'to_density':
            out_array[valid_mask] = array[valid_mask] / pixel_area[valid_mask]
        elif direction == 'from_density':
            out_array[valid_mask] = array[valid_mask] * pixel_area[valid_mask]
        else:
            raise AssertionError(f'Invalid direction: {direction}')
        return out_array

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1), pixel_area_in_m2_by_latitude],
        _convert, target_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)


if __name__ == "__main__":
    current_input_dirs = [val['dir_name'] for key, val in LOCATION_MAP.items()] 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        help = "The the location key to preprocess data for schistosomiasis."
               " {'sen' | 'tza' | 'civ'}"
    )

    parser.add_argument(
        "--input-dir",
        help = ("Input directory which should contain the following sub"
                f" directories: {current_input_dirs}.")
        )
    parser.add_argument("--output-dir", help = "Directory to save outputs.")
    parser.add_argument(
            "--input-suffix", required=False, default='',
            help = "Suffix to use at end of input file paths.")

    args = vars(parser.parse_args())

    key_loc = args['key']
    key_suffix = args['input_suffix']

    input_dir =  os.path.join(
        args['input_dir'], LOCATION_MAP[key_loc]['dir_name'])
    input_data = os.path.join(input_dir, 'suitability layers')
    procured_data = os.path.join(input_dir, 'procured-data')
    
    preprocess_dir = os.path.join(args['output_dir'], 'preprocessed')

    # Project data to Senegal with linear units of meters
    raw_input_data = {}
    raw_input_data['water_temp_dry_raster_path'] = os.path.join(
        input_data, f'habsuit_waterTemp_dry_2019_{key_loc}{key_suffix}.tif')
    raw_input_data['water_temp_wet_raster_path'] = os.path.join(
        input_data, f'habsuit_waterTemp_wet_2019_{key_loc}{key_suffix}.tif')
    raw_input_data['ndvi_dry_raster_path'] = os.path.join(
        input_data, f'habsuit_NDVI_dry_2019_{key_loc}{key_suffix}.tif')
    raw_input_data['ndvi_wet_raster_path'] = os.path.join(
        input_data, f'habsuit_NDVI_wet_2019_{key_loc}{key_suffix}.tif')
    raw_input_data['water_presence_path'] = os.path.join(
        procured_data, f'basin_water_mask_{key_loc}.tif')
    #raw_input_data['water_presence_path'] = os.path.join(
    #    preprocess_dir, f'basin_water_mask_nodata_{key_loc}.tif')

    work_token_dir = os.path.join(preprocess_dir, '_taskgraph_working_dir')
    n_workers = -1  # Synchronous execution
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    for key, path in raw_input_data.items():
        # Before warping we need to check that nodata is defined, otherwise
        # this can leave us with artifacts where data is being created
        # during warping that is outside the original extents.
        raster_info = pygeoprocessing.get_raster_info(path)
        nodata = raster_info['nodata'][0]
        nodata_path = path
        nodata_task_list = []
        if nodata is None:
            nodata_path = os.path.join(
                preprocess_dir, f'nodata_{os.path.basename(path)}')

            nodata_task = graph.add_task(
                define_nodata,
                kwargs={
                    'raster_path': path,
                    'output_path': nodata_path,
                    'nodata_value': pygeoprocessing.choose_nodata(raster_info['datatype']),
                },
                target_path_list=[nodata_path],
                task_name=f'{key_loc} - Define nodata to {key}'
            )
            nodata_task_list.append(nodata_task)

        target_projected_path = os.path.join(
            preprocess_dir, f'projected_{os.path.basename(path)}')

        project_task = graph.add_task(
            _gdal_warp,
            kwargs={
                'target_path': target_projected_path,
                'input_path': nodata_path,
                'dstSRS': LOCATION_MAP[key_loc]['srs'],
                'xRes': 30,
                'yRes': 30,
            },
            dependent_task_list=nodata_task_list,
            target_path_list=[target_projected_path],
            task_name=f'{key_loc} - Reproject {key} to EPSG:{LOCATION_MAP[key_loc]["epsg"]}'
        )

    # Handle population special case by converting from population count to
    # population density
    # LandScan 2020 population count from Oak Ridge National Laboratory
    # What does the value in the LandScan datasets represent?
    #   The value of each cell represents an estimated population count for that cell.
    # LandScan FAQ: https://landscan.ornl.gov/about
    landscan_pop_path = os.path.join(
        procured_data, 'landscan-global-2020.tif')
    landscan_info = pygeoprocessing.get_raster_info(landscan_pop_path)
    # Need to wait for landscan_info operation
    graph.join()
    # Clip to key_loc AOIs
    for aoi_name in LOCATION_MAP[key_loc]['aoi_dirs']:
        aoi_path = os.path.join(procured_data, aoi_name, f'{aoi_name}.shp')
        population_count_path = os.path.join(
            preprocess_dir, f'{aoi_name}_landscan_2020.tif')
        landscan_mask_task = graph.add_task(
            pygeoprocessing.geoprocessing.align_and_resize_raster_stack,
            kwargs={
                'base_raster_path_list': [landscan_pop_path],
                'target_raster_path_list': [population_count_path],
                'resample_method_list': ['bilinear'],
                'target_pixel_size': landscan_info['pixel_size'],
                'bounding_box_mode': 'intersection',
                'base_vector_path_list': [aoi_path],
                'raster_align_index': 0,
                },
            target_path_list=[population_count_path],
            task_name=f'Mask landscan by AOI: {aoi_name}'
        )
        population_density_path = os.path.join(
            preprocess_dir, f'{aoi_name}_population_density.tif')
        # Population count to population density
        population_task = graph.add_task(
            _convert_to_from_density,
            kwargs={
                'source_raster_path': population_count_path,
                'target_raster_path': population_density_path,
                'direction':'to_density'
                },
            dependent_task_list=[landscan_mask_task],
            target_path_list=[population_density_path],
            task_name=f'Population count in lat/lon to density in meter: {aoi_name}'
        )
        population_projected_path = os.path.join(
            preprocess_dir, f'population_density_projected_{aoi_name}.tif')
        # Project population density raster
        project_pop_task = graph.add_task(
            _gdal_warp,
            kwargs={
                'target_path': population_projected_path,
                'input_path': population_density_path,
                'dstSRS': LOCATION_MAP[key_loc]['srs']
            },
            target_path_list=[population_projected_path],
            dependent_task_list=[population_task],
            task_name=f'Reproject pop density to EPSG: {LOCATION_MAP[key_loc]["epsg"]}:{aoi_name}'
        )
        population_proj_count_path = os.path.join(
            preprocess_dir, f'population_count_projected_{aoi_name}.tif')
        # Projected population density in meters to population count
        pop_count_task = graph.add_task(
            func=_pop_density_to_count,
            kwargs={
                'pop_density_path': population_projected_path,
                'target_path': population_proj_count_path,
            },
            target_path_list=[population_proj_count_path],
            dependent_task_list=[project_pop_task],
            task_name=f'Population meter density to meter count: {aoi_name}')

    graph.close()
    graph.join()
    graph = None

    # Already projected to local projection 32628 via voila web app
    # 30 x 30 resolution (1-arcsecond, 0.000277777777778 degrees)
    dem_path = os.path.join(procured_data, f'srtm-{key_loc}-projected.tiff')

