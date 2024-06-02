"""Setup arguments and call schisto module."""
import logging
import os
import re
import sys

import numpy

import pygeoprocessing
from pygeoprocessing import geoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr

from src.natcap.invest import schistosomiasis

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)

logging.getLogger('taskgraph').setLevel('DEBUG')

_SENEGAL_EPSG = 32628 # WGS 84 / UTM zone 28N
_SENEGAL_SRS = osr.SpatialReference()
_SENEGAL_SRS.ImportFromEPSG(_SENEGAL_EPSG)

_TANZANIA_EPSG = 21037 # Arc 1960 / UTM zonea 37S 
_TANZANIA_SRS = osr.SpatialReference()
_TANZANIA_SRS.ImportFromEPSG(_TANZANIA_EPSG)

_CIV_EPSG = 2043 # Abidjan 1987 / UTM zone 29N
_CIV_SRS = osr.SpatialReference()
_CIV_SRS.ImportFromEPSG(_CIV_EPSG)

LOCATION_MAP = {
    'sen': {'srs': _SENEGAL_SRS, 'dir_name': 'Senegal', 'epsg': _SENEGAL_EPSG},
    'civ': {'srs': _CIV_SRS, 'dir_name': 'CIV', 'epsg': _CIV_EPSG},
    'tza': {'srs': _TANZANIA_SRS, 'dir_name':'Tanzania', 'epsg': _TANZANIA_EPSG},
    }

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
    """Taskgraph wrapper for gdal warp."""
    kwargs={
        'destNameOrDestDS': target_path,
        'srcDSOrSrcDSTab': input_path,
        'dstSRS': dstSRS,
        'xRes': xRes,
        'yRes': yRes,
        }
    gdal.Warp(**kwargs)

def _pop_density_to_count(pop_density_path, target_path):
    """ """
    population_raster_info = pygeoprocessing.get_raster_info(pop_density_path)
    pop_pixel_area = abs(numpy.multiply(*population_raster_info['pixel_size']))

    kwargs={
        'op': lambda x: x*pop_pixel_area,  # convert count per pixel to meters sq to hectares
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
    key_loc = 'sen'
    input_dir =  os.path.join(
        'C:', os.sep, 'Users', 'ddenu', 'Workspace', 'Repositories',
        'schistosomiasis', 'data', LOCATION_MAP[key_loc]['dir_name'])
    input_data = os.path.join(input_dir, 'suitability layers')
    #input_data = os.path.join(
    #    'C:', os.sep, 'Users', 'ddenu', 'Workspace', 'Repositories',
    #    'schistosomiasis', 'data', 'suitability layers')
    procured_data = os.path.join(input_dir, 'procured-data')
    #procured_data = os.path.join(
    #    'C:', os.sep, 'Users', 'ddenu', 'Workspace', 'Repositories',
    #    'schistosomiasis', 'data', 'procured-data')
    preprocess_dir = os.path.join(input_dir, 'preprocessed')
    #preprocess_dir = os.path.join(
    #    'C:', os.sep, 'Users', 'ddenu', 'Workspace', 'Repositories',
    #    'schistosomiasis', 'data', 'preprocessed')
    output_dir = os.path.join(input_dir, 'output_tests_beta')
    #output_dir = os.path.join(
    #    'C:', os.sep, 'Users', 'ddenu', 'Workspace', 'Repositories',
    #    'schistosomiasis', 'data', 'output_tests_beta')
    LOGGER.debug(f'Output dir: {output_dir}')


    # Project data to Senegal with linear units of meters
    raw_input_data = {}
    raw_input_data['water_temp_dry_raster_path'] = os.path.join(
        input_data, f'habsuit_waterTemp_dry_2019_{key_loc}.tif')
    raw_input_data['water_temp_wet_raster_path'] = os.path.join(
        input_data, f'habsuit_waterTemp_wet_2019_{key_loc}.tif')
    raw_input_data['ndvi_dry_raster_path'] = os.path.join(
        input_data, f'habsuit_NDVI_dry_2019_{key_loc}.tif')
    raw_input_data['ndvi_wet_raster_path'] = os.path.join(
        input_data, f'habsuit_NDVI_wet_2019_{key_loc}.tif')
    #raw_input_data['water_presence_path'] = os.path.join(
    #    input_data, 'sen_basin_water_mask.tif')
    #raw_input_data['water_presence_path'] = os.path.join(
    #    procured_data, f'basin_water_mask_{key_loc}.tif')
    raw_input_data['water_presence_path'] = os.path.join(
        preprocess_dir, f'basin_water_mask_nodata_{key_loc}.tif')

    work_token_dir = os.path.join(preprocess_dir, '_taskgraph_working_dir')
    n_workers = -1  # Synchronous execution
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)


    args = {}
    for key, path in raw_input_data.items():
        # Before warping we need to check that nodata is deined, otherwise
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
                task_name=f'{key_loc} - Reproject {key} to EPSG:{LOCATION_MAP[key_loc]["epsg"]}'
            )
            nodata_task_list.append(nodata_task)

        target_projected_path = os.path.join(
            preprocess_dir, f'projected_{os.path.basename(path)}')
        args[key] = target_projected_path

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
    #population_count_path = os.path.join(
    #    procured_data, 'sen_pd_2020_1km_UNadj.tif')
    population_count_path = os.path.join(
        procured_data, f'{key_loc}_pd_2020_1km_UNadj.tif')
    population_density_path = os.path.join(
        preprocess_dir, 'population_density.tif')
    # Population count to population density
    population_task = graph.add_task(
        _convert_to_from_density,
        kwargs={
            'source_raster_path': population_count_path,
            'target_raster_path': population_density_path,
            'direction':'to_density'
            },
        target_path_list=[population_density_path],
        task_name=f'Population count in lat/lon to density in meter'
    )
    population_projected_path = os.path.join(
        preprocess_dir, 'population_density_projected.tif')
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
        task_name=f'Reproject pop density to EPSG: {LOCATION_MAP[key_loc]["epsg"]}'
    )
    population_proj_count_path = os.path.join(
        preprocess_dir, 'population_count_projected.tif')
    # Projected population density in meters to population count
    pop_count_task = graph.add_task(
        func=_pop_density_to_count,
        kwargs={
            'pop_density_path': population_projected_path,
            'target_path': population_proj_count_path,
        },
        target_path_list=[population_proj_count_path],
        dependent_task_list=[project_pop_task],
        task_name=f'Population meter density to meter count')

    graph.close()
    graph.join()
    graph = None

    # Already projected to local projection 32628 via voila web app
    # 30 x 30 resolution (1-arcsecond, 0.000277777777778 degrees)
    dem_path = os.path.join(procured_data, f'srtm-{key_loc}-projected.tiff')
    # For development and testing purposes use an already pit filled DEM
    # Doing this because TaskGraph was NOT reading this task as already
    # completed and was always recalculating, which was slow for development.
    #dem_path = os.path.join(procured_data, 'pit_filled_dem.tif')

    args['workspace_dir'] = output_dir
    args['results_suffix'] = ""

    args['calc_water_distance'] = True
    args['water_distance_func_type'] = 'exponential'
    args['water_distance_table_path'] = os.path.join(procured_data, 'exponential-water-distance.csv')

    args['calc_temperature'] = True
    args['temperature_func_type'] = 'default'
    args['temperature_table_path'] = ''
    #args['temperature_func_type'] = 'linear'
    #args['temperature_table_path'] = os.path.join(procured_data, 'linear-temperature.csv')

    args['calc_ndvi'] = True
    args['ndvi_func_type'] = 'default'
    args['ndvi_table_path'] = ''

    args['calc_population'] = True
    args['population_func_type'] = 'default'
    args['population_table_path'] = ''
    args['population_count_path'] = population_proj_count_path
    #args['population_count_path'] = population_count_path
    
    args['urbanization_func_type'] = 'default'
    args['urbanization_table_path'] = ''

    args['calc_water_velocity'] = True
    args['water_velocity_func_type'] = 'default'
    args['water_velocity_table_path'] = ''
    args['dem_path'] = dem_path

    schistosomiasis.execute(args)
