"""NatCap InVEST-like module for Schistosomiasis model.
There are suitability functions for each snail (BT and BG) and
parasite (SH and SM). There is a version for the wet season and the dry season;
i.e., NDVI for both wet and dry season, along with 8 NDVI suitability layers
(1 per species per season).
"""
import logging
import os
import re
import tempfile
import shutil
import sys

from natcap.invest import spec_utils
from natcap.invest import gettext
from natcap.invest import utils
from natcap.invest.spec_utils import u
from natcap.invest import validation
import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph
from osgeo import gdal
from osgeo import osr

LOGGER = logging.getLogger(__name__)

UINT32_NODATA = int(numpy.iinfo(numpy.uint32).max)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255

SCHISTO = "Schistosomiasis"

MODEL_SPEC = {
    'model_id': 'schisto',
    'model_name': SCHISTO,
    'pyname': 'natcap.invest.schistosomiasis',
    'userguide': "",
    'aliases': (),
    "ui_spec": {
        "order": [
            ['workspace_dir', 'results_suffix'],
            ['population_count_path', 'water_temp_dry_raster_path',
             'water_temp_wet_raster_path', 'ndvi_dry_raster_path',
             'ndvi_wet_raster_path', 'dem_path', 'water_presence_path']],
        "hidden": ["n_workers"],
        "forum_tag": '',
        "sampledata": {
            "filename": "Foo.zip"
        }
    },
    'args_with_spatial_overlap': {
        'spatial_keys': [
            'population_count_path', 'dem_path',
            'water_temp_dry_raster_path', 'water_temp_wet_raster_path',
            'ndvi_dry_raster_path', 'ndvi_wet_raster_path', 'water_presence_path'],
        'different_projections_ok': True,
    },
    'args': {
        'workspace_dir': spec_utils.WORKSPACE,
        'results_suffix': spec_utils.SUFFIX,
        'n_workers': spec_utils.N_WORKERS,
        'population_count_path': {
            'type': 'raster',
            'name': 'population raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the number of inhabitants per pixel."
            ),
        },
        'water_temp_dry_raster_path': {
            'type': 'raster',
            'name': 'water temp dry raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the water temp for dry season."
            ),
        },
        'water_temp_wet_raster_path': {
            'type': 'raster',
            'name': 'water temp wet raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the water temp for wet season."
            ),
        },
        'ndvi_dry_raster_path': {
            'type': 'raster',
            'name': 'ndvi dry raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the ndvi for dry season."
            ),
        },
        'ndvi_wet_raster_path': {
            'type': 'raster',
            'name': 'ndvi wet raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the ndvi for wet season."
            ),
        },
        'water_presence_path': {
            'type': 'raster',
            'name': 'water presence',
            'bands': {1: {'type': 'integer'}},
            'about': (
                "A raster indicating presence of water."
            ),
        },
        'dem_path': {
            **spec_utils.DEM,
            "projected": True
        },
#        "biophysical_table_path": {
#            "type": "csv",
#            "index_col": "suit_factor",
#            "columns": {
#                "suit_factor": {
#                    "type": "string",
#                    "about": gettext("Suitability factor")},
#                "function_type": {
#                    "type": "string",
#                    "about": gettext("Function type to model")},
#                "usle_p": {
#                    "type": "number",
#                    "about": gettext("Support practice factor for the USLE")}
#            },
#            "about": gettext(
#                "A table mapping each suitibility factor to a function."),
#            "name": gettext("biophysical table")
#        },
    },
    'outputs': {
        'output': {
            "type": "directory",
            "contents": {
                "water_temp_suit_dry_sm.tif": {
                    "about": "",
                    "bands": {1: {
                        "type": "number",
                        "units": u.m**2,
                    }}},
                },
            },
        'intermediate': {
            'type': 'directory',
            'contents': {
                '_taskgraph_working_dir': spec_utils.TASKGRAPH_DIR,
            },
        }
    }
}


_OUTPUT_BASE_FILES = {
    'water_temp_suit_dry_sm': 'water_temp_suit_dry_sm.tif',
    'water_temp_suit_wet_sm': 'water_temp_suit_wet_sm.tif',
    'ndvi_suit_dry_sm': 'ndvi_suit_dry_sm.tif',
    'ndvi_suit_wet_sm': 'ndvi_suit_wet_sm.tif',
    'water_velocity_suit': 'water_velocity_suit.tif',
    'water_proximity_suit': 'water_proximity_suit.tif',
    'rural_pop_suit': 'rural_pop_suit.tif',
    #'water_stability_suit': 'water_stability_suit.tif',
}

_INTERMEDIATE_BASE_FILES = {
    'aligned_population': 'aligned_population.tif',
    'masked_population': 'masked_population.tif',
    'aligned_water_temp_dry': 'aligned_water_temp_dry.tif',
    'aligned_water_temp_wet': 'aligned_water_temp_wet.tif',
    'aligned_ndvi_dry': 'aligned_ndvi_dry.tif',
    'aligned_ndvi_wet': 'aligned_ndvi_wet.tif',
    'aligned_dem': 'aligned_dem.tif',
    'pit_filled_dem': 'pit_filled_dem.tif',
    'slope': 'slope.tif',
    'aligned_water_presence': 'aligned_water_presence.tif',
    'aligned_lulc': 'aligned_lulc.tif',
    'masked_lulc': 'masked_lulc.tif',
    'aligned_mask': 'aligned_valid_pixels_mask.tif',
    'reprojected_admin_boundaries': 'reprojected_admin_boundaries.gpkg',
}


def execute(args):
    """Schistosomiasis.

    Args:
        args['workspace_dir'] (string): (required) Output directory for
            intermediate, temporary and final files.
        args['results_suffix'] (string): (optional) String to append to any
            output file.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for executing the tasks of this model.  If omitted, computation
            will take place in the current process.

        args['lulc_raster_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.

        args['water_presence_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.

        args['ndvi_dry_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.
        args['ndvi_wet_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.

        args['water_temp_dry_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.
        args['water_temp_wet_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.

        args['population_count_path'] (string): (required) A string path to a
            GDAL-compatible population raster containing people count per
            square km.  Must be linearly projected in meters.
        args['dem_path'] (string): (required) A string path to a
            GDAL-compatible population raster containing people count per
            square km.  Must be linearly projected in meters.

    """
    LOGGER.info(f"Execute {SCHISTO}")

    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    utils.make_directories([output_dir, intermediate_dir])

    suffix = utils.make_suffix_string(args, 'results_suffix')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_dir)],
        suffix)

    work_token_dir = os.path.join(intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous execution
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    ### Align and set up datasets
    #Questions
    # 1) what should rasters be aligned to? What is the resolution to do operations on?
    # 2) should we align and resize at the end or up front?

    squared_default_pixel_size = _square_off_pixels(args['water_temp_wet_raster_path'])

    raster_input_list = [
        args['water_temp_dry_raster_path'],
        args['water_temp_wet_raster_path'],
        args['water_presence_path'],
        args['ndvi_dry_raster_path'],
        args['ndvi_wet_raster_path'],
        args['dem_path']]
    aligned_input_list = [
        file_registry['aligned_water_temp_dry'],
        file_registry['aligned_water_temp_wet'],
        file_registry['aligned_water_presence'],
        file_registry['aligned_ndvi_dry'],
        file_registry['aligned_ndvi_wet'],
        file_registry['aligned_dem']]

    align_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        kwargs={
            'base_raster_path_list': raster_input_list,
            'target_raster_path_list': aligned_input_list,
            'resample_method_list': ['near']*len(raster_input_list),
            'target_pixel_size': squared_default_pixel_size,
            'bounding_box_mode': 'union',
        },
        target_path_list=aligned_input_list,
        task_name='Align and resize input rasters'
    )
    align_task.join()

    raster_info = pygeoprocessing.get_raster_info(file_registry['aligned_water_temp_wet'])
    default_bb = raster_info['bounding_box']
    default_wkt = raster_info['projection_wkt']

    # NOTE: Need to handle population differently in case of scaling
    population_align_task = graph.add_task(
        _resample_population_raster,
        kwargs={
            'source_population_raster_path': args['population_count_path'],
            'target_population_raster_path': file_registry['aligned_population'],
            'lulc_pixel_size': squared_default_pixel_size,
            'lulc_bb': default_bb,
            'lulc_projection_wkt': default_wkt,
            'working_dir': intermediate_dir,
        },
        target_path_list=[file_registry['aligned_population']],
        task_name='Align and resize population'
    )

    ### Production functions

    ### Habitat stability
#    habitat_stability_task = graph.add_task(
#        _habitat_stability,
#        kwargs={
#            'surface_water_presence': args['surface_water_presence'],
#            'target_raster_path': file_registry['habitat_stability'],
#        },
#        target_path_list=[file_registry['aligned_population']],
#        task_name='Resample population to LULC resolution')

    # fill DEM pits
#    pit_fill_task = graph.add_task(
#        func=pygeoprocessing.routing.fill_pits,
#        args=(
#            (file_registry['aligned_dem'], 1),
#            file_registry['pit_filled_dem']),
#        target_path_list=[file_registry['pit_filled_dem']],
#        dependent_task_list=[align_task],
#        task_name='fill pits')

    # calculate slope
    slope_task = graph.add_task(
        func=pygeoprocessing.calculate_slope,
        args=(
            (args['dem_path'], 1),
            file_registry['slope']),
        dependent_task_list=[align_task],
        target_path_list=[file_registry['slope']],
        task_name='calculate slope')

    ### Water velocity
    water_vel_task = graph.add_task(
        _water_velocity_sm,
        kwargs={
            'slope_path': file_registry[f'slope'],
            'target_raster_path': file_registry[f'water_velocity_suit'],
        },
        dependent_task_list=[slope_task],
        target_path_list=[file_registry[f'water_velocity_suit']],
        task_name=f'Water Velocity Suit')

    ### Proximity to water
    water_proximity_task = graph.add_task(
        _water_proximity,
        kwargs={
            'water_presence_path': file_registry[f'aligned_water_presence'],
            'target_raster_path': file_registry[f'water_proximity_suit'],
        },
        dependent_task_list=[align_task],
        target_path_list=[file_registry[f'water_proximity_suit']],
        task_name=f'Water Proximity Suit')

    ### Rural population density
    rural_pop_task = graph.add_task(
        _rural_population_density,
        kwargs={
            'population_path': file_registry[f'aligned_population'],
            'target_raster_path': file_registry[f'rural_pop_suit'],
        },
        dependent_task_list=[population_align_task],
        target_path_list=[file_registry[f'rural_pop_suit']],
        task_name=f'Rural Population Suit')

    for season in ["dry", "wet"]:
        ### Water temperature
        water_temp_task = graph.add_task(
            _water_temp_sm,
            kwargs={
                'water_temp_path': file_registry[f'aligned_water_temp_{season}'],
                'target_raster_path': file_registry[f'water_temp_suit_{season}_sm'],
            },
            dependent_task_list=[align_task],
            target_path_list=[file_registry[f'water_temp_suit_{season}_sm']],
            task_name=f'Water Temp Suit for {season} SM')

        ### Vegetation coverage (NDVI)
        ndvi_task = graph.add_task(
            _ndvi_sm,
            kwargs={
                'ndvi_path': file_registry[f'aligned_ndvi_{season}'],
                'target_raster_path': file_registry[f'ndvi_suit_{season}_sm'],
            },
            dependent_task_list=[align_task],
            target_path_list=[file_registry[f'ndvi_suit_{season}_sm']],
            task_name=f'NDVI Suit for {season} SM')


    ### Population proximity to water

    ### Population density


    # save function shape plots




    graph.close()
    graph.join()

    LOGGER.info("Model completed")


def _habitat_stability(water_presence_path, target_raster_path):
    """ """
    water_presence_info = pygeoprocessing.get_raster_info(water_presence_path)
    water_presence_nodata = water_presence_info['nodata'][0]

    def op(array):
        output = numpy.full(
            array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (
            ~numpy.isclose(array, water_presence_nodata))

        # values with 1 month or less surface water presence set to 0
        lte_one_mask = (array <= 1) & valid_pixels
        output[lte_one_mask] = 0

        between_mask = (array > 1) & (array <= 1.75) & valid_pixels
        output[between_mask] = 1.33 * array[between_mask] - 1.33

        gt_mask = (array > 1.75) & valid_pixels
        output[gt_mask] = 1

        return output

    pygeoprocessing.raster_calculator(
        [(water_presence_path, 1)],
        op, target_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

def _water_temp_sm(water_temp_path, target_raster_path):
    """ """
    #SmWaterTemp <- function(Temp){ifelse(Temp<16, 0,ifelse(Temp<=35, -0.003 * (268/(Temp - 14.2) - 335) + 0.0336538, 0))}
    water_temp_info = pygeoprocessing.get_raster_info(water_temp_path)
    water_temp_nodata = water_temp_info['nodata'][0]
    def op(temp_array):
        output = numpy.full(
            temp_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(temp_array, water_temp_nodata))

        # if temp is less than 16 set to 0
        valid_range_mask = (temp_array>=16) & (temp_array<=35)
        output[valid_pixels] = (
            -0.003 * (268 / (temp_array[valid_pixels] - 14.2) - 335) + 0.0336538)
        output[~valid_range_mask] = 0
        output[~valid_pixels] = FLOAT32_NODATA

        return output

    pygeoprocessing.raster_calculator(
        [(water_temp_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _ndvi_sm(ndvi_path, target_raster_path):
    """ """
    #VegCoverage <- function(V){ifelse(V<0,0,ifelse(V<=0.3,3.33*V,1))}
    ndvi_info = pygeoprocessing.get_raster_info(ndvi_path)
    ndvi_nodata = ndvi_info['nodata'][0]
    def op(ndvi_array):
        output = numpy.full(
            ndvi_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(ndvi_array, ndvi_nodata))

        # if temp is less than 0 set to 0
        output[valid_pixels] = (3.33 * ndvi_array[valid_pixels])
        output[valid_pixels & (ndvi_array < 0)] = 0
        output[valid_pixels & (ndvi_array > 0.3)] = 1
        output[~valid_pixels] = FLOAT32_NODATA

        return output

    pygeoprocessing.raster_calculator(
        [(ndvi_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _water_proximity(water_presence_path, target_raster_path):
    """ """
    #ProxRisk <- function(prox){ifelse(prox<1000, 1,ifelse(prox<=15000, -0.0000714 * prox + 1.0714,0))}
    water_presence_info = pygeoprocessing.get_raster_info(water_presence_path)
    water_presence_nodata = water_presence_info['nodata'][0]
    def op(water_presence_array):
        output = numpy.full(
            water_presence_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(water_presence_array, water_presence_nodata))

        # if temp is less than 0 set to 0
        output[valid_pixels] = (3.33 * water_presence_array[valid_pixels])
        output[valid_pixels & (water_presence_array < 0)] = 0
        output[valid_pixels & (water_presence_array > 0.3)] = 1
        output[~valid_pixels] = FLOAT32_NODATA

        return output

    pygeoprocessing.raster_calculator(
        [(water_presence_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _urbanization(surface_water_presence, target_raster_path):
    """ """
    #UrbanRisk <- function(h){ifelse(h<1,1,1/(1+exp((h-3)/0.4)))}
    pass

def _rural_population_density(population_path, target_raster_path):
    """ """
    #RuralDenRisk <- function(h){ifelse(h<1,h,1)}
    population_info = pygeoprocessing.get_raster_info(population_path)
    population_nodata = population_info['nodata'][0]
    def op(population_array):
        output = numpy.full(
            population_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(population_array, population_nodata))

        output[valid_pixels & (population_array < 1)] = population_array[valid_pixels & (population_array < 1)]
        output[valid_pixels & (population_array >= 1)] = 1

        return output

    pygeoprocessing.raster_calculator(
        [(population_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _water_velocity_sm(slope_path, target_raster_path):
    """Slope suitability. """
    #WaterVel <- function(S){ifelse(S<=0.00014,-5714.3 * S + 1,-0.0029*S+0.2)}
    slope_info = pygeoprocessing.get_raster_info(slope_path)
    slope_nodata = slope_info['nodata'][0]

    def op(slope_array):
        output = numpy.full(
            slope_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(slope_array, slope_nodata))

        # percent slope to degrees
        output[valid_pixels] = numpy.degrees(numpy.arctan(slope_array[valid_pixels] / 100.0))
        mask_lt = valid_pixels & (output <= 0.00014)
        output[mask_lt] = -5714.3 * output[mask_lt] + 1
        mask_gt = valid_pixels & (output > 0.00014)
        output[mask_gt] = -0.0029 * output[mask_gt] + 0.2

        return output

    pygeoprocessing.raster_calculator(
        [(slope_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _trapezoid_op(raster_path):
    """ """
    #'y = y1 - (y2 - y1)/(x2-x1)  * x1 + (y2 - y1)/(x2-x1) * x 
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]
    # Need to define the shape of the trapezoid
    xa = 12   # Start of first linear equation
    ya = 0    # Value of left initial plateau
    xz = 40   # End of second linear equation
    yz = 0    # Value of trailing plateau (should be same as initial)

    xb = 20   # End of first linear equation (start of middle plateau)
    yb = 1    # Value of middle plateau
    xc = 30   # Start of second linear equation (end of middle plateau)
    yc = 1    # Value of middle plateau (same as yb)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        # First plateau
        mask_one = valid_pixels & (raster_array <= xa)
        output[mask_one] = ya
        # First slope
        # Second plateau
        mask_three = valid_pixels & (raster_array >= xb) & (raster_array <= xc)
        output[mask_three] = yb
        # Second slope
        # Third plateau
        mask_three = valid_pixels & (raster_array >= xz)
        output[mask_three] = yz

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _gaussian_op(raster_path, target_raster_path, loc=0, scale=1, lb=0, ub=40):
    """ """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    rv = scipy.stats.norm(loc=loc, scale=scale)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        output[valid_pixels] = rv.pdf(raster_array[valid_pixels])
        bounds_mask = valid_pixels & (raster_array <= lb) & (raster_array >= ub)
        output[bounds_mask] = 0

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _sshape_op(raster_path, target_raster_path, yin=1, yfin=0, xmed=15, inv_slope=3):
    """ """
    #y = yin + (yfin - yin)/(1 + exp(-(x - xmed)/invSlope)))
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        output[valid_pixels] = yin + (yfin - yin) / (1 + numpy.exp(-1 * ((raster_array[valid_pixels]) - xmed) / inv_slope))

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _exponential_decay_op(raster_path, target_raster_path, scalar=1, max_dist=1000):
    """ """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]
    xmed = 1
    decay_factor = 0.982

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        output[valid_pixels] = numpy.exp(-1 * (scalar / max_dist) * raster_array[valid_pixels])

        output[valid_pixels & raster_array < xmed] = 1
        #exp_mask = valid_pixels & (raster_array >= xmed)
        #output[exp_mask] = yin * (decay_factor**raster_array[exp_mask])


        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _linear_op(raster_path, target_raster_path, increasing=True):
    """ """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]
    intercept = 0
    if increasing:
        constant = 1
    else:
        constant = -1

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        output[valid_pixels] = constant * raster_array[valid_pixels] + intercept

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _square_off_pixels(raster_path):
    """Create square pixels from the provided raster.

    The pixel dimensions produced will respect the sign of the original pixel
    dimensions and will be the mean of the absolute source pixel dimensions.

    Args:
        raster_path (string): The path to a raster on disk.

    Returns:
        A 2-tuple of ``(pixel_width, pixel_height)``, in projected units.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_width, pixel_height = raster_info['pixel_size']

    if abs(pixel_width) == abs(pixel_height):
        return (pixel_width, pixel_height)

    pixel_tuple = ()
    average_absolute_size = (abs(pixel_width) + abs(pixel_height)) / 2
    for pixel_dimension_size in (pixel_width, pixel_height):
        # This loop allows either or both pixel dimension(s) to be negative
        sign_factor = 1
        if pixel_dimension_size < 0:
            sign_factor = -1

        pixel_tuple += (average_absolute_size * sign_factor,)

    return pixel_tuple


def _resample_population_raster(
        source_population_raster_path, target_population_raster_path,
        lulc_pixel_size, lulc_bb, lulc_projection_wkt, working_dir):
    """Resample a population raster without losing or gaining people.

    Population rasters are an interesting special case where the data are
    neither continuous nor categorical, and the total population count
    typically matters.  Common resampling methods for continuous
    (interpolation) and categorical (nearest-neighbor) datasets leave room for
    the total population of a resampled raster to significantly change.  This
    function resamples a population raster with the following steps:

        1. Convert a population count raster to population density per pixel
        2. Warp the population density raster to the target spatial reference
           and pixel size using bilinear interpolation.
        3. Convert the warped density raster back to population counts.

    Args:
        source_population_raster_path (string): The source population raster.
            Pixel values represent the number of people occupying the pixel.
            Must be linearly projected in meters.
        target_population_raster_path (string): The path to where the target,
            warped population raster will live on disk.
        lulc_pixel_size (tuple): A tuple of the pixel size for the target
            raster.  Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_bb (tuple): A tuple of the bounding box for the target raster.
            Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_projection_wkt (string): The Well-Known Text of the target
            spatial reference fro the target raster.  Passed directly to
            ``pygeoprocessing.warp_raster``.  Assumed to be a linear projection
            in meters.
        working_dir (string): The path to a directory on disk.  A new directory
            is created within this directory for the storage of temporary files
            and then deleted upon successful completion of the function.

    Returns:
        ``None``
    """
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    tmp_working_dir = tempfile.mkdtemp(dir=working_dir)
    population_raster_info = pygeoprocessing.get_raster_info(
        source_population_raster_path)
    pixel_area = numpy.multiply(*population_raster_info['pixel_size'])
    population_nodata = population_raster_info['nodata'][0]

    population_srs = osr.SpatialReference()
    population_srs.ImportFromWkt(population_raster_info['projection_wkt'])

    # Convert population pixel area to square km
    population_pixel_area = (
        pixel_area * population_srs.GetLinearUnits()) / 1e6

    def _convert_population_to_density(population):
        """Convert population counts to population per square km.

        Args:
            population (numpy.array): A numpy array where pixel values
                represent the number of people who reside in a pixel.

        Returns:
            """
        out_array = numpy.full(
            population.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_mask = ~utils.array_equals_nodata(population, population_nodata)
        out_array[valid_mask] = population[valid_mask] / population_pixel_area
        return out_array

    # Step 1: convert the population raster to population density per sq. km
    density_raster_path = os.path.join(tmp_working_dir, 'pop_density.tif')
    pygeoprocessing.raster_calculator(
        [(source_population_raster_path, 1)],
        _convert_population_to_density,
        density_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

    # Step 2: align to the LULC
    warped_density_path = os.path.join(tmp_working_dir, 'warped_density.tif')
    pygeoprocessing.warp_raster(
        density_raster_path,
        target_pixel_size=lulc_pixel_size,
        target_raster_path=warped_density_path,
        resample_method='bilinear',
        target_bb=lulc_bb,
        target_projection_wkt=lulc_projection_wkt)

    # Step 3: convert the warped population raster back from density to the
    # population per pixel
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(lulc_projection_wkt)
    # Calculate target pixel area in km to match above
    target_pixel_area = (
        numpy.multiply(*lulc_pixel_size) * target_srs.GetLinearUnits()) / 1e6

    def _convert_density_to_population(density):
        """Convert a population density raster back to population counts.

        Args:
            density (numpy.array): An array of the population density per
                square kilometer.

        Returns:
            A ``numpy.array`` of the population counts given the target pixel
            size of the output raster."""
        # We're using a float32 array here because doing these unit
        # conversions is likely to end up with partial people spread out
        # between multiple pixels.  So it's preserving an unrealistic degree of
        # precision, but that's probably OK because pixels are imprecise
        # measures anyways.
        out_array = numpy.full(
            density.shape, FLOAT32_NODATA, dtype=numpy.float32)

        # We already know that the nodata value is FLOAT32_NODATA
        valid_mask = ~numpy.isclose(density, FLOAT32_NODATA)
        out_array[valid_mask] = density[valid_mask] * target_pixel_area
        return out_array

    pygeoprocessing.raster_calculator(
        [(warped_density_path, 1)],
        _convert_density_to_population,
        target_population_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

    shutil.rmtree(tmp_working_dir, ignore_errors=True)


@validation.invest_validator
def validate(args, limit_to=None):
    return validation.validate(
        args, MODEL_SPEC['args'], MODEL_SPEC['args_with_spatial_overlap'])
