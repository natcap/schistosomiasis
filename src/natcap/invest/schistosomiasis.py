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
from pygeoprocessing import geoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr

import numpy
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

logging.getLogger('taskgraph').setLevel('DEBUG')

UINT32_NODATA = int(numpy.iinfo(numpy.uint32).max)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255

SCHISTO = "Schistosomiasis"

SPEC_FUNC_TYPES = {
    "type": "option_string",
    "options": {
        "default": {"display_name": gettext("Default used in paper.")},
        "linear": {"display_name": gettext("Linear")},
        "exponential": {"display_name": gettext("exponential")},
        "scurve": {"display_name": gettext("scurve")},
        "trapezoid": {"display_name": gettext("trapezoid")},
        "gaussian": {"display_name": gettext("gaussian")},
    },
    "about": gettext(
        "The function type to apply to the suitability factor."),
    "name": gettext("Suitability function type")
}
SPEC_FUNC_COLS = {
    'linear': {
        "xa": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "ya": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
        "xz": { "type": "number", "about": gettext(
                "Second points x coordinate that defines the line.")},
        "yz": { "type": "number", "about": gettext(
                "Second points y coordinate that defines the line.")},
    },
    'trapezoid': {
        "xa": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "ya": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
        "xb": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "yb": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
        "xc": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "yc": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
        "xz": { "type": "number", "about": gettext(
                "Second points x coordinate that defines the line.")},
        "yz": { "type": "number", "about": gettext(
                "Second points y coordinate that defines the line.")},
    },
    'gaussian': {
        "mean": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "std": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
        "lb": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "ub": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
    },
    'sshape': {
        "yin": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "yfin": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
        "xmed": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "inv_slope": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
    },
    'exponential': {
        "yin": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "xmed": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
        "decay_factor": {"type": "number", "about": gettext(
                "First points x coordinate that defines the line.")},
        "max_dist": { "type": "number", "about": gettext(
                "First points y coordinate that defines the line.")},
    },

}

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
        "calc_population": {
            "type": "boolean",
            "about": gettext("Calculate population."),
            "name": gettext("calculate population")
        },
        "population_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_population",
        },
        "population_table_path": {
            "type": "csv",
            #"index_col": "suit_factor",
            #"columns": **SPEC_FUNC_COLS['population_func_type'],
            "required": "population_func_type != default",
            "about": gettext(
                "A table mapping each suitibility factor to a function."),
            "name": gettext("population table")
        },
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
        "calc_water_distance": {
            "type": "boolean",
            "about": gettext("Calculate water_distance."),
            "name": gettext("calculate water_distance")
        },
        "water_distance_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_water_distance",
        },
        "water_distance_table_path": {
            "type": "csv",
            #"index_col": "suit_factor",
            #"columns": **SPEC_FUNC_COLS['water_distance_func_type'],
            "required": "water_distance_func_type != default",
            "about": gettext(
                "A table mapping each suitibility factor to a function."),
            "name": gettext("water_distance table")
        },
        'water_presence_path': {
            'type': 'raster',
            'name': 'water presence',
            'bands': {1: {'type': 'integer'}},
            'about': (
                "A raster indicating presence of water."
            ),
        },
        "calc_water_velocity": {
            "type": "boolean",
            "about": gettext("Calculate water_velocity."),
            "name": gettext("calculate water_velocity")
        },
        "water_velocity_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_water_velocity",
        },
        "water_velocity_table_path": {
            "type": "csv",
            #"index_col": "suit_factor",
            #"columns": **SPEC_FUNC_COLS['water_velocity_func_type'],
            "required": "water_velocity_func_type != default",
            "about": gettext(
                "A table mapping each suitibility factor to a function."),
            "name": gettext("water_velocity table")
        },
        'dem_path': {
            **spec_utils.DEM,
            "projected": True
        },
        "calc_temperature": {
            "type": "boolean",
            "about": gettext("Calculate temperature."),
            "name": gettext("calculate temperature")
        },
        "temperature_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_temperature",
        },
        "temperature_table_path": {
            "type": "csv",
            #"index_col": "suit_factor",
            #"columns": **SPEC_FUNC_COLS['temperature_func_type'],
            "required": "temperature_func_type != default",
            "about": gettext(
                "A table mapping each suitibility factor to a function."),
            "name": gettext("temperature table")
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
            "required": "calc_temperature",
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
            "required": "calc_temperature",
        },
        "calc_ndvi": {
            "type": "boolean",
            "about": gettext("Calculate ndvi."),
            "name": gettext("calculate ndvi")
        },
        "ndvi_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_ndvi",
        },
        "ndvi_table_path": {
            "type": "csv",
            #"index_col": "suit_factor",
            #"columns": **SPEC_FUNC_COLS['ndvi_func_type'],
            "required": "ndvi_func_type != default",
            "about": gettext(
                "A table mapping each suitibility factor to a function."),
            "name": gettext("ndvi table")
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
            "required": "calc_ndvi",
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
            "required": "calc_ndvi",
        },
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
    'population_density': 'population_density.tif',
    'population_hectares': 'population_hectare.tif',
    'aligned_water_temp_dry': 'aligned_water_temp_dry.tif',
    'aligned_water_temp_wet': 'aligned_water_temp_wet.tif',
    'aligned_ndvi_dry': 'aligned_ndvi_dry.tif',
    'aligned_ndvi_wet': 'aligned_ndvi_wet.tif',
    'aligned_dem': 'aligned_dem.tif',
    'pit_filled_dem': 'pit_filled_dem.tif',
    'slope': 'slope.tif',
    'degree_slope': 'degree_slope.tif',
    'aligned_water_presence': 'aligned_water_presence.tif',
    'aligned_lulc': 'aligned_lulc.tif',
    'masked_lulc': 'masked_lulc.tif',
    'aligned_mask': 'aligned_valid_pixels_mask.tif',
    'reprojected_admin_boundaries': 'reprojected_admin_boundaries.gpkg',
    'distance': 'distance.tif',
    'not_water_mask': 'not_water_mask.tif',
    'inverse_distance': 'inverse_distance.tif',
    'water_velocity_suit_plot': 'water_vel_suit_plot.png',
    'water_proximity_suit_plot': 'water_proximity_suit_plot.png',
    'water_temp_suit_dry_plot': 'water_temp_suit_dry_plot.png',
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

        args['calc_ndvi'] = True
        args['ndvi_func_type'] = 'default'
        args['ndvi_table_path'] = ''
        args['ndvi_dry_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.
        args['ndvi_wet_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.

        args['calc_temperature'] = True
        args['temperature_func_type'] = 'linear'
        args['temperature_table_path'] = os.path.join(procured_data, 'linear-temp.csv')
        args['water_temp_dry_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.
        args['water_temp_wet_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.

        args['calc_population'] = True
        args['population_func_type'] = 'default'
        args['population_table_path'] = ''
        args['population_count_path'] (string): (required) A string path to a
            GDAL-compatible population raster containing people count per
            square km.  Must be linearly projected in meters.

        args['calc_water_velocity'] = True
        args['water_velocity_func_type'] = 'default'
        args['water_velocity_table_path'] = ''
        args['dem_path'] (string): (required) A string path to a
            GDAL-compatible population raster containing people count per
            square km.  Must be linearly projected in meters.

    """
    LOGGER.info(f"Execute {SCHISTO}")

    FUNC_TYPES = {
        'trapezoid': _trapezoid_op,
        'linear': _linear_op,
        'exponential': _exponential_decay_op,
        's-curve': _sshape_op,
        'gaussian': _gaussian_op,
        }
    DEFAULT_FUNC_TYPES = {
        'temperature': _water_temp_sm,
        'ndvi': _ndvi_sm,
        'population': _rural_population_density,
        'water_velocity': _water_velocity_sm,
        'water_distance': _water_proximity,
        }
    PLOT_PARAMS = {
        'temperature': (0, 50),
        'ndvi': (-1, 1),
        'population': (0, 10),
        'water_velocity': (0, 30),
        'water_distance': (0, 1000),
        }

    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    func_plot_dir = os.path.join(intermediate_dir, 'plot_previews')
    utils.make_directories([output_dir, intermediate_dir, func_plot_dir])

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

    ### Display function choices
    # Read func params from table
    user_func_paths = [
        'temperature', 'ndvi', 'population', 'water_distance',
        'water_velocity']
    suit_func_to_use = {}
    for suit_key in user_func_paths:
        table_spec = MODEL_SPEC['args'][f'{suit_key}_table_path']
        func_type = args[f'{suit_key}_func_type']
        if func_type != 'default':
            table_spec['columns'] = SPEC_FUNC_COLS[func_type]
            func_params = utils.read_csv_to_dataframe(
                args[f'{suit_key}_table_path']).to_dict(orient='list')
            func_params = {
                key: val[0] for key, val in func_params.items()}
            user_func = FUNC_TYPES[func_type]
        else:
            func_params = None
            user_func = DEFAULT_FUNC_TYPES[suit_key]

        suit_func_to_use[suit_key] = {
            'func_name':user_func,
            'func_params':func_params
            }
        results = _generic_func_values(
            user_func, PLOT_PARAMS[suit_key], intermediate_dir, func_params)
        plot_path = os.path.join(func_plot_dir, f"{suit_key}-{func_type}.png")
        _plotter(
            results[0], results[1], save_path=plot_path,
            label_x=suit_key, label_y=func_type,
            title=f'{suit_key}--{func_type}', xticks=None, yticks=None)

    ### Align and set up datasets
    # Questions:
    # 1) what should rasters be aligned to? What is the resolution to do operations on?
    # 2) should we align and resize at the end or up front?

    squared_default_pixel_size = _square_off_pixels(
        args['water_temp_wet_raster_path'])

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
            'bounding_box_mode': 'intersection',
        },
        target_path_list=aligned_input_list,
        task_name='Align and resize input rasters'
    )
    align_task.join()

    raster_info = pygeoprocessing.get_raster_info(
        file_registry['aligned_water_temp_wet'])
    default_bb = raster_info['bounding_box']
    default_wkt = raster_info['projection_wkt']
    default_pixel_size = raster_info['pixel_size']

    # NOTE: Need to handle population differently in case of scaling
    # Returns population count
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


    ### Water velocity
    # fill DEM pits
    # NOTE: TODO
    # Skipping this currently because TG was always recalculating this step.
    # So for dev purposes I'm just passing in an already pit filled DEM
    pit_fill_task = graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (file_registry['aligned_dem'], 1),
            file_registry['pit_filled_dem']),
        target_path_list=[file_registry['pit_filled_dem']],
        dependent_task_list=[align_task],
        task_name='fill pits')

    # calculate slope
    slope_task = graph.add_task(
        func=pygeoprocessing.calculate_slope,
        args=(
            (file_registry['pit_filled_dem'], 1),
            file_registry['slope']),
        dependent_task_list=[pit_fill_task],
        target_path_list=[file_registry['slope']],
        task_name='calculate slope')

    degree_task = graph.add_task(
        pygeoprocessing.raster_map,
        kwargs={
            'op': _degree_op,
            'rasters': [file_registry['slope']],
            'target_path': file_registry['degree_slope'],
            'target_nodata': -9999,
        },
        dependent_task_list=[slope_task],
        target_path_list=[file_registry['degree_slope']],
        task_name=f'Slope percent to degree')

    water_vel_task = graph.add_task(
        suit_func_to_use['water_velocity']['func_name'],
        args=(file_registry[f'slope'], file_registry['water_velocity_suit']),
        kwargs=suit_func_to_use['water_velocity']['func_params'],
        dependent_task_list=[slope_task],
        target_path_list=[file_registry['water_velocity_suit']],
        task_name=f'Water Velocity Suit')

    ### Proximity to water in meters
    dist_edt_task = graph.add_task(
        func=pygeoprocessing.distance_transform_edt,
        args=(
            (file_registry['aligned_water_presence'], 1),
            file_registry['distance'],
            (default_pixel_size[0], default_pixel_size[0])),
        target_path_list=[file_registry['distance']],
        dependent_task_list=[align_task],
        task_name='distance edt')

    water_proximity_task = graph.add_task(
        #_water_proximity,
        suit_func_to_use['water_distance']['func_name'],
        args=(file_registry['distance'], file_registry['water_proximity_suit']),
        kwargs=suit_func_to_use['water_distance']['func_params'],
        dependent_task_list=[dist_edt_task],
        target_path_list=[file_registry[f'water_proximity_suit']],
        task_name=f'Water Proximity Suit')

    ### Rural population density
    # Population count to density in hectares
    pop_hectare_task = graph.add_task(
        func=_pop_count_to_density,
        kwargs={
            'pop_count_path': file_registry['aligned_population'],
            'target_path': file_registry['population_hectares'],
        },
        target_path_list=[file_registry['population_hectares']],
        dependent_task_list=[population_align_task],
        task_name=f'Population count to density in hectares.')

    rural_pop_task = graph.add_task(
        #_rural_population_density,
        suit_func_to_use['population']['func_name'],
        args=(
            file_registry['population_hectares'],
            file_registry['rural_pop_suit']),
        kwargs=suit_func_to_use['population']['func_params'],
        dependent_task_list=[pop_hectare_task],
        target_path_list=[file_registry['rural_pop_suit']],
        task_name=f'Rural Population Suit')

    for season in ["dry", "wet"]:
        ### Water temperature
        water_temp_task = graph.add_task(
            #_water_temp_sm,
            suit_func_to_use['temperature']['func_name'],
            args=(
                file_registry[f'aligned_water_temp_{season}'],
                file_registry[f'water_temp_suit_{season}_sm'],
            ),
            kwargs=suit_func_to_use['temperature']['func_params'],
            dependent_task_list=[align_task],
            target_path_list=[file_registry[f'water_temp_suit_{season}_sm']],
            task_name=f'Water Temp Suit for {season} SM')

        ### Vegetation coverage (NDVI)
        ndvi_task = graph.add_task(
            #_ndvi_sm,
            suit_func_to_use['ndvi']['func_name'],
            args=(
                file_registry[f'aligned_ndvi_{season}'],
                file_registry[f'ndvi_suit_{season}_sm'],
            ),
            kwargs=suit_func_to_use['ndvi']['func_params'],
            dependent_task_list=[align_task],
            target_path_list=[file_registry[f'ndvi_suit_{season}_sm']],
            task_name=f'NDVI Suit for {season} SM')

    not_water_mask_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs={
            'op': numpy.logical_not,
            'rasters': [file_registry['aligned_water_presence']],
            'target_path': file_registry['not_water_mask'],
            'target_nodata': 255,
            },
        target_path_list=[file_registry['not_water_mask']],
        dependent_task_list=[align_task],
        task_name='inverse water mask')

    inverse_dist_edt_task = graph.add_task(
        func=pygeoprocessing.distance_transform_edt,
        args=(
            (file_registry['not_water_mask'], 1),
            file_registry['inverse_distance'],
            (default_pixel_size[0], default_pixel_size[0])),
        target_path_list=[file_registry['inverse_distance']],
        dependent_task_list=[not_water_mask_task],
        task_name='inverse distance edt')
    ### Population proximity to water


    graph.close()
    graph.join()

    LOGGER.info("Model completed")


def _plot_results(input_raster_path, output_raster_path, plot_path, suit_name, func_name):
    input_array = pygeoprocessing.raster_to_numpy_array(input_raster_path).flatten()
    output_array = pygeoprocessing.raster_to_numpy_array(output_raster_path).flatten()
    _plotter(
        input_array, output_array, save_path=plot_path,
        label_x=suit_name, label_y=func_name,
        title=f'{suit_name}--{func_name}', xticks=None, yticks=None)


def _degree_op(slope): return numpy.degrees(numpy.arctan(slope / 100.0))


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
    #=IFS(TEMP<16, 0, TEMP<=35, -0.003*(268/(TEMP-14.2)-335)+0.0336538, TEMP>35, 0)
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

def _water_proximity(water_distance_path, target_raster_path):
    """ """
    #ProxRisk <- function(prox){ifelse(prox<1000, 1,ifelse(prox<=15000, -0.0000714 * prox + 1.0714,0))}
    water_distance_info = pygeoprocessing.get_raster_info(water_distance_path)
    water_distance_nodata = water_distance_info['nodata'][0]
    def op(water_distance_array):
        output = numpy.full(
            water_distance_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(water_distance_array, water_distance_nodata))

        # 
        lt_km_mask = valid_pixels & (water_distance_array < 1000)
        lt_gt_mask = valid_pixels & (water_distance_array >= 1000) & (water_distance_array <= 15000)
        gt_mask = valid_pixels & (water_distance_array > 15000)
        output[lt_km_mask] = 1
        output[lt_gt_mask] = -0.0000714 * water_distance_array[lt_gt_mask] + 1.0714
        output[gt_mask] = 0

        return output

    pygeoprocessing.raster_calculator(
        [(water_distance_path, 1)], op, target_raster_path, gdal.GDT_Float32,
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
        degree = numpy.full(
            slope_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        output = numpy.full(
            slope_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = ~numpy.isclose(slope_array, slope_nodata)

        # percent slope to degrees
        # https://support.esri.com/en-us/knowledge-base/how-to-convert-the-slope-unit-from-percent-to-degree-in-000022558
        degree[valid_pixels] = numpy.degrees(numpy.arctan(slope_array[valid_pixels] / 100.0))
        mask_lt = valid_pixels & (degree <= 0.00014)
        output[mask_lt] = -5714.3 * degree[mask_lt] + 1
        mask_gt = valid_pixels & (degree > 0.00014)
        output[mask_gt] = -0.0029 * degree[mask_gt] + 0.2

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

    slope_inc = (yb - ya) / (xb - xa)
    slope_dec = (yc - yz) / (xc - xz)
    intercept_inc = ya - (slope_inc * xa)
    intercept_dec = yc - (slope_dec * xc)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        # First plateau
        mask_one = valid_pixels & (raster_array <= xa)
        output[mask_one] = ya
        # First slope
        mask_linear_inc = valid_pixels & (raster_array > xa) & (raster_array < xb)
        output[mask_linear_inc] = (slope_inc * raster_array[mask_linear_inc]) + intercept_inc
        # Second plateau
        mask_three = valid_pixels & (raster_array >= xb) & (raster_array <= xc)
        output[mask_three] = yb
        # Second slope
        mask_linear_dec = valid_pixels & (raster_array > xc) & (raster_array < xz)
        output[mask_linear_dec] = (slope_dec * raster_array[mask_linear_dec]) + intercept_dec
        # Third plateau
        mask_four = valid_pixels & (raster_array >= xz)
        output[mask_four] = yz

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _gaussian_op(raster_path, target_raster_path, mean=0, std=1, lb=0, ub=40):
    """ """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    rv = scipy.stats.norm(loc=mean, scale=std)

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


def _sshape_op(
        raster_path, target_raster_path, yin=1, yfin=0, xmed=15, inv_slope=3):
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


def _exponential_decay_op(
        raster_path, target_raster_path, yin=1, xmed=1, decay_factor=0.982,
        max_dist=1000):
    """ """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        output[valid_pixels & (raster_array < xmed)] = yin
        exp_mask = valid_pixels & (raster_array >= xmed)
        output[exp_mask] = yin * (decay_factor**raster_array[exp_mask])

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _linear_op(raster_path, target_raster_path, xa, ya, xz, yz):
    """ """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    slope = (yz - ya) / (xz - xa)
    intercept = ya - (slope * xa)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~numpy.isclose(raster_array, raster_nodata))

        # First plateau
        mask_one = valid_pixels & (raster_array <= xa)
        output[mask_one] = ya
        # Line
        mask_linear_inc = valid_pixels & (raster_array > xa) & (raster_array < xz)
        output[mask_linear_inc] = (slope * raster_array[mask_linear_inc]) + intercept
        # Second plateau
        mask_two = valid_pixels & (raster_array >= xz)
        output[mask_two] = yz

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _generic_func_values(func_op, xrange, working_dir, kwargs):
    """Call a raster based function on a generic range of values.

    The point of this function is to be able to plot values in ``xrange``
    against ``func_op(x)``. Since ``func_op`` expects a raster to operate on
    we create a one with the values of ``xrange`` to pass in.

    Args:
        func_op (string): 
        xrange (string): 
        working_dir (string): 
        kwargs (dict): 

    Returns:
        values_x (numpy array): 
        numpy_values_y (numpy array): 
    """
    # Generic spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # NAD83 / UTM zone 11N
    srs_wkt = srs.ExportToWkt()
    origin = (463250, 4929700)
    pixel_size = (30, -30)

    values_x = numpy.linspace(xrange[0], xrange[1], 100).reshape(10,10)
    nodata_x = -999

    tmp_working_dir = tempfile.mkdtemp(dir=working_dir)

    func_input_path = os.path.join(tmp_working_dir, f'temp-{func_op.__name__}.tif')
    pygeoprocessing.numpy_array_to_raster(
        values_x, nodata_x, pixel_size, origin, srs_wkt, func_input_path)
    func_result_path = os.path.join(tmp_working_dir, f'temp-{func_op.__name__}-result.tif')

    LOGGER.debug(f"func kwargs: {kwargs}")
    if kwargs:
        func_op(func_input_path, func_result_path, **kwargs)
    else:
        func_op(func_input_path, func_result_path)

    numpy_values_y = pygeoprocessing.raster_to_numpy_array(func_result_path)

    shutil.rmtree(tmp_working_dir, ignore_errors=True)

    return (values_x, numpy_values_y)


def _plotter(values_x, values_y, save_path=None, label_x=None, label_y=None,
             title=None, xticks=None, yticks=None):
    """ """
    flattened_x_array = values_x.flatten()
    flattened_y_array = values_y.flatten()
    xmin=numpy.min(flattened_x_array)
    xmax=numpy.max(flattened_x_array)
    ymin=numpy.min(flattened_y_array)
    ymax=numpy.max(flattened_y_array)

    # plot
    #plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()

    ax.plot(flattened_x_array, flattened_y_array, linewidth=2.0)

    #ax.set(xlim=(xmin, xmax), xticks=numpy.arange(xmin + 10, xmax, 10),
    #        ylim=(ymin-.1, ymax+.1), yticks=numpy.arange(ymin, ymax + .25, .25))
    ax.set(xlim=(xmin, xmax), ylim=(ymin-.1, ymax+.1))

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    #plt.show()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


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
        valid_mask = ~pygeoprocessing.array_equals_nodata(population, population_nodata)
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

def _pop_count_to_density(pop_count_path, target_path):
    """Population count to population density in hectares."""
    population_raster_info = pygeoprocessing.get_raster_info(pop_count_path)
    pop_pixel_area = abs(numpy.multiply(*population_raster_info['pixel_size']))

    kwargs={
        'op': lambda x: (x/pop_pixel_area)*10000,  # convert count per pixel to meters sq to hectares
        'rasters': [pop_count_path],
        'target_path': target_path,
        'target_nodata': -1,
    }

    pygeoprocessing.raster_map(**kwargs)


@validation.invest_validator
def validate(args, limit_to=None):
    # Could roll custom validation for function defn tables
    #biophysical_df = validation.get_validated_dataframe(table_path, spec)
    return validation.validate(
        args, MODEL_SPEC['args'], MODEL_SPEC['args_with_spatial_overlap'])
