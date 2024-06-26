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
import subprocess
import sys

from natcap.invest import spec_utils
from natcap.invest import gettext
from natcap.invest import utils
from natcap.invest.spec_utils import u
from natcap.invest import validation
import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import pygeoprocessing.routing
from pygeoprocessing import geoprocessing
import taskgraph
from osgeo import gdal
from osgeo import osr

import numpy
import matplotlib.pyplot as plt
from scipy.stats import gmean

LOGGER = logging.getLogger(__name__)

logging.getLogger('taskgraph').setLevel('DEBUG')
# Was seeing a lot of font related logging
# https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger('matplotlib.font_manager').disabled = True

UINT32_NODATA = int(numpy.iinfo(numpy.uint32).max)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255

SCHISTO = "Schisto alpha"
SNAIL_PARASITE = {
        "sh": "S-haematobium",
        "sm": "S-mansoni",
        "bt": "Bulinus truncatus",
        "bg": "Biomphalaria"}

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
    'scurve': {
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
    }
}


FUNCS = ['linear', 'trapezoid', 'gaussian', 'scurve', 'exponential']

FUNC_PARAMS = {
    'population': {
        f'population_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_population and population_func_type == '{fn}'",
            "allowed": f"calc_population and population_func_type == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    },
    'water_distance': {
        f'water_distance_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_water_distance and water_distance_func_type == '{fn}'",
            "allowed": f"calc_water_distance and water_distance_func_type == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    },
    'water_velocity': {
        f'water_velocity_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_water_velocity and water_velocity_func_type == '{fn}'",
            "allowed": f"calc_water_velocity and water_velocity_func_type == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    },
    'temperature': {
        f'temperature_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_temperature and temperature_func_type == '{fn}'",
            "allowed": f"calc_temperature and temperature_func_type == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    },
    'ndvi': {
        f'ndvi_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_ndvi and ndvi_func_type == '{fn}'",
            "allowed": f"calc_ndvi and ndvi_func_type == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    }
}

MODEL_SPEC = {
    'model_id': 'schistosomiasis',
    'model_name': gettext(SCHISTO),
    'pyname': 'natcap.invest.schistosomiasis',
    'userguide': "schistosomiasis.html",
    'aliases': (),
    "ui_spec": {
        "order": [
            ['workspace_dir', 'results_suffix'],
            ['aoi_vector_path'],
            ["calc_population", "population_func_type",
             "population_count_path",
             {"Population parameters": list(FUNC_PARAMS['population'].keys())}],
            ["calc_water_distance", "water_distance_func_type",
             "water_presence_path",
             {"Water distance parameters": list(FUNC_PARAMS['water_distance'].keys())}],
            ["calc_water_velocity", "water_velocity_func_type",
             "dem_path",
             {"Water velocity parameters": list(FUNC_PARAMS['water_velocity'].keys())}],
            ["calc_temperature", "temperature_func_type",
             "water_temp_dry_raster_path", "water_temp_wet_raster_path",
             {"Temperature parameters": list(FUNC_PARAMS['temperature'].keys())}],
            ["calc_ndvi", "ndvi_func_type",
             "ndvi_dry_raster_path", "ndvi_wet_raster_path",
             {"NDVI parameters": list(FUNC_PARAMS['ndvi'].keys())}],
            ["urbanization_func_type", "urbanization_table_path"],
        ],
        "hidden": ["n_workers"],
        "forum_tag": 'schisto',
        "sampledata": {
            "filename": "schisto-demo.zip"
        }
    },
    'args_with_spatial_overlap': {
        'spatial_keys': [
            'aoi_vector_path', 'population_count_path', 'dem_path',
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
            "name": gettext("calculate population"),
            "required": False
        },
        "aoi_vector_path": {
            **spec_utils.AOI,
            "projected": True,
            "projection_units": u.meter,
            "about": gettext(
                "Map of the area(s) of interest over which to run the model "
                "and aggregate valuation results. Required if Run Valuation "
                "is selected and the Grid Connection Points table is provided."
            )
        },
        "population_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_population",
            "allowed": "calc_population"
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
            "required": "calc_population",
            "allowed": "calc_population"
        },
        **FUNC_PARAMS['population'],
        "urbanization_func_type": {
            **SPEC_FUNC_TYPES,
            "required": True,
            #"allowed": "calc_urbanization"
        },
        "urbanization_table_path": {
            "type": "csv",
            #"index_col": "suit_factor",
            #"columns": **SPEC_FUNC_COLS['urbanization_func_type'],
            "required": "urbanization_func_type != 'default'",
            "allowed": "urbanization_func_type != 'default'",
            "about": gettext(
                "A table mapping each suitibility factor to a function."),
            "name": gettext("urbanization table")
        },
        "calc_water_distance": {
            "type": "boolean",
            "about": gettext("Calculate water distance."),
            "name": gettext("calculate water distance"),
            "required": False
        },
        "water_distance_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_water_distance",
            "allowed": "calc_water_distance",
        },
        'water_presence_path': {
            'type': 'raster',
            'name': 'water presence',
            'bands': {1: {'type': 'integer'}},
            'about': (
                "A raster indicating presence of water."
            ),
            "required": "calc_water_distance",
            "allowed": "calc_water_distance"
        },
        **FUNC_PARAMS['water_distance'],
        "calc_water_velocity": {
            "type": "boolean",
            "about": gettext("Calculate water velocity."),
            "name": gettext("calculate water velocity"),
            "required": False
        },
        "water_velocity_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_water_velocity",
            "allowed": "calc_water_velocity"
        },
        **FUNC_PARAMS['water_velocity'],
        'dem_path': {
            **spec_utils.DEM,
            "projected": True,
            "required": "calc_water_velocity",
            "allowed": "calc_water_velocity"
        },
        "calc_temperature": {
            "type": "boolean",
            "about": gettext("Calculate temperature."),
            "name": gettext("calculate temperature"),
            "required": False
        },
        "temperature_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_temperature",
            "allowed": "calc_temperature"
        },
        **FUNC_PARAMS['temperature'],
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
            "allowed": "calc_temperature"
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
            "allowed": "calc_temperature"
        },
        "calc_ndvi": {
            "type": "boolean",
            "about": gettext("Calculate NDVI."),
            "name": gettext("calculate NDVI"),
            "required": False
        },
        "ndvi_func_type": {
            **SPEC_FUNC_TYPES,
            "required": "calc_ndvi",
            "allowed": "calc_ndvi"
        },
        **FUNC_PARAMS['ndvi'],
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
            "allowed": "calc_ndvi"
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
            "allowed": "calc_ndvi"
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
    'water_temp_suit_dry_sh': 'water_temp_suit_dry_sh.tif',
    'water_temp_suit_wet_sh': 'water_temp_suit_wet_sh.tif',
    'water_temp_suit_dry_bg': 'water_temp_suit_dry_bg.tif',
    'water_temp_suit_wet_bg': 'water_temp_suit_wet_bg.tif',
    'water_temp_suit_dry_bt': 'water_temp_suit_dry_bt.tif',
    'water_temp_suit_wet_bt': 'water_temp_suit_wet_bt.tif',
    'ndvi_suit_dry': 'ndvi_suit_dry.tif',
    'ndvi_suit_wet': 'ndvi_suit_wet.tif',
    'water_velocity_suit': 'water_velocity_suit.tif',
    'water_proximity_suit': 'water_proximity_suit.tif',
    'rural_pop_suit': 'rural_pop_suit.tif',
    'urbanization_suit': 'urbanization_suit.tif',
    'rural_urbanization_suit': 'rural_urbanization_suit.tif',
    'water_stability_suit': 'water_stability_suit.tif',
    'habitat_suit_geometric_mean': 'habitat_suit_geometric_mean.tif',
}

_INTERMEDIATE_BASE_FILES = {
    'aligned_pop_count': 'aligned_population_count.tif',
    'aligned_pop_density': 'aligned_pop_density.tif',
    'masked_population': 'masked_population.tif',
    'population_density': 'population_density.tif',
    'population_hectares': 'population_hectare.tif',
    'aligned_water_temp_dry': 'aligned_water_temp_dry.tif',
    'aligned_water_temp_wet': 'aligned_water_temp_wet.tif',
    'aligned_ndvi_dry': 'aligned_ndvi_dry.tif',
    'aligned_ndvi_wet': 'aligned_ndvi_wet.tif',
    'aligned_dem': 'aligned_dem.tif',
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
            square km.
        
        args['urbanization_func_type'] = 'default'
        args['urbanization_table_path'] = ''

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
        'temperature': _water_temp_suit,
        'ndvi': _ndvi,
        'population': _rural_population_density,
        'urbanization': _urbanization,
        'water_velocity': _water_velocity,
        'water_distance': _water_proximity,
        }
    PLOT_PARAMS = {
        'temperature': (0, 50),
        'ndvi': (-1, 1),
        'population': (0, 10),
        'urbanization': (0, 10),
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
        'water_velocity', 'urbanization']
    suit_func_to_use = {}
    for suit_key in user_func_paths:
        func_type = args[f'{suit_key}_func_type']
        if func_type != 'default':
            func_params = {}
            for key in SPEC_FUNC_COLS[func_type].keys():
                func_params[key] = float(args[f'{suit_key}_{func_type}_param_{key}'])
            user_func = FUNC_TYPES[func_type]
        else:
            func_params = None
            user_func = DEFAULT_FUNC_TYPES[suit_key]

        suit_func_to_use[suit_key] = {
            'func_name':user_func,
            'func_params':func_params
            }
        # NOTE: adding this if/else to handle snail+parasite combos for temperature suitability
        if func_params == None and suit_key == 'temperature':
            for op_key in SNAIL_PARASITE.keys():
                func_params = {'op_key': op_key}
                results = _generic_func_values(
                    user_func, PLOT_PARAMS[suit_key], intermediate_dir, func_params)
                plot_path = os.path.join(func_plot_dir, f"{suit_key}-{func_type}-{op_key}.png")
                _plotter(
                    results[0], results[1], save_path=plot_path,
                    label_x=suit_key, label_y=func_type,
                    title=f'{suit_key}--{func_type}-{op_key}', xticks=None, yticks=None)
        else:
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

    vector_mask_options = {
        'mask_vector_path': args['watersheds_path'],
    }
    align_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        kwargs={
            'base_raster_path_list': raster_input_list,
            'target_raster_path_list': aligned_input_list,
            'resample_method_list': ['near']*len(raster_input_list),
            'target_pixel_size': squared_default_pixel_size,
            'bounding_box_mode': 'intersection',
            'vector_mask_options': vector_mask_options,
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
    # Returns population count aligned to other inputs
    population_align_task = graph.add_task(
        _resample_population_raster,
        kwargs={
            'source_population_raster_path': args['population_count_path'],
            'target_pop_count_raster_path': file_registry['aligned_pop_count'],
            'target_pop_density_raster_path': file_registry['aligned_pop_density'],
            'lulc_pixel_size': squared_default_pixel_size,
            'lulc_bb': default_bb,
            'lulc_projection_wkt': default_wkt,
            'working_dir': intermediate_dir,
        },
        target_path_list=[file_registry['aligned_pop_count'], file_registry['aligned_pop_density']],
        task_name='Align and resize population'
    )

    ### Production functions
    suitability_tasks = []
    habitat_suit_risk_paths = []
    outputs_to_tile = []

    default_color_path = os.path.join(
        "C:", os.sep, "Users", "ddenu", "Workspace", "Repositories",
        "schistosomiasis", "data", "water-temp-wet-colors.txt")
    
    ### Habitat stability
    habitat_stability_task = graph.add_task(
        _habitat_stability,
        kwargs={
            'surface_water_presence': args['surface_water_presence'],
            'months': 1.75,
            'target_raster_path': file_registry['habitat_stability'],
        },
        target_path_list=[file_registry['habitat_stability']],
        task_name='habitat stability')
    suitability_tasks.append(habitat_suitability_task)
    habitat_suit_risk_paths.append(file_registry['habitat_stability'])
    outputs_to_tile.append((file_registry['habitat_suitability'], default_color_path))


    ### Water velocity
    # calculate slope
    slope_task = graph.add_task(
        func=pygeoprocessing.calculate_slope,
        args=(
            (file_registry['aligned_dem'], 1),
            file_registry['slope']),
        target_path_list=[file_registry['slope']],
        dependent_task_list=[align_task],
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
    suitability_tasks.append(water_vel_task)
    habitat_suit_risk_paths.append(file_registry['water_velocity_suit'])
    outputs_to_tile.append((file_registry['water_velocity_suit'], default_color_path))

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
        suit_func_to_use['water_distance']['func_name'],
        args=(file_registry['distance'], file_registry['water_proximity_suit']),
        kwargs=suit_func_to_use['water_distance']['func_params'],
        dependent_task_list=[dist_edt_task],
        target_path_list=[file_registry[f'water_proximity_suit']],
        task_name=f'Water Proximity Suit')
    suitability_tasks.append(water_proximity_task)
    outputs_to_tile.append((file_registry[f'water_proximity_suit'], default_color_path))

    ### Rural population density and urbanization
    # Population count to density in hectares
    pop_hectare_task = graph.add_task(
        func=_pop_count_to_density,
        kwargs={
            'pop_count_path': file_registry['aligned_pop_count'],
            'target_path': file_registry['population_hectares'],
        },
        target_path_list=[file_registry['population_hectares']],
        dependent_task_list=[population_align_task],
        task_name=f'Population count to density in hectares.')

    rural_pop_task = graph.add_task(
        suit_func_to_use['population']['func_name'],
        args=(
            file_registry['population_hectares'],
            file_registry['rural_pop_suit']),
        kwargs=suit_func_to_use['population']['func_params'],
        dependent_task_list=[pop_hectare_task],
        target_path_list=[file_registry['rural_pop_suit']],
        task_name=f'Rural Population Suit')
    suitability_tasks.append(rural_pop_task)
    outputs_to_tile.append((file_registry[f'rural_pop_suit'], default_color_path))
    
    urbanization_task = graph.add_task(
        suit_func_to_use['urbanization']['func_name'],
        args=(
            file_registry['population_hectares'],
            file_registry['urbanization_suit']),
        kwargs=suit_func_to_use['urbanization']['func_params'],
        dependent_task_list=[pop_hectare_task],
        target_path_list=[file_registry['urbanization_suit']],
        task_name=f'Urbanization Suit')
    suitability_tasks.append(urbanization_task)
    outputs_to_tile.append((file_registry[f'urbanization_suit'], default_color_path))
    
    rural_urbanization_task = graph.add_task(
        _rural_urbanization_combined,
        args=(
            file_registry['population_hectares'],
            file_registry['rural_pop_suit'],
            file_registry['urbanization_suit'],
            file_registry['rural_urbanization_suit'],
            ),
        dependent_task_list=[rural_pop_task, urbanization_task],
        target_path_list=[file_registry['rural_urbanization_suit']],
        task_name=f'Rural Urbanization Suit')
    #suitability_tasks.append(rural_urbanization_task)
    outputs_to_tile.append((file_registry[f'rural_urbanization_suit'], default_color_path))

    for season in ["dry", "wet"]:
        ### Water temperature
        for op_key in SNAIL_PARASITE.keys():
            # NOTE: adding this if/else to handle default funcs for each
            # snail+parasite combo vs manual func, where manual func will
            # currently apply to each snail+parasite combo 
            # If func_params are None then use op_key with default function
            if suit_func_to_use['temperature']['func_params'] == None:
                water_temp_task = graph.add_task(
                    suit_func_to_use['temperature']['func_name'],
                    args=(
                        file_registry[f'aligned_water_temp_{season}'],
                        file_registry[f'water_temp_suit_{season}_{op_key}'],
                        op_key
                    ),
                    kwargs=suit_func_to_use['temperature']['func_params'],
                    dependent_task_list=[align_task],
                    target_path_list=[file_registry[f'water_temp_suit_{season}_{op_key}']],
                    task_name=f'Water Temp Suit for {season} {op_key}')
            else:
                water_temp_task = graph.add_task(
                    suit_func_to_use['temperature']['func_name'],
                    args=(
                        file_registry[f'aligned_water_temp_{season}'],
                        file_registry[f'water_temp_suit_{season}_{op_key}'],
                    ),
                    kwargs=suit_func_to_use['temperature']['func_params'],
                    dependent_task_list=[align_task],
                    target_path_list=[file_registry[f'water_temp_suit_{season}_{op_key}']],
                    task_name=f'Water Temp Suit for {season} {op_key}')
            suitability_tasks.append(water_temp_task)
            habitat_suit_risk_paths.append(file_registry[f'water_temp_suit_{season}_{op_key}'])
            outputs_to_tile.append((file_registry[f'water_temp_suit_{season}_{op_key}'], default_color_path))

        ### Vegetation coverage (NDVI)
        ndvi_task = graph.add_task(
            suit_func_to_use['ndvi']['func_name'],
            args=(
                file_registry[f'aligned_ndvi_{season}'],
                file_registry[f'ndvi_suit_{season}'],
            ),
            kwargs=suit_func_to_use['ndvi']['func_params'],
            dependent_task_list=[align_task],
            target_path_list=[file_registry[f'ndvi_suit_{season}']],
            task_name=f'NDVI Suit for {season} SM')
        suitability_tasks.append(ndvi_task)
        habitat_suit_risk_paths.append(file_registry[f'ndvi_suit_{season}'])
        outputs_to_tile.append((file_registry[f'ndvi_suit_{season}'], default_color_path))

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


    ### Geometric mean of water risks
    geometric_mean_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs={
            'op': _geometric_mean_op,
            'rasters': habitat_suit_risk_paths,
            'target_path': file_registry['habitat_suit_geometric_mean'],
            'target_nodata': BYTE_NODATA,
            },
        target_path_list=[file_registry['habitat_suit_geometric_mean']],
        dependent_task_list=suitability_tasks,
        task_name='geometric mean')


    ### Convolve habitat suit geometric mean over land
    # TODO: add this to be an input to the model
    # TODO: mask out water bodies to nodata and not include in risk
    #decay_dist_m = 2000
    decay_dist_m = 15 * 1000
    kernel_path = os.path.join(
        intermediate_dir, f'kernel{suffix}.tif')
    max_dist_pixels = abs(
        decay_dist_m / squared_default_pixel_size[0])
    kernel_func = pygeoprocessing.kernels.create_distance_decay_kernel

    def decay_func(dist_array):
        return _kernel_gaussian(
            dist_array, max_distance=max_dist_pixels)

    kernel_kwargs = dict(
        target_kernel_path=kernel_path,
        distance_decay_function=decay_func,
        max_distance=max_dist_pixels,
        normalize=False)

    kernel_task = graph.add_task(
        kernel_func,
        kwargs=kernel_kwargs,
        task_name=(
            f'Create guassian kernel - {decay_dist_m}m'),
        target_path_list=[kernel_path])

    convolved_hab_risk_path = os.path.join(
        intermediate_dir,
        f'distance_weighted_hab_risk_within_{decay_dist_m}{suffix}.tif')
    convolved_hab_risk_task = graph.add_task(
        _convolve_and_set_lower_bound,
        kwargs={
            'signal_path_band': (file_registry['habitat_suit_geometric_mean'], 1),
            'kernel_path_band': (kernel_path, 1),
            'target_path': convolved_hab_risk_path,
            'working_dir': intermediate_dir,
        },
        task_name=f'Convolve hab risk - {decay_dist_m}m',
        target_path_list=[convolved_hab_risk_path],
        dependent_task_list=[kernel_task, geometric_mean_task])
    
    masked_convolved_path = os.path.join(
        intermediate_dir,
        f'masked_hab_risk_within_{decay_dist_m}{suffix}.tif')
    mask_convolve_task = graph.add_task(
        _nodata_mask_op,
        kwargs={
            'input_path': convolved_hab_risk_path,
            'mask_path': file_registry['aligned_water_presence'],
            'target_path': masked_convolved_path,
        },
        task_name=f'Mask convolve hab risk - {decay_dist_m}m',
        target_path_list=[masked_convolved_path],
        dependent_task_list=[convolved_hab_risk_task])

    ### Weight convolved risk by urbanization
    risk_to_pop_path = os.path.join(
        output_dir, f'risk_to_pop{suffix}.tif')
    risk_to_pop_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs={
            'op': _multipy_op,
            'rasters': [file_registry['rural_urbanization_suit'], convolved_hab_risk_path],
            'target_path': risk_to_pop_path,
            'target_nodata': FLOAT32_NODATA,
            },
        target_path_list=[risk_to_pop_path],
        dependent_task_list=[convolved_hab_risk_task, urbanization_task],
        task_name='risk to population')
    outputs_to_tile.append((risk_to_pop_path, default_color_path))

    # water habitat suitability gets at the risk of maximum potential schisto exposure
    # schisto exposure x urbanization gets at the risk of likelihood of exposure given socioeconomic factors
    # final risk, is population. Where are there the most people at the highest risk.

    ### Multiply risk_to_pop by people count?
    # Want to get to how many people are at risk
    # Multiply by count or by density
    # TODO: raw and scaled outputs for convolved risk, urbanization x raw convolved, and risk to people
    risk_to_pop_count_path = os.path.join(
        output_dir, f'risk_to_pop_count{suffix}.tif')
    risk_to_pop_count_task = graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs={
            'op': _multiply_op,
            'rasters': [risk_to_pop_path, population_count_path],
            'target_path': risk_to_pop_count_path,
            'target_nodata': FLOAT32_NODATA,
        target_path_list=[risk_to_pop_count_path],
        dependent_task_list=[risk_to_pop_task],
        task_name='risk to pop_count')
    outputs_to_tile.append((risk_to_pop_count_path, default_color_path))

    graph.close()
    graph.join()

    ### Tile outputs
    #tile_task = graph.add_task(
    #    _tile_raster,
    #    kwargs={
    #        'raster_path': file_registry['water_temp_suit_wet_sm'],
    #        'color_relief_path': color_relief_path,
    #    },
    #    task_name=f'Tile temperature',
    #    dependent_task_list=suitability_tasks)
    for raster_path, color_path in outputs_to_tile:
        #_tile_raster(raster_path, color_path)
        continue 


    LOGGER.info("Model completed")


# raster_map op for geometric mean of habitat suitablity risk layers.
# `arrays` is expected to be a list of numpy arrays
def _geometric_mean_op(*arrays):
    """
     raster_map op for geometric mean of habitat suitablity risk layers.
     `arrays` is expected to be a list of numpy arrays

     In practice this function has been slow and I wonder if it'd be
     more efficient to write our own version of scipy.stats.gmean
     """
    # Treat 0 values as numpy.nan so can omit them from geometric mean
    #for array in arrays:
    #    array[array==0] = numpy.nan

    #result = gmean(arrays, axis=0, nan_policy='omit')
    #nan_mask = numpy.isnan(result)
    #result[nan_mask] = BYTE_NODATA
    #return result
    return gmean(arrays, axis=0)

def _rural_urbanization_combined(pop_density_path, rural_path, urbanization_path):
    """Combine the rural and urbanization functions."""
    rural_info = pygeoprocessing.get_raster_info(rural_path)
    rural_nodata = rural_info['nodata'][0]
    urbanization_info = pygeoprocessing.get_raster_info(urbanization_path)
    urbanization_nodata = urbanization_info['nodata'][0]

    def _rural_urbnization_op(pop_density_array, rural_array, urbanization_array):
        output = numpy.full(
            rural_array.shape, BYTE_NODATA, dtype=numpy.float32)
        nodata_mask = (
                pygeoprocessing.array_equals_nodata(rural_array, rural_nodata) |
                pygeoprocessing.array_equals_nodata(urbanization_array, urbanization_nodata) )

        use_rural_mask = pop_density_array <= 1
        output[use_rural_mask] = rural_array[use_rural_mask]
        output[~use_rural_mask] = urbanization_array[~use_rural_mask]
        output[nodata_mask] = BYTE_NODATA

        return output
    
    pygeoprocessing.raster_calculator(
        [(pop_density_path, 1), (rural_path, 1), (urbanization_path, 1)],
        _rural_urbanization_op, target_raster_path, gdal.GDT_Float32, BYTE_NODATA)

def _multiply_op(array_one, array_two): return numpy.multiply(array_one, array_two)

def _tile_raster(raster_path, color_relief_path):
    """ """
    # Set up directory and paths for outputs
    base_dir = os.path.dirname(raster_path)
    base_name = os.path.splitext(os.path.basename(raster_path))[0]
    rgb_raster_path = os.path.join(base_dir, f'{base_name}_rgb.tif')
    tile_dir = os.path.join(base_dir, f'{base_name}_tiles')

    if not os.path.isdir(tile_dir):
        os.mkdir(tile_dir)
    gdaldem_cmd = f'gdaldem color-relief -co COMPRESS=LZW {raster_path} {color_relief_path} {rgb_raster_path}'
    subprocess.call(gdaldem_cmd, shell=True)
    tile_cmd = f'gdal2tiles --xyz -r near -e --zoom=1-13 --process=4 {rgb_raster_path} {tile_dir}'
    print(tile_cmd)
    subprocess.call(tile_cmd, shell=True)

def _weighted_mean_risk_index(suitability_risk_list, target_raster_path):
    """ """
    pass

def _plot_results(input_raster_path, output_raster_path, plot_path, suit_name, func_name):
    input_array = pygeoprocessing.raster_to_numpy_array(input_raster_path).flatten()
    output_array = pygeoprocessing.raster_to_numpy_array(output_raster_path).flatten()
    _plotter(
        input_array, output_array, save_path=plot_path,
        label_x=suit_name, label_y=func_name,
        title=f'{suit_name}--{func_name}', xticks=None, yticks=None)


def _degree_op(slope): return numpy.degrees(numpy.arctan(slope / 100.0))


def _habitat_stability(water_presence_path, months, target_raster_path):
    """

    Arguments:
        water_presence_path (string): 
        months (float): number of consecutive months for water to be considered habitat.
        target_raster_path (string): 


    """
    water_presence_info = pygeoprocessing.get_raster_info(water_presence_path)
    water_presence_nodata = water_presence_info['nodata'][0]

    def op(array):
        output = numpy.full(
            array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (
            ~pygeoprocessing.array_equals_nodata(array, water_presence_nodata))

        # values with 1 month or less surface water presence set to 0
        lte_one_mask = (array <= 1) & valid_pixels
        output[lte_one_mask] = 0

        between_mask = (array > 1) & (array <= months) & valid_pixels
        output[between_mask] = 1.33 * array[between_mask] - 1.33

        gt_mask = (array > months) & valid_pixels
        output[gt_mask] = 1

        return output

    pygeoprocessing.raster_calculator(
        [(water_presence_path, 1)],
        op, target_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

### Water temperature functions ###

def _get_temp_op(key):
    TEMP_OP_MAP = {
        "sh": _water_temp_op_sh, 
        "sm": _water_temp_op_sm, 
        "bg": _water_temp_op_bg, 
        "bt": _water_temp_op_bt, 
    }
    return TEMP_OP_MAP[key]

def _water_temp_op_sm(temp_array, temp_nodata):
    """Water temperature suitability for S. mansoni."""
    #SmWaterTemp <- function(Temp){ifelse(Temp<16, 0,ifelse(Temp<=35, -0.003 * (268/(Temp - 14.2) - 335) + 0.0336538, 0))}
    #=IFS(TEMP<16, 0, TEMP<=35, -0.003*(268/(TEMP-14.2)-335)+0.0336538, TEMP>35, 0)
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    #nodata_pixels = (numpy.isclose(temp_array, temp_nodata))
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 or higher than 35 set to 0
    valid_range_mask = (temp_array>=16) & (temp_array<=35)
    output[valid_range_mask] = (
        -0.003 * (268 / (temp_array[valid_range_mask] - 14.2) - 335) + 0.0336538)
    output[~valid_range_mask] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output

def _water_temp_op_sh(temp_array, temp_nodata):
    """Water temperature suitability for S. haematobium."""
    #ShWaterTemp <- function(Temp){ifelse(Temp<17, 0,ifelse(Temp<=33, -0.006 * (295/(Temp - 15.3) - 174) + 0.056, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    #nodata_pixels = (numpy.isclose(temp_array, temp_nodata))
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=17) & (temp_array<=33)
    output[valid_range_mask] = (
        -0.006 * (295 / (temp_array[valid_range_mask] - 15.3) - 174) + 0.056)
    output[~valid_range_mask] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output

def _water_temp_op_bt(temp_array, temp_nodata):
    """Water temperature suitability for Bulinus truncatus."""
    #BtruncatusWaterTempNEW <- function(Temp){ifelse(Temp<17, 0,ifelse(Temp<=33, -48.173 + 8.534e+00 * Temp + -5.568e-01 * Temp^2 + 1.599e-02 * Temp^3 + -1.697e-04 * Temp^4, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    #nodata_pixels = (numpy.isclose(temp_array, temp_nodata))
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=17) & (temp_array<=33)
    output[valid_range_mask] = (
        -48.173 + (8.534 * temp_array[valid_range_mask]) + 
        (-5.568e-01 * numpy.power(temp_array[valid_range_mask], 2)) +
        (1.599e-02 * numpy.power(temp_array[valid_range_mask], 3)) +
        (-1.697e-04 * numpy.power(temp_array[valid_range_mask], 4)))
    output[~valid_range_mask] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output

def _water_temp_op_bg(temp_array, temp_nodata):
    """Water temperature suitability for Biomphalaria."""
    #BglabrataWaterTempNEW <- function(Temp){ifelse(Temp<16, 0,ifelse(Temp<=35, -29.9111 + 5.015e+00 * Temp + -3.107e-01 * Temp^2 +8.560e-03 * Temp^3 + -8.769e-05 * Temp^4, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    #nodata_pixels = (numpy.isclose(temp_array, temp_nodata))
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=16) & (temp_array<=35)
    output[valid_range_mask] = (
        -29.9111 + (5.015 * temp_array[valid_range_mask]) + 
        (-3.107e-01 * numpy.power(temp_array[valid_range_mask], 2)) +
        (8.560e-03 * numpy.power(temp_array[valid_range_mask], 3)) +
        (-8.769e-05 * numpy.power(temp_array[valid_range_mask], 4)))
    output[~valid_range_mask] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output


def _water_temp_suit(water_temp_path, target_raster_path, op_key):
    """

        Args:
            water_temp_path (string):
            op_key (string):
            target_raster_path (string):

        Returns:
    """
    print(f"op_key in func: {op_key}")
    water_temp_info = pygeoprocessing.get_raster_info(water_temp_path)
    water_temp_nodata = water_temp_info['nodata'][0]
    op = _get_temp_op(op_key)

    pygeoprocessing.raster_calculator(
        [(water_temp_path, 1), (water_temp_nodata, "raw")], op,
        target_raster_path, gdal.GDT_Float32, BYTE_NODATA)

### End water temperature functions ###

def _ndvi(ndvi_path, target_raster_path):
    """ """
    #VegCoverage <- function(V){ifelse(V<0,0,ifelse(V<=0.3,3.33*V,1))}
    ndvi_info = pygeoprocessing.get_raster_info(ndvi_path)
    ndvi_nodata = ndvi_info['nodata'][0]
    def op(ndvi_array):
        output = numpy.full(
            ndvi_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(ndvi_array, ndvi_nodata))

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
        valid_pixels = (~pygeoprocessing.array_equals_nodata(water_distance_array, water_distance_nodata))

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

def _urbanization(pop_density_path, target_raster_path):
    """

    UrbanRisk <- function(h){ifelse(h<1,1,1/(1+exp((h-3)/0.4)))}

    Args:
        pop_density_path (string): population density in per hectare

    """
    population_info = pygeoprocessing.get_raster_info(pop_density_path)
    population_nodata = population_info['nodata'][0]
    def op(pop_density_array):
        output = numpy.full(
            pop_density_array.shape, BYTE_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(pop_density_array, population_nodata))

        output[valid_pixels] = (
            1 / (1 + numpy.exp((pop_density_array[valid_pixels] - 3) / 0.4)))
        output[valid_pixels & (pop_density_array < 1)] = 1

        return output

    pygeoprocessing.raster_calculator(
        [(pop_density_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        BYTE_NODATA)

def _rural_population_density(population_path, target_raster_path):
    """ """
    #RuralDenRisk <- function(h){ifelse(h<1,h,1)}
    population_info = pygeoprocessing.get_raster_info(population_path)
    population_nodata = population_info['nodata'][0]
    def op(population_array):
        output = numpy.full(
            population_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(population_array, population_nodata))

        output[valid_pixels & (population_array < 1)] = population_array[valid_pixels & (population_array < 1)]
        output[valid_pixels & (population_array >= 1)] = 1

        return output

    pygeoprocessing.raster_calculator(
        [(population_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _water_velocity(slope_path, target_raster_path):
    """Slope suitability. """
    #WaterVel <- function(S){ifelse(S<=0.00014,-5714.3 * S + 1,-0.0029*S+0.2)}
    slope_info = pygeoprocessing.get_raster_info(slope_path)
    slope_nodata = slope_info['nodata'][0]

    def op(slope_array):
        degree = numpy.full(
            slope_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        output = numpy.full(
            slope_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = ~pygeoprocessing.array_equals_nodata(slope_array, slope_nodata)

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
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

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
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

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
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

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
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

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
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

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
        source_population_raster_path, target_pop_count_raster_path,
        target_pop_density_raster_path,
        lulc_pixel_size, lulc_bb, lulc_projection_wkt, working_dir):
    """Resample a population raster without losing or gaining people.

    Population rasters are an interesting special case where the data are
    neither continuous nor categorical, and the total population count
    typically matters. Common resampling methods for continuous
    (interpolation) and categorical (nearest-neighbor) datasets leave room for
    the total population of a resampled raster to significantly change. This
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
    pixel_area = abs(numpy.multiply(*population_raster_info['pixel_size']))
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
    pygeoprocessing.warp_raster(
        density_raster_path,
        target_pixel_size=lulc_pixel_size,
        target_raster_path=target_pop_density_raster_path,
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
        valid_mask = ~pygeoprocessing.array_equals_nodata(density, FLOAT32_NODATA)
        out_array[valid_mask] = density[valid_mask] * target_pixel_area
        return out_array

    pygeoprocessing.raster_calculator(
        [(target_pop_density_raster_path, 1)],
        _convert_density_to_population,
        target_pop_count_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

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
            valid_mask = ~pygeoprocessing.array_equals_nodata(array, source_nodata)

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
        #'op': lambda x: (x/pop_pixel_area)*10000,  # convert count per pixel to meters sq to hectares
        'op': lambda x: (x / pop_pixel_area) / 10000,  # convert count per pixel to meters sq to hectares
        'rasters': [pop_count_path],
        'target_path': target_path,
        'target_nodata': -1,
    }

    pygeoprocessing.raster_map(**kwargs)

def _pop_density_to_hectares(pop_density_path, target_path):
    """Population density in square km to population density in hectares."""
    population_raster_info = pygeoprocessing.get_raster_info(pop_density_path)

    kwargs={
        'op': lambda x: x/100,
        'rasters': [pop_density_path],
        'target_path': target_path,
        'target_nodata': -1,
    }

    pygeoprocessing.raster_map(**kwargs)




def _kernel_gaussian(distance, max_distance):
    """Create a gaussian kernel.

    Args:
        distance (numpy.array): An array of euclidean distances (in pixels)
            from the center of the kernel.
        max_distance (float): The maximum distance of the kernel.  Pixels that
            are more than this number of pixels will have a value of 0.

    Returns:
        ``numpy.array`` with dtype of numpy.float32 and same shape as
        ``distance.
    """
    kernel = numpy.zeros(distance.shape, dtype=numpy.float32)
    pixels_in_radius = (distance <= max_distance)
    kernel[pixels_in_radius] = (
        (numpy.e ** (-0.5 * ((distance[pixels_in_radius] / max_distance) ** 2))
         - numpy.e ** (-0.5)) / (1 - numpy.e ** (-0.5)))
    return kernel


def _convolve_and_set_lower_bound(
        signal_path_band, kernel_path_band, target_path, working_dir):
    """Convolve a raster and set all values below 0 to 0.

    Args:
        signal_path_band (tuple): A 2-tuple of (signal_raster_path, band_index)
            to use as the signal raster in the convolution.
        kernel_path_band (tuple): A 2-tuple of (kernel_raster_path, band_index)
            to use as the kernel raster in the convolution.  This kernel should
            be non-normalized.
        target_path (string): Where the target raster should be written.
        working_dir (string): The working directory that
            ``pygeoprocessing.convolve_2d`` may use for its intermediate files.

    Returns:
        ``None``
    """
    pygeoprocessing.convolve_2d(
        signal_path_band=signal_path_band,
        kernel_path_band=kernel_path_band,
        target_path=target_path,
        working_dir=working_dir,
        #mask_nodata=True,
        mask_nodata=False,
        #normalize_kernel=True
        )

    # Sometimes there are negative values that should have been clamped to 0 in
    # the convolution but weren't, so let's clamp them to avoid support issues
    # later on.
    target_raster = gdal.OpenEx(target_path, gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)
    target_nodata = target_band.GetNoDataValue()
    for block_data, block in pygeoprocessing.iterblocks(
            (target_path, 1)):
        valid_pixels = ~pygeoprocessing.array_equals_nodata(block, target_nodata)
        block[(block < 0) & valid_pixels] = 0
        target_band.WriteArray(
            block, xoff=block_data['xoff'], yoff=block_data['yoff'])

    target_band = None
    target_raster = None

@validation.invest_validator
def validate(args, limit_to=None):
    return validation.validate(args, MODEL_SPEC['args'])
