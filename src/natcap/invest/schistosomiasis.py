"""NatCap InVEST-like module for Schistosomiasis model.
There are suitability functions for each snail (BT and BG) and
parasite (SH and SM). There is a version for the wet season and the dry season;
i.e., NDVI for both wet and dry season, along with 8 NDVI suitability layers
(1 per species per season).
"""
#TODO:
# - Check out leafmap.org / leafmap python project

import logging
import os
import tempfile
import shutil
import subprocess
import json

from natcap.invest import spec_utils
from natcap.invest import gettext
from natcap.invest import utils
from natcap.invest.spec_utils import u
from natcap.invest import validation
import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import taskgraph
from osgeo import gdal
from osgeo import osr

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

logging.getLogger('taskgraph').setLevel('DEBUG')
# Was seeing a lot of font related logging
# https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger('matplotlib.font_manager').disabled = True

FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255

POP_RISK = {
    '0%': '247 251 255',
    '20%': '209 226 243',
    '40%': '154 200 224',
    '60%': '82 157 204',
    '80%': '29 108 177',
    '100%': '8 48 107',
    'nv': '1 1 1 0'
}

GENERIC_RISK = {
    '0%': '255 255 178',
    '25%': '254 204 92',
    '50%': '253 141 60',
    '75%': '240 59 32',
    '100%': '189 0 38',
    'nv': '0 0 0 0'
}

SCHISTO = "Schistosomiasis"

SNAIL_OPTIONS = [ 
        ("bt", "Default: Bulinus truncatus"),
        ("bg", "Default: Biomphalaria")]
PARASITE_OPTIONS = [
        ("sh", "Default: S. haematobium"),
        ("sm", "Defualt: S. mansoni")]

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
            "required": f"population_func_type == '{fn}'",
            "allowed": f"population_func_type == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    },
    'water_proximity': {
        f'water_proximity_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_water_proximity and water_proximity_func_type == '{fn}'",
            "allowed": f"calc_water_proximity and water_proximity_func_type == '{fn}'",
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
    'snail_water_temp': {
        f'snail_water_temp_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_temperature and snail_water_temp_func_type == '{fn}'",
            "allowed": f"calc_temperature and snail_water_temp_func_type == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    },
    'parasite_water_temp': {
        f'parasite_water_temp_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_temperature and parasite_water_temp_func_type == '{fn}'",
            "allowed": f"calc_temperature and parasite_water_temp_func_type == '{fn}'",
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

def user_input_id(input_id):
    return {
        f'custom_{input_id}_{fn}_param_{key}': {
            **spec,
            'name': f'{key}',
            "required": f"calc_custom_{input_id} and custom_func_type_{input_id} == '{fn}'",
            "allowed": f"calc_custom_{input_id} and custom_func_type_{input_id} == '{fn}'",
        }
        for fn in FUNCS for key, spec in SPEC_FUNC_COLS[fn].items()
    }

FUNC_PARAMS_USER = user_input_id

def temp_spec_func_types(default_type):
    """Specify multiple default options for temperature."""
    options_dict = {
        'snail': SNAIL_OPTIONS,
        'parasite': PARASITE_OPTIONS,
    }

    default_param_list = options_dict[default_type]

    default_options = {
        f"{key}":{"display_name": gettext(f"{name}")} for key, name in default_param_list
    }

    return {
        "type": "option_string",
        "options": {
            **default_options,
            "linear": {"display_name": gettext("Linear")},
            "exponential": {"display_name": gettext("exponential")},
            "scurve": {"display_name": gettext("scurve")},
            "trapezoid": {"display_name": gettext("trapezoid")},
            "gaussian": {"display_name": gettext("gaussian")},
        },
        "about": gettext(
            "The function type to apply to the suitability factor."),
        "name": gettext(f"{default_type} suitability function type")
    }

TEMP_SPEC_FUNC_TYPES = temp_spec_func_types

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
            ['decay_distance'],
            ["water_presence_path"],
            ["population_count_path", "population_func_type",
             {"Population parameters": list(FUNC_PARAMS['population'].keys())}],
#            ["calc_water_proximity", "water_proximity_func_type",
#             {"Water proximity parameters": list(FUNC_PARAMS['water_proximity'].keys())}],
            ["calc_water_depth", "water_depth_weight"],
            ["calc_temperature", "water_temp_dry_path", "water_temp_wet_path",
            "snail_water_temp_dry_weight", "snail_water_temp_wet_weight", "snail_water_temp_func_type", 
              {"Snail temperature parameters": list(FUNC_PARAMS['snail_water_temp'].keys())},
            "parasite_water_temp_dry_weight", "parasite_water_temp_wet_weight", "parasite_water_temp_func_type", 
              {"Parasite temperature parameters": list(FUNC_PARAMS['parasite_water_temp'].keys())}],
            ["calc_ndvi", "ndvi_func_type",
             "ndvi_dry_path", "ndvi_dry_weight",
             "ndvi_wet_path", "ndvi_wet_weight",
             {"NDVI parameters": list(FUNC_PARAMS['ndvi'].keys())}],
            ["calc_water_velocity", "water_velocity_func_type",
             "dem_path", "water_velocity_weight",
             {"Water velocity parameters": list(FUNC_PARAMS['water_velocity'].keys())}],
            ["calc_custom_one", "custom_func_type_one",
             "custom_path_one", "custom_weight_one",
             {"Input parameters": list(FUNC_PARAMS_USER(1).keys())}],
            ["calc_custom_two", "custom_func_type_two",
             "custom_path_two", "custom_weight_two",
             {"Input parameters": list(FUNC_PARAMS_USER(2).keys())}],
            ["calc_custom_three", "custom_func_type_three",
             "custom_path_three", "custom_weight_three",
             {"Input parameters": list(FUNC_PARAMS_USER(3).keys())}],
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
            'ndvi_dry_raster_path', 'ndvi_wet_raster_path', 'water_presence_path',
            'custom_path_one', 'custom_path_two', 'custom_path_three'],
        'different_projections_ok': True,
    },
    'args': {
        'workspace_dir': spec_utils.WORKSPACE,
        'results_suffix': spec_utils.SUFFIX,
        'n_workers': spec_utils.N_WORKERS,
        "decay_distance": {
            "type": "number",
            "units": u.meter,
            "about": gettext("Maximum threat distance from water risk."),
            "name": gettext("max decay distance")
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
        **FUNC_PARAMS['population'],
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
            "required": True,
        },
        "population_func_type": {
            **SPEC_FUNC_TYPES,
            "required": True,
        },
        "calc_water_depth": {
            "type": "boolean",
            "about": gettext(
                "Calculate water depth. Using the water presence raster input, "
                "uses a water distance from shore as a proxy for depth."),
            "name": gettext("calculate water depth"),
            "required": False
        },
        "water_depth_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("water depth risk weight"),
            "required": "calc_water_depth",
            "allowed": "calc_water_depth"
        },
#        "calc_water_proximity": {
#            "type": "boolean",
#            "about": gettext("Calculate water proximity. Uses the water presence raster input."),
#            "name": gettext("calculate water proximity"),
#            "required": False
#        },
#        "water_proximity_func_type": {
#            **SPEC_FUNC_TYPES,
#            "required": "calc_water_proximity",
#            "allowed": "calc_water_proximity",
#        },
#        **FUNC_PARAMS['water_proximity'],
        "water_presence_path": {
            'type': 'raster',
            'name': 'water presence',
            'bands': {1: {'type': 'integer'}},
            'about': (
                "A raster indicating presence of water."
            ),
            "required": True,
        },
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
        "water_velocity_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("water velocity risk weight"),
            "required": "calc_water_velocity",
            "allowed": "calc_water_velocity"
        },
        "calc_temperature": {
            "type": "boolean",
            "about": gettext("Calculate water temperature."),
            "name": gettext("calculate water temperature"),
            "required": False
        },
        'water_temp_dry_path': {
            'type': 'raster',
            'name': 'dry season temperature raster',
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
        'water_temp_wet_path': {
            'type': 'raster',
            'name': 'wet season temperature raster',
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
        "snail_water_temp_func_type": {
            **temp_spec_func_types('snail'),
            "required": "calc_temperature",
            "allowed": "calc_temperature"
        },
        **FUNC_PARAMS['snail_water_temp'],
        "parasite_water_temp_func_type": {
            **temp_spec_func_types('parasite'),
            "required": "calc_temperature",
            "allowed": "calc_temperature"
        },
        **FUNC_PARAMS['parasite_water_temp'],
        "snail_water_temp_dry_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("snail water temp dry risk weight"),
            "required": "calc_temperature",
            "allowed": "calc_temperature"
        },
        "snail_water_temp_wet_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("snail water temp wet risk weight"),
            "required": "calc_temperature",
            "allowed": "calc_temperature"
        },
        "parasite_water_temp_dry_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("parasite water temp dry risk weight"),
            "required": "calc_temperature",
            "allowed": "calc_temperature"
        },
        "parasite_water_temp_wet_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("parasite water temp wet risk weight"),
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
        "ndvi_dry_path": {
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
        "ndvi_dry_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("ndvi dry risk weight"),
            "required": "calc_ndvi",
            "allowed": "calc_ndvi"
        },
        "ndvi_wet_path": {
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
        "ndvi_wet_weight": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("ndvi wet risk weight"),
            "required": "calc_ndvi",
            "allowed": "calc_ndvi"
        },
        "calc_custom_one": {
            "type": "boolean",
            "required": False,
            "about": gettext("User defined suitability function."),
            "name": gettext("Additional user defined suitability input.")
        },
        "custom_func_type_one": {
            **SPEC_FUNC_TYPES,
            "required": "calc_custom_one",
            "allowed": "calc_custom_one"
        },
        **FUNC_PARAMS_USER('one'),
        'custom_path_one': {
            'type': 'raster',
            'name': 'custom raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the user suitability."
            ),
            "required": "calc_custom_one",
            "allowed": "calc_custom_one"
        },
        "custom_weight_one": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("User risk weight"),
            "required": "calc_custom_one",
            "allowed": "calc_custom_one"
        },
        "calc_custom_two": {
            "type": "boolean",
            "required": "calc_custom_one",
            "allowed": "calc_custom_one",
            "about": gettext("User defined suitability function."),
            "name": gettext("Additional user defined suitability input.")
        },
        "custom_func_type_two": {
            **SPEC_FUNC_TYPES,
            "required": "calc_custom_two",
            "allowed": "calc_custom_two"
        },
        **FUNC_PARAMS_USER('two'),
        'custom_path_two': {
            'type': 'raster',
            'name': 'custom raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the user suitability."
            ),
            "required": "calc_custom_two",
            "allowed": "calc_custom_two"
        },
        "custom_weight_two": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("User risk weight"),
            "required": "calc_custom_two",
            "allowed": "calc_custom_two",
        },
        "calc_custom_three": {
            "type": "boolean",
            "required": "calc_custom_two",
            "allowed": "calc_custom_two",
            "about": gettext("User defined suitability function."),
            "name": gettext("Additional user defined suitability input.")
        },
        "custom_func_type_three": {
            **SPEC_FUNC_TYPES,
            "required": "calc_custom_three",
            "allowed": "calc_custom_three"
        },
        **FUNC_PARAMS_USER('three'),
        'custom_path_three': {
            'type': 'raster',
            'name': 'custom raster',
            'bands': {
                1: {'type': 'number', 'units': u.count}
            },
            'projected': True,
            'projection_units': u.meter,
            'about': (
                "A raster representing the user suitability."
            ),
            "required": "calc_custom_three",
            "allowed": "calc_custom_three"
        },
        "custom_weight_three": {
            "type": "ratio",
            "about": gettext("The weight this factor should have on overall risk."),
            "name": gettext("User risk weight"),
            "required": "calc_custom_three",
            "allowed": "calc_custom_three"
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
    'snail_water_temp_suit_dry': 'snail_water_temp_suit_dry.tif',
    'snail_water_temp_suit_wet': 'snail_water_temp_suit_wet.tif',
    'parasite_water_temp_suit_dry': 'parasite_water_temp_suit_dry.tif',
    'parasite_water_temp_suit_wet': 'parasite_water_temp_suit_wet.tif',
    'ndvi_suit_dry': 'ndvi_suit_dry.tif',
    'ndvi_suit_wet': 'ndvi_suit_wet.tif',
    'water_velocity_suit': 'water_velocity_suit.tif',
    'water_proximity_suit': 'water_proximity_suit.tif',
    'water_depth_suit': 'water_depth_suit.tif',
    'rural_pop_suit': 'rural_pop_suit.tif',
    'urbanization_suit': 'urbanization_suit.tif',
    'rural_urbanization_suit': 'rural_urbanization_suit.tif',
    #'water_stability_suit': 'water_stability_suit.tif',
    'habitat_stability_suit': 'habitat_stability_suit.tif',
    'habitat_suit_weighted_mean': 'habitat_suit_weighted_mean.tif',
    'custom_input_suit_one': 'custom_input_suit_one.tif',
    'custom_input_suit_two': 'custom_input_suit_two.tif',
    'custom_input_suit_three': 'custom_input_suit_three.tif',
    'normalized_convolved_risk': 'normalized_convolved_risk.tif',
}

_INTERMEDIATE_BASE_FILES = {
    'aligned_pop_count': 'aligned_population_count.tif',
    'aligned_pop_density': 'aligned_pop_density.tif',
    'aligned_water_temp_dry': 'aligned_water_temp_dry.tif',
    'aligned_water_temp_wet': 'aligned_water_temp_wet.tif',
    'aligned_ndvi_dry': 'aligned_ndvi_dry.tif',
    'aligned_ndvi_wet': 'aligned_ndvi_wet.tif',
    'aligned_mask': 'aligned_valid_pixels_mask.tif',
    'aligned_dem': 'aligned_dem.tif',
    'aligned_water_presence': 'aligned_water_presence.tif',
    'aligned_lulc': 'aligned_lulc.tif',
    'aligned_custom_input_one': 'aligned_custom_input_one.tif',
    'aligned_custom_input_two': 'aligned_custom_input_two.tif',
    'aligned_custom_input_three': 'aligned_custom_input_three.tif',
    'masked_population': 'masked_population.tif',
    'population_density': 'population_density.tif',
    'population_hectares': 'population_hectare.tif',
    'slope': 'slope.tif',
    'degree_slope': 'degree_slope.tif',
    'masked_lulc': 'masked_lulc.tif',
    'reprojected_admin_boundaries': 'reprojected_admin_boundaries.gpkg',
    'distance': 'distance.tif',
    'inverse_water_mask': 'inverse_water_mask.tif',
    'distance_from_shore': 'distance_from_shore.tif',
    'water_velocity_suit_plot': 'water_vel_suit_plot.png',
    'water_proximity_suit_plot': 'water_proximity_suit_plot.png',
    'water_temp_suit_dry_plot': 'water_temp_suit_dry_plot.png',
    'water_depth_suit_plot': 'water_depth_suit_plot.png',
    'custom_input_suit_one_plot': 'custom_input_suit_one_plot.png',
    'custom_input_suit_two_plot': 'custom_input_suit_two_plot.png',
    'custom_input_suit_three_plot': 'custom_input_suit_three_plot.png',
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

        args['population_func_type'] = 'default'
        args['population_table_path'] = ''
        args['population_count_path'] (string): (required) A string path to a
            GDAL-compatible population raster containing people count per
            square km.
        
        args['calc_water_velocity'] = True
        args['water_velocity_func_type'] = 'default'
        args['water_velocity_table_path'] = ''
        args['dem_path'] (string): (required) A string path to a
            GDAL-compatible population raster containing people count per
            square km.  Must be linearly projected in meters.

    """
    LOGGER.info(f"Execute {SCHISTO}")

    HABITAT_RISK_KEYS = [
        'water_velocity', 'water_temp_dry', 'ndvi_dry', 'water_temp_wet',
        'ndvi_wet', 'custom_one', 'custom_two', 'custom_three']

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
        #'water_proximity': _water_proximity,
        'water_depth': _water_depth_suit,
        }
    PLOT_PARAMS = {
        'temperature': (0, 50),
        'ndvi': (-1, 1),
        'population': (0, 10),
        'urbanization': (0, 10),
        'water_velocity': (0, 30),
        #'water_proximity': (0, 1000),
        'water_depth': (0, 2000),
        }

    output_dir = os.path.join(args['workspace_dir'], 'output')
    intermediate_dir = os.path.join(args['workspace_dir'], 'intermediate')
    func_plot_dir = os.path.join(intermediate_dir, 'plot_previews')
    color_profiles_dir = os.path.join(intermediate_dir, 'color-profiles')
    utils.make_directories(
            [output_dir, intermediate_dir, func_plot_dir, color_profiles_dir])

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
    
    # Capture parameters necessary for Notebook companion, save to JSON.
    nb_json_config_path = os.path.join(output_dir, 'nb-json-config.json')
    nb_json_config = {}

    ### Save plots of function choices
    # Read func params from table
    # Excluding 'water_proximity' for now.
    user_func_paths = [
        'ndvi', 'population', 'water_velocity', 'urbanization', 'water_depth']
    suit_func_to_use = {}
    # NOTE: saving a companion index.txt file for the notebook to be able
    # display the plots over the http server. This isn't an ideal solution.
    nb_json_config['plot_paths'] = []
    
    for suit_key in user_func_paths:
        if suit_key in ['urbanization', 'water_depth']:
            func_type = 'default'
        else:
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

        plot_png_name = f"{suit_key}-{func_type}.png"
        results = _generic_func_values(
            user_func, PLOT_PARAMS[suit_key], intermediate_dir, func_params)
        plot_path = os.path.join(func_plot_dir, plot_png_name)
        _plotter(
            results[0], results[1], save_path=plot_path,
            label_x=suit_key, label_y=func_type,
            title=f'{suit_key}--{func_type}', xticks=None, yticks=None)
        # Track the current plots in the NB json config
        nb_json_config['plot_paths'].append(plot_png_name)
    
    # Handle Temperature separately because of snail, parasite pairing
    user_func_paths = ['snail_water_temp', 'parasite_water_temp']
    for suit_key in user_func_paths:
        func_type = args[f'{suit_key}_func_type']
        if func_type in ['sh', 'sm', 'bg', 'bt']:
            func_params = {'op_key': func_type}
            user_func = DEFAULT_FUNC_TYPES['temperature']
        else:
            func_params = {}
            for key in SPEC_FUNC_COLS[func_type].keys():
                func_params[key] = float(args[f'{suit_key}_{func_type}_param_{key}'])
            user_func = FUNC_TYPES[func_type]

        suit_func_to_use[suit_key] = {
            'func_name':user_func,
            'func_params':func_params,
        }

        results = _generic_func_values(
            user_func, PLOT_PARAMS['temperature'], intermediate_dir, func_params)
        plot_path = os.path.join(func_plot_dir, f"{suit_key}-{func_type}.png")
        _plotter(
            results[0], results[1], save_path=plot_path,
            label_x=suit_key, label_y=func_type,
            title=f'{suit_key}--{func_type}', xticks=None, yticks=None)

    # Get the extents and center of the AOI for notebook companion
    aoi_info = pygeoprocessing.get_vector_info(args['aoi_vector_path'])
    # WGS84 WKT
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    wgs84_wkt = srs.ExportToWkt()
    wgs84_bb = pygeoprocessing.geoprocessing.transform_bounding_box(
            aoi_info['bounding_box'], aoi_info['projection_wkt'], wgs84_wkt)
    #aoi_lat_center = wgs84_bb[1] + ((wgs84_bb[3] - wgs84_bb[1]) / 2)
    #aoi_lon_center = wgs84_bb[0] + ((wgs84_bb[2] - wgs84_bb[0]) / 2)
    aoi_center = (((wgs84_bb[1] + wgs84_bb[3]) / 2), ((wgs84_bb[0] + wgs84_bb[2]) / 2))
    nb_json_config['aoi_center'] = aoi_center

    ### Align and set up datasets
    # Use the water presence raster for resolution and aligning
    squared_default_pixel_size = _square_off_pixels(
        args['water_presence_path'])

    # Built up a list of provided optional rasters to align
    raster_input_list = [args['water_presence_path']]
    aligned_input_list = [file_registry['aligned_water_presence']]
    # TODO: add 'user_1', 'user_2', etc... provided inputs
    conditional_list = [
        ('calc_temperature', ['water_temp_dry_path', 'water_temp_wet_path']),
        ('calc_ndvi', ['ndvi_dry_path', 'ndvi_wet_path']),
        ('calc_water_velocity', ['dem_path']),
        ('calc_custom_one', ['custom_path_one']),
        ('calc_custom_two', ['custom_path_two']),
        ('calc_custom_three', ['custom_path_three']),
    ]
    for conditional, key_list in conditional_list:
        if args[conditional]:
            temp_paths = [args[path_key] for path_key in key_list]
            raster_input_list += temp_paths
            temp_align_paths = [file_registry[f'aligned_{path_key[:-5]}'] for path_key in key_list]
            aligned_input_list += temp_align_paths 

    align_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        kwargs={
            'base_raster_path_list': raster_input_list,
            'base_vector_path_list': [args['aoi_vector_path']],
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
        file_registry['aligned_water_presence'])
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

    ### Production functions ###
    suitability_tasks = []
    habitat_suit_risk_paths = []
    habitat_suit_risk_weights = []
    outputs_to_tile = []

    ### Habitat stability
    # NOTE: not currently calculating this because we don't have the data.
    # We are just using a binary water presence raster input
#    habitat_stability_task = graph.add_task(
#        _habitat_stability,
#        kwargs={
#            'water_presence_path': file_registry['aligned_water_presence'],
#            'months': 1.75,
#            'target_raster_path': file_registry['habitat_stability_suit'],
#        },
#        target_path_list=[file_registry['habitat_stability_suit']],
#        task_name='habitat stability')
#    suitability_tasks.append(habitat_stability_task)
#    habitat_suit_risk_paths.append(file_registry['habitat_stability_suit'])
#    outputs_to_tile.append((file_registry['habitat_stability_suit'], default_color_path))


    ### Water velocity
    if args['calc_water_velocity']:
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

        # water velocity risk is actually being calculated over the landscape
        # and not just where water is present. should it be masked to 
        # water presence?
        water_vel_task = graph.add_task(
            suit_func_to_use['water_velocity']['func_name'],
            args=(file_registry[f'slope'], file_registry['water_velocity_suit']),
            kwargs=suit_func_to_use['water_velocity']['func_params'],
            dependent_task_list=[slope_task],
            target_path_list=[file_registry['water_velocity_suit']],
            task_name=f'Water Velocity Suit')
        suitability_tasks.append(water_vel_task)
        #habitat_suit_risk_paths.append(file_registry['water_velocity_suit'])
        #habitat_suit_risk_weights.append(float(args['water_velocity_weight']))
        #outputs_to_tile.append((file_registry['water_velocity_suit'], default_color_path))

    ### Proximity to water in meters
    # NOT USING this suitability metric. Production Func. 9 in colab.
#    dist_edt_task = graph.add_task(
#        func=pygeoprocessing.distance_transform_edt,
#        args=(
#            (file_registry['aligned_water_presence'], 1),
#            file_registry['distance'],
#            (default_pixel_size[0], default_pixel_size[0])),
#        target_path_list=[file_registry['distance']],
#        dependent_task_list=[align_task],
#        task_name='distance edt')
#
#    water_proximity_task = graph.add_task(
#        suit_func_to_use['water_proximity']['func_name'],
#        args=(file_registry['distance'], file_registry['water_proximity_suit']),
#        kwargs=suit_func_to_use['water_proximity']['func_params'],
#        dependent_task_list=[dist_edt_task],
#        target_path_list=[file_registry[f'water_proximity_suit']],
#        task_name=f'Water Proximity Suit')
#    suitability_tasks.append(water_proximity_task)
    #outputs_to_tile.append((file_registry[f'water_proximity_suit'], default_color_path))

    ### Combined rural population and urbanization
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
    #outputs_to_tile.append((file_registry[f'rural_pop_suit'], default_color_path))
    
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
    #outputs_to_tile.append((file_registry[f'urbanization_suit'], default_color_path))
    
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

    # Temperature and ndvi have different input drivers for wet and dry seasons.
    for season in ["dry", "wet"]:
        ### Water temperature
        if args['calc_temperature']:
            for temp_key in ['snail_water_temp', 'parasite_water_temp']:
                # NOTE: TODO I'm not sure this if/else is needed anymore if we use the kwargs signature approach
                if 'op_key' in suit_func_to_use[temp_key]['func_params']:
                    water_temp_task = graph.add_task(
                        suit_func_to_use[temp_key]['func_name'],
                        kwargs={
                            'water_temp_path':file_registry[f'aligned_water_temp_{season}'],
                            'target_raster_path':file_registry[f'{temp_key}_suit_{season}'],
                            **suit_func_to_use[temp_key]['func_params'],
                            },
                        dependent_task_list=[align_task],
                        target_path_list=[file_registry[f'{temp_key}_suit_{season}']],
                        task_name=f'{temp_key} suit for {season}')
                else:
                    water_temp_task = graph.add_task(
                        suit_func_to_use[temp_key]['func_name'],
                        args=(
                            file_registry[f'aligned_water_temp_{season}'],
                            file_registry[f'{temp_key}_suit_{season}'],
                        ),
                        kwargs=suit_func_to_use[temp_key]['func_params'],
                        dependent_task_list=[align_task],
                        target_path_list=[file_registry[f'{temp_key}_suit_{season}']],
                        task_name=f'{temp_key} suit for {season}')
                suitability_tasks.append(water_temp_task)
                habitat_suit_risk_paths.append(file_registry[f'{temp_key}_suit_{season}'])
                habitat_suit_risk_weights.append(float(args[f'{temp_key}_{season}_weight']))
                outputs_to_tile.append((file_registry[f'{temp_key}_suit_{season}'], default_color_path))

        ### Vegetation coverage (NDVI)
        if args['calc_ndvi']:
            ndvi_task = graph.add_task(
                suit_func_to_use['ndvi']['func_name'],
                args=(
                    file_registry[f'aligned_ndvi_{season}'],
                    file_registry[f'ndvi_suit_{season}'],
                ),
                kwargs=suit_func_to_use['ndvi']['func_params'],
                dependent_task_list=[align_task],
                target_path_list=[file_registry[f'ndvi_suit_{season}']],
                task_name=f'NDVI Suit for {season}')
            suitability_tasks.append(ndvi_task)
            habitat_suit_risk_paths.append(file_registry[f'ndvi_suit_{season}'])
            habitat_suit_risk_weights.append(float(args[f'ndvi_{season}_weight']))
            outputs_to_tile.append((file_registry[f'ndvi_suit_{season}'], default_color_path))

    ### Distance from shore, proxy for depth ###
    if args['calc_water_depth']:
        inverse_water_mask_task = graph.add_task(
            _inverse_water_mask_op,
            kwargs={
                'input_path': file_registry['aligned_water_presence'],
                'target_path': file_registry['inverse_water_mask'],
                },
            target_path_list=[file_registry['inverse_water_mask']],
            dependent_task_list=[align_task],
            task_name='inverse water mask')

        distance_from_shore_task = graph.add_task(
            func=pygeoprocessing.distance_transform_edt,
            args=(
                (file_registry['inverse_water_mask'], 1),
                file_registry['distance_from_shore'],
                (default_pixel_size[0], default_pixel_size[0])),
            target_path_list=[file_registry['distance_from_shore']],
            dependent_task_list=[inverse_water_mask_task],
            task_name='inverse distance edt')
            
        water_depth_suit_path = file_registry['water_depth_suit']
        water_depth_suit_task = graph.add_task(
            suit_func_to_use['water_depth']['func_name'],
            args=(
                file_registry[f'distance_from_shore'],
                water_depth_suit_path,
            ),
            kwargs=suit_func_to_use['water_depth']['func_params'],
            dependent_task_list=[distance_from_shore_task],
            target_path_list=[water_depth_suit_path],
            task_name=f'Water Depth Suit')

        suitability_tasks.append(water_depth_suit_task)
        habitat_suit_risk_paths.append(water_depth_suit_path)
        habitat_suit_risk_weights.append(float(args['water_depth_weight']))
        outputs_to_tile.append((water_depth_suit_path, default_color_path))

    ### Custom functions provided by user
    for custom_index in ['one', 'two', 'three']:
        if args[f'calc_custom_{custom_index}']:
            custom_task = graph.add_task(
                suit_func_to_use[f'custom_{custom_index}']['func_name'],
                args=(
                    file_registry[f'aligned_ndvi_{season}'],
                    file_registry[f'ndvi_suit_{season}'],
                ),
                kwargs=suit_func_to_use['ndvi']['func_params'],
                dependent_task_list=[align_task],
                target_path_list=[file_registry[f'ndvi_suit_{season}']],
                task_name=f'NDVI Suit for {season}')
            suitability_tasks.append(ndvi_task)
            habitat_suit_risk_paths.append(file_registry[f'ndvi_suit_{season}'])
            habitat_suit_risk_weights.append(float(args[f'ndvi_{season}_weight']))
            outputs_to_tile.append((file_registry[f'ndvi_suit_{season}'], default_color_path))


    ### Population proximity to water


    ### Weighted arithmetic mean of water risks
    weighted_mean_task = graph.add_task(
        _weighted_mean,
        kwargs={
            'rasters': habitat_suit_risk_paths,
            'weight_values': habitat_suit_risk_weights,
            'target_path': file_registry['habitat_suit_weighted_mean'],
            'target_nodata': BYTE_NODATA,
            },
        target_path_list=[file_registry['habitat_suit_weighted_mean']],
        dependent_task_list=suitability_tasks,
        task_name='weighted mean')
    outputs_to_tile.append((file_registry[f'habitat_suit_weighted_mean'], default_color_path))


    ### Convolve habitat suit weighted mean over land

    # TODO: add decay dist to be an input to the model
    # TODO: mask out water bodies to nodata and not include in risk
    #decay_dist_m = 5000
    #decay_dist_m = 15 * 1000
    decay_dist_m = float(args['decay_distance'])

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
        output_dir,
        f'convolved_hab_risk{suffix}.tif')
    convolved_hab_risk_task = graph.add_task(
        _convolve_and_set_lower_bound,
        kwargs={
            'signal_path_band': (file_registry['habitat_suit_weighted_mean'], 1),
            'kernel_path_band': (kernel_path, 1),
            'target_path': convolved_hab_risk_path,
            'working_dir': intermediate_dir,
            'normalize': False,
        },
        task_name=f'Convolve hab risk - {decay_dist_m}m',
        target_path_list=[convolved_hab_risk_path],
        dependent_task_list=[kernel_task, weighted_mean_task])
    outputs_to_tile.append((convolved_hab_risk_path, default_color_path))
        
    # min-max normalize the absolute risk convolution.
    # min is known to be 0, so we don't misrepresent positive risk values.
    normalize_task = graph.add_task(
        _normalize_raster,
        kwargs={
            'raster_path': convolved_hab_risk_path,
            'target_path': file_registry['normalized_convolved_risk'],
        },
        dependent_task_list=[convolved_hab_risk_task],
        target_path_list=[file_registry['normalized_convolved_risk']],
        task_name=f'Normalize convolved risk')
    outputs_to_tile.append((file_registry['normalized_convolved_risk'], default_color_path))
    
    # TODO: do we want to mask out water presence?
#    masked_convolved_path = os.path.join(
#        intermediate_dir,
#        f'masked_hab_risk_within_{decay_dist_m}{suffix}.tif')
#    mask_convolve_task = graph.add_task(
#        _water_mask_op,
#        kwargs={
#            'input_path': convolved_hab_risk_path,
#            'mask_path': file_registry['aligned_water_presence'],
#            'target_path': masked_convolved_path,
#        },
#        task_name=f'Mask convolve hab risk - {decay_dist_m}m',
#        target_path_list=[masked_convolved_path],
#        dependent_task_list=[convolved_hab_risk_task])

    base_risk_path_list = [convolved_hab_risk_path, file_registry['normalized_convolved_risk']] 
    base_task_list = [normalize_task, convolved_hab_risk_task] 
    for calc_type, base_risk_path, base_task in zip(['abs', 'rel'], base_risk_path_list, base_task_list):
        ### Weight convolved risk by urbanization
        risk_to_pop_path = os.path.join(
            output_dir, f'risk_to_pop_{calc_type}{suffix}.tif')
        risk_to_pop_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs={
                'op': _multiply_op,
                'rasters': [file_registry['rural_urbanization_suit'], base_risk_path],
                'target_path': risk_to_pop_path,
                #'target_nodata': FLOAT32_NODATA,
                },
            target_path_list=[risk_to_pop_path],
            dependent_task_list=[base_task, urbanization_task],
            task_name=f'risk to population {calc_type}')
        outputs_to_tile.append((risk_to_pop_path, pop_color_path))
        
        # water habitat suitability gets at the risk of maximum potential schisto exposure
        # schisto exposure x urbanization gets at the risk of likelihood of exposure given socioeconomic factors
        # final risk, is population. Where are there the most people at the highest risk.

        ### Multiply risk_to_pop by people count?
        # Want to get to how many people are at risk
        # Multiply by count or by density
        # TODO: raw and scaled outputs for convolved risk, urbanization x raw convolved, and risk to people
        risk_to_pop_count_path = os.path.join(
            output_dir, f'risk_to_pop_count_{calc_type}{suffix}.tif')
        risk_to_pop_count_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs={
                'op': _multiply_op,
                'rasters': [risk_to_pop_path, file_registry['aligned_pop_count']],
                'target_path': risk_to_pop_count_path,
                #'target_nodata': FLOAT32_NODATA,
                },
            target_path_list=[risk_to_pop_count_path],
            dependent_task_list=[risk_to_pop_task],
            task_name=f'risk to pop_count {calc_type}')
        outputs_to_tile.append((risk_to_pop_count_path, pop_color_path))


    # NOTE: saving a companion jupyter-layers.txt file for the notebook to be able
    # display only the currently selected risk layers over the http server.
    # TODO: make a general notebook JSON config output. include:
    #     - Bounds, or extents to center the map on, lat,lon
    nb_json_config['layers'] = []
    for raster_path, _ in outputs_to_tile:
        base_name = os.path.splitext(os.path.basename(raster_path))[0]
        nb_json_config['layers'].append(base_name)
    with open(nb_json_config_path, 'w', encoding='utf-8') as f:
        json.dump(nb_json_config, f, ensure_ascii=False, indent=4)

    graph.close()
    graph.join()

    # Write color profiles to text file for gdaldem 
    default_color_path = os.path.join(color_profiles_dir, 'generic-risk-style.txt')
    pop_color_path = os.path.join(color_profiles_dir, 'generic-pop-risk-style.txt')
    color_path_list = [default_color_path, pop_color_path]
    for color_profile, profile_path in zip([GENERIC_RISK, POP_RISK], color_path_list):
        with open(profile_path, 'w') as f:
            for break_key, rgb_val in color_profile.items():
                f.write(break_key + ' ' + rgb_val + '\n')

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
        _tile_raster(raster_path, color_path)


    LOGGER.info("Model completed")


def _water_depth_suit(shore_distance_path, target_raster_path):
    """ """
    #'y = y1 - (y2 - y1)/(x2-x1)  * x1 + (y2 - y1)/(x2-x1) * x 
    raster_info = pygeoprocessing.get_raster_info(shore_distance_path)
    raster_nodata = raster_info['nodata'][0]
    
    # Need to define the shape of the function
    # Taken from the shared google sheet
    xa = 0 
    ya = 1 
    xb = 210  
    yb = 0.097
    xc = 211
    yc = 0.076184 
    xd = 2000
    yd = 0

    slope_one = (yb - ya) / (xb - xa)
    slope_two = (yc - yb) / (xc - xb)
    slope_three = (yc - yd) / (xc - xd)
    y_intercept_two = yb - (slope_two * xb)
    y_intercept_three = yc - (slope_three * xc)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = ~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata)

        # First line
        mask_one = valid_pixels & (raster_array <= xb)
        output[mask_one] = (slope_one * raster_array[mask_one]) + ya
        
        # Second line
        mask_two = valid_pixels & (raster_array > xb) & (raster_array <= xc)
        output[mask_two] = (slope_two * raster_array[mask_two]) + y_intercept_two
        
        # Third line
        mask_three = valid_pixels & (raster_array > xc) & (raster_array <= xd)
        output[mask_three] = (slope_three * raster_array[mask_three]) + y_intercept_three

        # Everything greater than xd is 0
        mask_final = valid_pixels & (raster_array > xd)
        output[mask_final] = 0

        return output

    pygeoprocessing.raster_calculator(
        [(shore_distance_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)

def _inverse_water_mask_op(input_path, target_path):
    """
    """
    input_info = pygeoprocessing.get_raster_info(input_path)
    input_nodata = input_info['nodata'][0]
    input_datatype = input_info['datatype']
    #numpy_datatype = pygeoprocessing._gdal_to_numpy_type(input_datatype)

    def _inverse_op(input_array):
        output = numpy.full(input_array.shape, input_nodata)
        nodata_mask = pygeoprocessing.array_equals_nodata(input_array, input_nodata)
        water_mask = input_array == 1

        output[water_mask] = 0
        output[nodata_mask] = 1

        return output
    
    pygeoprocessing.raster_calculator(
        [(input_path, 1)], _inverse_op, target_path,
        input_datatype, input_nodata)

def _water_mask_op(input_path, mask_path, target_path):
    """
    """
    input_info = pygeoprocessing.get_raster_info(input_path)
    input_nodata = input_info['nodata'][0]
    mask_info = pygeoprocessing.get_raster_info(mask_path)
    mask_nodata = mask_info['nodata'][0]

    def _mask_op(input_array, mask_array):
        output = numpy.full(
            input_array.shape, input_nodata, dtype=numpy.float32)
        nodata_mask = pygeoprocessing.array_equals_nodata(input_array, input_nodata)

        mask = mask_array == 1
        output[mask] = input_nodata
        output[nodata_mask] = input_nodata

        return output
    
    pygeoprocessing.raster_calculator(
        [(input_path, 1), (mask_path, 1)],
        _mask_op, target_path, gdal.GDT_Float32, input_nodata)


def _weighted_mean(rasters, weight_values, target_path, target_nodata):
    """Weighted arithmetic mean wrapper."""

    def _weighted_mean_op(*arrays):
        """
        raster_map op for weighted arithmetic mean of habitat suitablity risk layers.
        `arrays` is expected to be a list of numpy arrays
        """

        return numpy.average(arrays, axis=0, weights=weight_values)

    pygeoprocessing.raster_map(
        op=_weighted_mean_op,
        rasters=rasters,
        target_path=target_path,
        target_nodata=target_nodata,
        #target_dtype=numpy.float32,
    )


def _normalize_raster(raster_path, target_path):
    """Min-Max normalization with mininum fixed to 0."""

    raster = gdal.OpenEx(raster_path) 
    band = raster.GetRasterBand(1)
    # returns (min, max, mean, std)
    stats = band.GetStatistics(False, True)
    max_val = stats[1]
    min_val = 0

    def _normalize_op(array):
        """raster_map op for normalization."""

        return (array - min_val) / (max_val - min_val)

    pygeoprocessing.raster_map(
        op=_normalize_op,
        rasters=[raster_path],
        target_path=target_path,
    )

def _rural_urbanization_combined(pop_density_path, rural_path, urbanization_path, target_raster_path):
    """Combine the rural and urbanization functions."""
    rural_info = pygeoprocessing.get_raster_info(rural_path)
    rural_nodata = rural_info['nodata'][0]
    urbanization_info = pygeoprocessing.get_raster_info(urbanization_path)
    urbanization_nodata = urbanization_info['nodata'][0]

    def _rural_urbanization_op(pop_density_array, rural_array, urbanization_array):
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
    gdaldem_cmd = f'gdaldem color-relief -q -alpha -co COMPRESS=LZW {raster_path} {color_relief_path} {rgb_raster_path}'
    subprocess.call(gdaldem_cmd, shell=True)
    tile_cmd = f'gdal2tiles --xyz -r near -q -e --zoom=1-10 --process=4 {rgb_raster_path} {tile_dir}'
    print(tile_cmd)
    subprocess.call(tile_cmd, shell=True)

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
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 or higher than 35 set to 0
    valid_range_mask = (temp_array>=16) & (temp_array<=35) & ~nodata_pixels
    output[valid_range_mask] = (
        -0.003 * (268 / (temp_array[valid_range_mask] - 14.2) - 335) + 0.0336538)
    output[~nodata_pixels & (temp_array < 16)] = 0
    output[~nodata_pixels & (temp_array > 35)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output

def _water_temp_op_sh(temp_array, temp_nodata):
    """Water temperature suitability for S. haematobium."""
    #ShWaterTemp <- function(Temp){ifelse(Temp<17, 0,ifelse(Temp<=33, -0.006 * (295/(Temp - 15.3) - 174) + 0.056, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=17) & (temp_array<=33) & ~nodata_pixels
    output[valid_range_mask] = (
        -0.006 * (295 / (temp_array[valid_range_mask] - 15.3) - 174) + 0.056)
    output[~nodata_pixels & (temp_array < 17)] = 0
    output[~nodata_pixels & (temp_array > 33)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output

def _water_temp_op_bt(temp_array, temp_nodata):
    """Water temperature suitability for Bulinus truncatus."""
    #BtruncatusWaterTempNEW <- function(Temp){ifelse(Temp<17, 0,ifelse(Temp<=33, -48.173 + 8.534e+00 * Temp + -5.568e-01 * Temp^2 + 1.599e-02 * Temp^3 + -1.697e-04 * Temp^4, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=17) & (temp_array<=33) & ~nodata_pixels
    output[valid_range_mask] = (
        -48.173 + (8.534 * temp_array[valid_range_mask]) + 
        (-5.568e-01 * numpy.power(temp_array[valid_range_mask], 2)) +
        (1.599e-02 * numpy.power(temp_array[valid_range_mask], 3)) +
        (-1.697e-04 * numpy.power(temp_array[valid_range_mask], 4)))
    output[~nodata_pixels & (temp_array < 17)] = 0
    output[~nodata_pixels & (temp_array > 33)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output

def _water_temp_op_bg(temp_array, temp_nodata):
    """Water temperature suitability for Biomphalaria."""
    #BglabrataWaterTempNEW <- function(Temp){ifelse(Temp<16, 0,ifelse(Temp<=35, -29.9111 + 5.015e+00 * Temp + -3.107e-01 * Temp^2 +8.560e-03 * Temp^3 + -8.769e-05 * Temp^4, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=16) & (temp_array<=35) & ~nodata_pixels
    output[valid_range_mask] = (
        -29.9111 + (5.015 * temp_array[valid_range_mask]) + 
        (-3.107e-01 * numpy.power(temp_array[valid_range_mask], 2)) +
        (8.560e-03 * numpy.power(temp_array[valid_range_mask], 3)) +
        (-8.769e-05 * numpy.power(temp_array[valid_range_mask], 4)))
    output[~nodata_pixels & (temp_array < 16)] = 0
    output[~nodata_pixels & (temp_array > 35)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output

def _water_temp_suit(water_temp_path, target_raster_path, op_key):
    """

        Args:
            water_temp_path (string):
            target_raster_path (string):
            op_key (string):

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
            ndvi_array.shape, BYTE_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(ndvi_array, ndvi_nodata))

        # if temp is less than 0 set to 0
        mask = valid_pixels & (ndvi_array>=0) & (ndvi_array<=0.3)
        output[mask] = (3.33 * ndvi_array[mask])
        output[valid_pixels & (ndvi_array < 0)] = 0
        output[valid_pixels & (ndvi_array > 0.3)] = 1
        output[~valid_pixels] = BYTE_NODATA

        return output

    pygeoprocessing.raster_calculator(
        [(ndvi_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        BYTE_NODATA)

def _water_proximity(water_proximity_path, target_raster_path):
    """ """
    #ProxRisk <- function(prox){ifelse(prox<1000, 1,ifelse(prox<=15000, -0.0000714 * prox + 1.0714,0))}
    water_proximity_info = pygeoprocessing.get_raster_info(water_proximity_path)
    water_proximity_nodata = water_proximity_info['nodata'][0]
    def op(water_proximity_array):
        output = numpy.full(
            water_proximity_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(water_proximity_array, water_proximity_nodata))

        # 
        lt_km_mask = valid_pixels & (water_proximity_array < 1000)
        lt_gt_mask = valid_pixels & (water_proximity_array >= 1000) & (water_proximity_array <= 15000)
        gt_mask = valid_pixels & (water_proximity_array > 15000)
        output[lt_km_mask] = 1
        output[lt_gt_mask] = -0.0000714 * water_proximity_array[lt_gt_mask] + 1.0714
        output[gt_mask] = 0

        return output

    pygeoprocessing.raster_calculator(
        [(water_proximity_path, 1)], op, target_raster_path, gdal.GDT_Float32,
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
    target_pixel_area = abs((
        numpy.multiply(*lulc_pixel_size) * target_srs.GetLinearUnits()) / 1e6)

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

    # 10,000 square meters equals 1 hectares.
    kwargs={
        'op': lambda x: (x / pop_pixel_area) * 10000,  # convert count per pixel to meters sq to hectares
        #'op': lambda x: (x / pop_pixel_area) / 10000,  # convert count per pixel to meters sq to hectares
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
        signal_path_band, kernel_path_band, target_path, working_dir, normalize):
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
        normalize (bool): whether to normalize the kernel

    Returns:
        ``None``
    """
    pygeoprocessing.convolve_2d(
        signal_path_band=signal_path_band,
        kernel_path_band=kernel_path_band,
        target_path=target_path,
        working_dir=working_dir,
        #ignore_nodata_and_edges=True,
        ignore_nodata_and_edges=False,
        mask_nodata=False,
        normalize_kernel=normalize
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
