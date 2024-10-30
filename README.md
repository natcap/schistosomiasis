# Schistosomiasis Plugin for InVEST

In collaboration with the De Leo Lab at Stanford University we developed a tool to incorporate schistosomiasis risk into dam and water infrastructure development decisions. This work is based on the [Walz et al. 2015](https://doi.org/10.1371/journal.pntd.0004217) paper with additional contributions by Andy Chamberlain and Guilio De Leo from the De Leo Lab, and Lisa Mandle and Doug Denu from the Natural Capital Project.

## About Schistosomiasis

## About InVEST Plugins

## The model

Add info here about the schistosomiasis model, inputs, calculating risk. Essentially could be the user guide section.

## Developer Guide

### Running in InVEST

Currently, plugins for InVEST is still under active development. A fork of InVEST has been created to specifically support running the schisto plugin ([schisto-invest]](https://github.com/natcap/schistosomiasis-invest)). In the future this plugin will be run directly in InVEST.

**Download schist-invest**: [a link to download invest]

**Install plugin into schisto-invest**
With the schistosomiasis InVEST version installed and open, select "Manage Plugins" in the top right. Copy the git url into the "Add a plugin" field and select "Add". For this plugin the url would be: https://github.com/natcap/schistosomiasis.

The plugin will take several minutes to install, between 5-10 minutes. Once the plugin is installed a new "Schistosomiasis" model option will appear in the home list of models, with a "plugin" badge next to it. Select this to run the schistosomiasis model.

### The companion Jupyter Notebook

### Modifying the model

The core of the model lives in `src/natcap/invest/schistosomiasis.py`. 
