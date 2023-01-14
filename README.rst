Welcome to the documentation about the multimelt package!
=========================================================

.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs|

.. |docs| image:: http://readthedocs.org/projects/multimelt/badge/?version=latest
    :alt: Documentation Status
    :target: http://multimelt.readthedocs.io/en/latest/?badge=latest

.. end-badges

.. multimelt documentation master file, created by
   sphinx-quickstart on Mon Aug 10 11:47:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Multimelt contains a large amount of existing basal melt parameterisations for Antarctic ice shelves. The functions are written as such that the main input needed are temperature and salinity profiles in front of the ice shelf, the rest happens in the functions.

Also, multimelt contains functions to create masks of the Antarctic continent and the different ice shelves, and other geographical parameters on a circum-Antarctic scale, for input on a stereographic grid. 

To use this package, clone the github repository locally and install it into you python/conda environment via 

.. code-block:: bash
    
    pip install .

It contains four example python notebooks:

* ``prepare_mask_example.ipynb`` : a script using geometric circum-Antarctic input to produce masks of the ice shelves, and the needed box characteristics and plume characteristics
* ``conversion_CTtoPT_SAtoSP.ipynb`` : a script to convert 3D fields of conservative temperature to potential temperature and 3D fields of absolute salinity to practical salinity
* ``T_S_profiles_per_ice_shelf.ipynb`` : a script to created averaged temperature and salinity profiles in front of the different ice shelves
* ``compute_melt_example.ipynb`` : a script showing how to apply the melting functions


The documentation can be found here: http://multimelt.readthedocs.io/

Don't hesitate to contact me if any questions arise: clara.burgard@univ-grenoble-alpes.fr

How to cite multimelt
---------------------

The detailed description of the application of the functions in multimelt is found in `Burgard et al., 2022`_ and should therefore, when used, be cited as follows:

Burgard, C., Jourdain, N. C., Reese, R., Jenkins, A., and Mathiot, P. (2022): An assessment of basal melt parameterisations for Antarctic ice shelves, The Cryosphere, https://doi.org/10.5194/tc-16-4931-2022. 


.. _`Burgard et al., 2022`: https://doi.org/10.5194/tc-16-4931-2022



