Multimelt
=========

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

.. arc3o documentation master file, created by
   sphinx-quickstart on Mon Aug 10 11:47:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

This is a package containing the most prominent parameterisations for the melt at the base of Antarctic ice shelves

To use this package, clone the github repository locally and install it into you python/conda environment via <code>pip install .</code>

It contains two example python notebooks:
- prepare_mask_example.ipynb : a script using geometric circum-Antarctic input to produce masks of the ice shelves, and the needed box characteristics and plume characteristics
- compute_melt_example.ipynb : a script showing how to apply the melting functions

More detailed explanations and documentation coming soon...
Don't hesitate to contact me if any questions arise: clara.burgard@univ-grenoble-alpes.fr


ARC3O
=====
.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs|

.. |docs| image:: http://readthedocs.org/projects/arc3o/badge/?version=latest
    :alt: Documentation Status
    :target: http://arc3o.readthedocs.io/en/latest/?badge=latest

.. end-badges

.. arc3o documentation master file, created by
   sphinx-quickstart on Mon Aug 10 11:47:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation about the Arctic Ocean Observation Operator!
=========================================================================

The Arctic Ocean Observation Operator (ARC3O) computes brightness temperatures at 6.9 GHz, 
vertical polarization, based on climate model output. More information about the motivation, 
structure and evaluation can be found in `Burgard et al., 2020a`_ and `Burgard et al., 2020b`_. 

Currently, it is customized for output of the Max Planck Institute Earth System Model but can be 
used for other models if the variable names are changed accordingly in the ARC3O functions.

You can access the detailed documentation here: https://arc3o.readthedocs.io/

How to cite ARC3O
-----------------

The detailed description and evaluation of ARC3O is found in `Burgard et al., 2020b`_ and should 
therefore, when used, be cited as follows:

Burgard, C., Notz, D., Pedersen, L. T., and Tonboe, R. T. (2020): "The Arctic Ocean Observation Operator for 6.9 GHz (ARC3O) – Part 2: Development and evaluation", *The Cryosphere*, 14, 2387–2407, https://doi.org/10.5194/tc-14-2387-2020.

.. _`Burgard et al., 2020a`: https://tc.copernicus.org/articles/14/2369/2020/
.. _`Burgard et al., 2020b`: https://tc.copernicus.org/articles/14/2387/2020/


