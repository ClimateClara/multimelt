.. multimelt documentation master file, created by
   sphinx-quickstart on Mon Mar 28 17:45:38 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. arc3o documentation master file, created by
   sphinx-quickstart on Mon Aug 10 11:47:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation about the multimelt package!
=========================================================

Multimelt contains a large amount of existing basal melt parameterisations for Antarctic ice shelves. The functions are written as such that the main input needed are temperature and salinity profiles in front of the ice shelf, the rest happens in the functions.

Also, multimelt contains functions to create masks of the Antarctic continent and the different ice shelves, and other geographical parameters on a circum-Antarctic scale, for input on a stereographic grid. 


Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Getting started:
   
   start/about
   start/installation

.. toctree::
   :maxdepth: 2
   :caption: User's Guide:

   guide/prepare_masks
   guide/prepare_input_T_S_profiles
   guide/compute_melt
   
.. toctree::
   :maxdepth: 2
   :caption: Help & References:   
   
   api/multimelt
   literature/references
   literature/publications
    

How to cite multimelt
---------------------

The detailed description of the application of the functions in multimelt is found in `Burgard et al., 2022`_ and should therefore, when used, be cited as follows:

Burgard, C., Jourdain, N. C., Reese, R., Jenkins, A., and Mathiot, P.: An assessment of basal melt parameterisations for Antarctic ice shelves, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2022-32, in review, 2022. 


.. _`Burgard et al., 2022`: https://doi.org/10.5194/tc-2022-32


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
