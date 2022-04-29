.. _prod_melt:

multimelt provides two types of "services". On the one hand, it can produce masks of the different circum-Antarctic ice shelves and geometric properties on each ice shelf level (neede for the pdifferent parameterisations). On the other hand, it computes 2D and 1D metrics related to basal melt of ice shelves.

The procedure to create the masks and box and plume characteristics is shown in the notebook ``prepare_mask_example.ipynb``. The steps are also explained more in detail in :ref:`prod_masks`.

The procedure to compute melt rates from temperature and salinity profiles is shown in the notebook ``compute_melt_example.ipynb``. The steps are also explained more in detail in :ref:`prep_mask_general`, :ref:`prep_box_charac` and :ref:`prep_plume_charac`.

How to run (2) : Computing basal melt rates
===========================================

The melt function is designed to receive information about the ice-shelf geometry (formatted with :func:`multimelt.create_isf_mask_functions.create_mask_and_metadata_isf`,:func:`multimelt.plume_functions.prepare_plume_charac`, and :func:`multimelt.box_functions.box_charac_file`),  and about temperature and salinity profiles in front of one or several ice shelves. The output is a range of 2D and 1D variables describing the melt rates at the base of the ice shelf or ice shelves, if you have several. 

Input data
^^^^^^^^^^

To be continued...

Running
^^^^^^^

To be continued...

Output
^^^^^^

To be continued...
