.. _prepare_prof:


multimelt provides three types of "services". (1)It can produce masks of the different circum-Antarctic ice shelves and geometric properties on each ice shelf level (neede for the pdifferent parameterisations). (2) It computes mean profiles of temperature and salinity in front of the ice shelves in given domains of interest. (3) It computes 2D and 1D metrics related to basal melt of ice shelves.

The procedure to create the masks and box and plume characteristics is shown in the notebook ``prepare_mask_example.ipynb``. The steps are also explained more in detail in :ref:`prep_mask_general`, :ref:`prep_box_charac` and :ref:`prep_plume_charac`.

The procedure to compute and format the input temperature (T) and salinity (S) profiles from 2D fields is shown in the notebook ``T_S_profiles_per_ice_shelf.ipynb``. The steps are also explained more in detail in :ref:`prepare_prof`. !BE CAREFUL! If your 2D fields are in conservative temperature and absolute salinity, do not forget to convert them with the script ``conversion_CTtoPT_SAtoSP.ipynb``.

The procedure to compute melt rates from temperature and salinity profiles is shown in the notebook ``compute_melt_example.ipynb``. The steps are also explained more in detail in :ref:`prod_melt`.


How to run (2) : Preparing the input profiles
==========================================================

Conversion from conservative temperature to potential temperature and from absolute salinity to practical salinity
------------------------------------------------------------------------------------------------------------------

Input data
^^^^^^^^^^

Currently, most of the functions of ``multimelt`` are tailored for **circum-Antarctic** model output from NEMO, the Nucleus for European Modelling of the Ocean :cite:p:`nemo19`, interpolated to a south polar stereographic grid (EPSG:3031). However, it can be used for other models/grids if the variable names are changed accordingly either in the functions or the model's output (requires a bit of digging).


The conversion script ``conversion_CTtoPT_SAtoSP.ipynb`` needs these files as input:

* ``file_mask``: file containing the depth information of your depth coordinate, for each x/y pair if it is not constant (we will assume that it is approximately constant and take a mean over all x/y pairs at each depth level)
* ``file_isf``: file containing masks created earlier with :func:`multimelt.create_isf_mask_functions.create_mask_and_metadata_isf`
* ``file_TS_orig``: 3D field of conservative temperature, absolute salinity and sea-surface temperature for just one time step
* ``ts_files``: 3D fields of conservative temperature and absolute salinity for all time steps (can be several files), they need to contain the 3D potential temperature ``votemper``, the 3D practical salinity ``vosaline``, and the 2D sea-surface temperature ``sosst``

Running
^^^^^^^

Use the notebook ``conversion_CTtoPT_SAtoSP.ipynb``.


Output
^^^^^^

The resulting netcdf file contains the following variables:

* ``theta_ocean``: 3D field of potential temperature
* ``salinity_ocean``: 3D field of practical salinity 

Prepare input temperature and salinity profiles averaged over given domains in front of the ice shelves
-------------------------------------------------------------------------------------------------------

Input data
^^^^^^^^^^

Currently, most of the functions of ``multimelt`` are tailored for **circum-Antarctic** model output from NEMO, the Nucleus for European Modelling of the Ocean :cite:p:`nemo19`, interpolated to a south polar stereographic grid (EPSG:3031). However, it can be used for other models/grids if the variable names are changed accordingly either in the functions or the model's output (requires a bit of digging).

The input profiles are currently computed averaged over five domains: within 10, 25, 50 and 100 km of the ice-shelf front on the continental shelf, and offshore (for bathymetry deeper than the continental shelf).

The computation script ``T_S_profiles_per_ice_shelf.ipynb`` (1) calculates the distance to the ice front for the small domain in front of the ice shelf and (2) takes the ocean points at a given distance of the ice front and averages over them.It needs these files as input:

* ``file_mask_orig``: file containing the variable ``bathy_metry``
* ``file_isf_orig``: file containing masks created earlier with :func:`multimelt.create_isf_mask_functions.create_mask_and_metadata_isf`
* ``T_S_ocean_oneyear`` and ``T_S_ocean_files``: 3D field of potential temperature ``theta_ocean`` and practical salinity ``salinity_ocean``

Running
^^^^^^^

Use the notebook ``T_S_profiles_per_ice_shelf.ipynb``.


Output
^^^^^^

The main result is the netcdf ``'T_S_mean_prof_corrected_km_contshelf_and_offshore_1980-2018.nc'`` containing profiles of ``theta_ocean`` and ``salinity_ocean`` averaged over different domains in front of the ice shelf (currently: within 10, 25, 50 and 100 km of the ice-shelf front on the continental shelf, and offshore for bathymetry deeper than the continental shelf).

It produces several temporary netcdf files needed in case the script crashes but that can be deleted afterwards:

* ``'dist_to_ice_front_only_contshelf_oneFRIS.nc'``
* ``'mask_offshore_oneFRIS.nc'``
* ``'ds_sum_for_mean_contshelf.nc'``
* ``'T_S_mean_prof_corrected_km_contshelf_1980-2018.nc'``
* ``'ds_sum_for_mean_offshore.nc'``
* ``'T_S_mean_prof_corrected_km_offshore_1980-2018.nc'``



