.. _prod_melt:

multimelt provides three types of "services". (1)It can produce masks of the different circum-Antarctic ice shelves and geometric properties on each ice shelf level (neede for the pdifferent parameterisations). (2) It computes mean profiles of temperature and salinity in front of the ice shelves in given domains of interest. (3) It computes 2D and 1D metrics related to basal melt of ice shelves.

The procedure to create the masks and box and plume characteristics is shown in the notebook ``prepare_mask_example.ipynb``. The steps are also explained more in detail in :ref:`prep_mask_general`, :ref:`prep_box_charac` and :ref:`prep_plume_charac`.

The procedure to compute and format the input temperature (T) and salinity (S) profiles from 2D fields is shown in the notebook ``T_S_profiles_per_ice_shelf.ipynb``. The steps are also explained more in detail in :ref:`prepare_prof`. !BE CAREFUL! If your 2D fields are in conservative temperature and absolute salinity, do not forget to convert them with the script ``conversion_CTtoPT_SAtoSP.ipynb``.

The procedure to compute melt rates from temperature and salinity profiles is shown in the notebook ``compute_melt_example.ipynb``. The steps are also explained more in detail in :ref:`prod_melt`.

How to run (3) : Computing basal melt rates
===========================================

The melt function is designed to receive information about the ice-shelf geometry (formatted with :func:`multimelt.create_isf_mask_functions.create_mask_and_metadata_isf`, :func:`multimelt.plume_functions.prepare_plume_charac`, and :func:`multimelt.box_functions.box_charac_file`),  and about temperature and salinity profiles in front of one or several ice shelves (formatted in ``T_S_profiles_per_ice_shelf.ipynb``). The output is a range of 2D and 1D variables describing the melt rates at the base of the ice shelf or ice shelves, if you have several. 

Input data
^^^^^^^^^^

To compute the melt, in :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`), you need

| > the list of ice shelf IDs you are interested in (``nisf_list``)


| > 2D geometric info (``geometry_info_2D`` : 

    * The variables contained in ``'plume_charac_file.nc'`` created as shown in :ref:`prep_plume_charac`
    * The variables ``'ISF_mask'``, ``'latitude'`` and ``'longitude'`` contained in ``'mask_file.nc'`` created as shown in :ref:`prep_mask_general`
    * ``ice_draft_pos``: ice draft depth (positive downwards)
    * ``grid_cell_area_weighted``: area of the grid cells weighted with the ice shelf concentration
    * ``isfdraft_conc``: ice shelf concentration
    
| > a stacked mask of the ice shelf regions to reduce calculation time and only select ice shelf point (``isf_stack_mask``). Can be created as follows:

.. code-block:: python

    isf_stack_mask = multimelt.useful_functions.create_stacked_mask(ISF_mask, nisf_list, ['y','x'], 'mask_coord')

    
| > 1D geometric info (``geometry_info_1D``) : 

    * The variables ``'front_bot_depth_avg'``, ``'front_bot_depth_max'``, and ``'isf_name'`` contained in ``'mask_file.nc'`` created as shown in :ref:`prep_mask_general`
    
| > Temperature and salinity profiles (``T_S_profile``): one dataset containing ``theta_ocean``, the potential temperature in °C, and ``salinity_ocean``, the practical salinity in psu, both over at least the dimensions ``Nisf`` and ``depth``.

| > Input parameters to the different parameterisations: e.g. ``gamma``, ``E0``, ``C``

*If you want to use the box or PICOP parameterisation*

 *> The variables contained in* ``'box_charac_file.nc'`` *created as shown in* :ref:`prep_box_charac` 


Running
^^^^^^^

To run the simple parameterisations, use the following command

.. code-block:: python

    nisf_list = geometry_info_1D.Nisf
    T_S_profile = file_TS.ffill(dim='depth')

    mparam = # POSSIBILITIES: ['linear_local', 'quadratic_local', 'quadratic_local_locslope', 'quadratic_local_cavslope', 'quadratic_mixed_mean', 'quadratic_mixed_locslope','quadratic_mixed_cavslope'] 

    gamma = # fill in
    ds_2D, ds_1D = meltf.calculate_melt_rate_1D_and_2D_all_isf(nisf_list, 
                                                                T_S_profile, g
                                                                geometry_info_2D, 
                                                                geometry_info_1D, 
                                                                isf_stack_mask, 
                                                                mparam, 
                                                                gamma, 
                                                                U_param=True)

    ds_2D.to_netcdf(outputpath_melt+'melt_rates_2D_'+mparam+'.nc')
    ds_1D.to_netcdf(outputpath_melt+'melt_rates_1D_'+mparam+'.nc')

To run the plume parameterisations, use the following command

.. code-block:: python

    nisf_list = geometry_info_1D.Nisf
    T_S_profile = file_TS.ffill(dim='depth')

    mparam = # POSSIBILITIES: ['lazero19_2', 'lazero19_modif2']

    gamma = # fill in
    E0 = # fill in

    ds_2D, ds_1D = meltf.calculate_melt_rate_1D_and_2D_all_isf(nisf_list, 
                                                                T_S_profile, 
                                                                geometry_info_2D, 
                                                                geometry_info_1D, 
                                                                isf_stack_mask,
                                                                mparam, 
                                                                gamma, 
                                                                E0=E0, 
                                                                verbose=True)

    ds_2D.to_netcdf(outputpath_melt+'melt_rates_2D_'+mparam+'.nc')
    ds_1D.to_netcdf(outputpath_melt+'melt_rates_1D_'+mparam+'.nc')

To run the box parameterisations, use the following command

.. code-block:: python

    nisf_list = geometry_info_1D.Nisf
    T_S_profile = file_TS.ffill(dim='depth') 
    picop_opt = 'no'

    nD_config = # POSSIBILITIES: 1 to 4
    pism_version = # POSSIBILITIES: 'yes' or 'no'

    mparam = 'boxes_'+str(nD_config)+'_pism'+pism_version+'_picop'+picop_opt

    C = # fill in
    gamma = # fill in

    ds_2D, ds_1D = meltf.calculate_melt_rate_1D_and_2D_all_isf(nisf_list, 
                                                                T_S_profile, 
                                                                geometry_info_2D, 
                                                                geometry_info_1D, 
                                                                isf_stack_mask, 
                                                                mparam, 
                                                                gamma,
                                                                C=C, 
                                                                angle_option='appenB', 
                                                                box_charac_2D=box_charac_all_2D, 
                                                                box_charac_1D=box_charac_all_1D, 
                                                                box_tot=nD_config, 
                                                                box_tot_option='nD_config', 
                                                                pism_version=pism_version, 
                                                                picop_opt=picop_opt)

    ds_2D.to_netcdf(outputpath_melt+'melt_rates_2D_'+mparam+'.nc')
    ds_1D.to_netcdf(outputpath_melt+'melt_rates_1D_'+mparam+'.nc')
    
To run the PICOP parameterisations, use the following command

.. code-block:: python

    nisf_list = geometry_info_1D.Nisf
    T_S_profile = file_TS.ffill(dim='depth') 

    nD_config = # POSSIBILITIES: 1 to 4    
    pism_version = # POSSIBILITIES: 'yes' or 'no'
    picop_opt = # POSSIBILITIES: '2018' or '2019'

    mparam = 'boxes_'+str(nD_config)+'_pism'+pism_version+'_picopyes'

    C = # for box part - fill in
    gamma = # for box part - fill in

    gamma_plume = # for plume part - fill in
    E0 = # for plume part - fill in

    ds_2D, ds_1D = meltf.calculate_melt_rate_1D_and_2D_all_isf(nisf_list, 
                                                                T_S_profile, 
                                                                geometry_info_2D, 
                                                                geometry_info_1D, 
                                                                isf_stack_mask, 
                                                                mparam, 
                                                                gamma,
                                                                C=C, 
                                                                E0=E0, 
                                                                angle_option='appenB',
                                                                box_charac_2D=box_charac_all_2D, 
                                                                box_charac_1D=box_charac_all_1D, 
                                                                box_tot=nD_config, 
                                                                box_tot_option='nD_config', 
                                                                pism_version=pism_version, 
                                                                picop_opt=picop_opt, 
                                                                gamma_plume=gamma_plume)

    ds_2D.to_netcdf(outputpath_melt+'melt_rates_2D_'+mparam+'.nc')
    ds_1D.to_netcdf(outputpath_melt+'melt_rates_1D_'+mparam+'.nc')

To run DeepMelt, use the following command

.. code-block:: python

    nisf_list = geometry_info_1D.Nisf
    T_S_profile = file_TS.ffill(dim='depth') 
    
    mparam = 'DeepMelt'
    
    path_model = './multimelt/deepmelt_extras/' # correct the path if it does not work
    
    geometry_info_2D_extended = geometry_info_2D.merge(file_isf[['dGL','dIF']]).merge(deepmelt_charac)
    
    deepmelt_norm_file = xr.open_dataset(path_model+'DeepMelt_norm_metrics.nc')
    deepmelt_norm = deepmelt_norm_file.to_dataframe()
    
    ds_2D, ds_1D = meltf.calculate_melt_rate_1D_and_2D_all_isf(nisf_list, 
                                                               T_S_profile, 
                                                               geometry_info_2D, 
                                                               geometry_info_1D, 
                                                               isf_stack_mask, 
                                                               mparam, 
                                                               None,
                                                               deepmelt_model=path_model+'DeepMelt_',
                                                               deepmelt_norm=deepmelt_norm)
    
    ds_2D.to_netcdf(outputpath_melt+'melt_rates_2D_'+mparam+'.nc')
    ds_1D.to_netcdf(outputpath_melt+'melt_rates_1D_'+mparam+'.nc')


Output
^^^^^^

The xr.Dataset ``ds_2D`` contains the following variables on a map (2D):

* ``melt_m_ice_per_s``: melt rate in m ice per second
* ``melt_m_ice_per_y``: melt rate in m ice per year (computed per default but can also be removed by re-defining the list ``options_2D`` given to :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`)
* ``melt_m_we_per_y``: melt rate in m water equivalent per year (computed per default but can also be removed by re-defining the list ``options_2D`` given to :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`)

The xr.Dataset ``ds_1D`` contains the following integrated variables (1D):
* ``melt_m_ice_per_y_tot``: total (accumulated) melt over each ice shelf in m ice per year
* ``melt_m_ice_per_y_avg``: average melt for each ice shelf in m ice per year (computed per default but can also be removed by re-defining the list ``options_1D`` given to :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`)
* ``melt_m_ice_per_y_min``: minimum melt for each ice shelf in m ice per year (computed per default but can also be removed by re-defining the list ``options_1D`` given to :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`)
* ``melt_m_ice_per_y_max``: maximum melt for each ice shelf in m ice per year (computed per default but can also be removed by re-defining the list ``options_1D`` given to :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`)
* ``melt_we_per_y_tot``: total (accumulated) melt over each ice shelf in m water equivalent per year (computed per default but can also be removed by re-defining the list ``options_1D`` given to :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`)
* ``melt_Gt_per_y_tot``: total melt over each ice shelf in Gt per year (computed per default but can also be removed by re-defining the list ``options_1D`` given to :func:`multimelt.melt_functions.calculate_melt_rate_1D_and_2D_all_isf`)
