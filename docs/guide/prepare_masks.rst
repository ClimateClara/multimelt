.. _prod_masks:


multimelt provides three types of "services". (1)It can produce masks of the different circum-Antarctic ice shelves and geometric properties on each ice shelf level (neede for the pdifferent parameterisations). (2) It computes mean profiles of temperature and salinity in front of the ice shelves in given domains of interest. (3) It computes 2D and 1D metrics related to basal melt of ice shelves.

The procedure to create the masks and box and plume characteristics is shown in the notebook ``prepare_mask_example.ipynb``. The steps are also explained more in detail in :ref:`prep_mask_general`, :ref:`prep_box_charac` and :ref:`prep_plume_charac`.

The procedure to compute and format the input temperature (T) and salinity (S) profiles from 2D fields is shown in the notebook ``T_S_profiles_per_ice_shelf.ipynb``. The steps are also explained more in detail in :ref:`prepare_prof`. !BE CAREFUL! If your 2D fields are in conservative temperature and absolute salinity, do not forget to convert them with the script ``conversion_CTtoPT_SAtoSP.ipynb``.

The procedure to compute melt rates from temperature and salinity profiles is shown in the notebook ``compute_melt_example.ipynb``. The steps are also explained more in detail in :ref:`prod_melt`.


How to run (1) : Preparing masks and geometric information
==========================================================

.. _prep_mask_general:

Preparing masks to identify ice shelf and main ice shelf characteristics around Antarctica
------------------------------------------------------------------------------------------

Input data
^^^^^^^^^^

Currently, the mask functions of ``multimelt`` are tailored for **circum-Antarctic** model output from NEMO, the Nucleus for European Modelling of the Ocean :cite:p:`nemo19`, interpolated to a south polar stereographic grid (EPSG:3031). However, it can be used for other models/grids if the variable names are changed accordingly either in the functions or the model's output (requires a bit of digging).


The mask function needs these geometric variables as input:
    
* ``file_msk``: file containing the circum-Antarctic information about basic masks with: 0 = ocean, 1 = ice shelves, 2 = grounded ice
* ``file_bed_orig``: bathymetry [m], positive with depth
* ``file_draft``: ice draft depth [m], positive with depth
* ``file_isf_conc``: ice shelf concentration (if some grid cells are not fully covered with ice)
* ``latlon_boundaries``: the latitude/longitude boundaries of the ice shelves, as defined for example in './mask_info/lonlat_masks.txt'
* ``isf_metadata``: ice shelf name corresponding to ID in file above and data from :cite:t:`rignot13` about the different ice shelves, as shown in './mask_info/iceshelves_metadata_Nico.txt'
* ``file_metadata_GL_flux``: flux across grounding line from :cite:t:`rignot13`, as shown in './mask_info/GL_flux_rignot13.csv'

Running
^^^^^^^

You can create an xr.Dataset containing the main geometric information with the function :func:`multimelt.create_isf_mask_functions.create_mask_and_metadata_isf` as follows:

.. code-block:: python

    import xarray as xr
    import multimelt.create_isf_mask_functions as isfmf
    
    inputpath_metadata = './multimelt/mask_info/'
    outputpath_mask = # path where you want to store your mask netcdf file
    
    file_bed_orig = # xr.DataArray containing the bathymetry (on grid EPSG:3031)
    file_draft = # xr.DataArray containing the actual ice draft depth (not smoothed out through a grid cell mean when the ice concentration is <1)
    file_msk = # xr.DataArray containing mask: 0 = ocean, 1 = ice shelves, 2 = grounded ice (on grid EPSG:3031)
    file_isf_conc = # xr.DataArray containing the ice shelf concentration in each grid cell

    xx = file_msk['x']
    yy = file_msk['y']
    
    whole_ds = isfmf.create_mask_and_metadata_isf(
                        file_msk, # file containing info about the grid (needs to be a domain centered around the South Pole!)                                       
                        -1*file_bed_orig, # negative bathymetry           
                        file_msk, # original mask
                        -1*file_draft, # negative ice draft depth
                        file_isf_conc, # ice shelf concentration
                        False, # not chunked (CAREFUL! chunks not necessarily supported yet)
                        inputpath_metadata+'lonlat_masks.txt', # lon/lat boundaries of the ice shelves
                        outputpath_mask, # output path for output to write out intermediate steps
                        inputpath_metadata + 'iceshelves_metadata_Nico.txt', # file containing name and Rignot data about the different ice shelves
                        ground_point ='no', # if 'yes', the grounding line is defined on the ice shelf points at the border to the ground
                        FRIS_one=True, # do you want to count Filchner-Ronne as one ice shelf? True if yes, False if you want to have them as two separate ice shelves
                        variable_geometry=False, # TO BE USED FOR GEOMETRIES DIFFERENT FROM PRESENT - if True, the ice shelves havee a slightly different geometry than present and the limits have to be changed
                        write_ismask = 'yes', write_groundmask = 'yes', write_outfile='yes', # if you already wrote one of these files, you can set option to 'no'
                        dist=40, # Defines the size of the starting square for the ground mask - should be small if the resolution is coarse and high if the resolution is fine - can be modulated
                        add_fac=120, # Defines additional iterations for the propagation for the ground mask - can be modulated
                        connectivity=4, # if variable_geometry = True:if 8 it looks at all 8 directions to see define neighbouring ice shelf points, if 4 only horizontally and vertically
                        threshold=4, # if variable_geometry = True: an entity of 4 points is considered as one ice shelf
                        write_metadata = 'yes' # writes out the file with only metadata
                        dist=40, # Defines the size of the starting square for the ground mask - should be small if the resolution is coarse and high if the resolution is fine - can be modulated
                        add_fac=120 # Defines additional iterations for the propagation for the ground mask - can be modulated
                                             ) 

    # Write to netcdf
    print('------- WRITE TO NETCDF -----------')
    whole_ds.to_netcdf(outputpath_mask + 'mask_file.nc','w')


Output
^^^^^^

The resulting netcdf file contains the following variables:

* ``ISF_mask``: a map (on x and y) masking the ice shelves (0 for grounded, 1 for ocean, isf ID for ice shelves)
* ``GL_mask``: a map (on x and y) masking the grounding line of the ice shelves (isf ID for grounding line, NaN elsewhere)
* ``IF_mask``: a map (on x and y) masking the ice front of the ice shelves (isf ID for ice front, NaN elsewhere)
* ``PP_mask``: a map (on x and y) masking the pinning points of the ice shelves (isf ID for pinning points, NaN elsewhere)
* ``ground_mask``: a map (on x and y) masking mainland vs islands mask (0 for islands, 1 for ocean and ice shelves, 2 for mainland)
* ``isf_name``: ice shelf name corresponding to ID in ``ISF_mask``
* ``isf_melt``: ice shelf melt as given in :cite:`rignot13` [Gt/yr]
* ``melt_uncertainty``: ice shelf melt uncertainty as given in :cite:`rignot13` [Gt/yr]
* ``isf_area_rignot``: ice shelf area as given in :cite:`rignot13` [km^2]
* ``isf_area_here``: ice shelf area inferred from the input data [km^2]
* ``ratio_isf_areas``: ratio isf area here/Rignot  
* ``front_bot_depth_max``: maximum depth between ice shelf draft and ocean bottom at the ice-shelf front [m] 
* ``front_bot_depth_avg``: average depth between ice shelf draft and ocean bottom at the ice-shelf front [m] 
* ``front_ice_depth_min``: minimum distance between sea surface and ice shelf front depth [m]
* ``front_ice_depth_avg``: average distance between sea surface and ice shelf front depth [m]
* ``front_min_lat``: Minimum latitude of the ice shelf front 
* ``front_max_lat``: Maximum latitude of the ice shelf front 
* ``front_min_lon``: Minimum longitude of the ice shelf front 
* ``front_max_lon``: Maximum longitude of the ice shelf front 
* ``dGL``: Shortest distance to respective grounding line [m]
* ``dIF``: Shortest distance to respective ice front [m]    
* ``dGL_dIF``: Shortest distance to respective ice shelf front (only for grounding line points)

.. _prep_box_charac:

Preparing the box characteristics
---------------------------------

Input data
^^^^^^^^^^

The box and plume characteristics are inferred from the mask file ``'mask_file.nc'`` produced using :func:`multimelt.create_isf_mask_functions.create_mask_and_metadata_isf`. 

.. code-block:: python

    import xarray as xr

    whole_ds = xr.open_dataset(outputpath_mask + 'mask_file.nc')

In the NEMO case, we decide to focus on the ice shelves that are resolved enough on our grid, here the ones larger than 2500 km^2:

.. code-block:: python

    nonnan_Nisf = whole_ds['Nisf'].where(np.isfinite(whole_ds['front_bot_depth_max']), drop=True).astype(int)
    file_isf_nonnan = whole_ds.sel(Nisf=nonnan_Nisf)
    large_isf = file_isf_nonnan['Nisf'].where(file_isf_nonnan['isf_area_here'] >= 2500, drop=True) # only look at ice shelves with area larger than 2500 km2
    file_isf = file_isf_nonnan.sel(Nisf=large_isf)

Running
^^^^^^^

.. code-block:: python

    import xarray as xr
    import multimelt.box_functions as bf

    outputpath_boxes = # path where you want to store your box characteristics netcdf file

    file_draft = # xr.DataArray containing the actual ice draft depth (not smoothed out through a grid cell mean when the ice concentration is <1)
    file_isf_conc = # xr.DataArray containing the ice shelf concentration in each grid cell


    isf_var_of_int = file_isf[['ISF_mask', 'GL_mask', 'dGL', 'dIF', 'latitude', 'longitude', 'isf_name']]
    out_2D, out_1D = bf.box_charac_file(file_isf['Nisf'], # ice shelf ID list
                                        isf_var_of_int, # variables of interest from file_isf
                                        -1*file_draft, # negative ice draft depth
                                        file_isf_conc, # ice shelf concentration
                                        outputpath_boxes, # output path for netcdfs
                                        max_nb_box=10 # maximum amount of boxes to explore
                                        )

    print('------ WRITE TO NETCDF -------')
    out_2D.to_netcdf(outputpath_boxes + 'boxes_2D.nc')
    out_1D.to_netcdf(outputpath_boxes + 'boxes_1D.nc')

Output
^^^^^^

The resulting netcdf file ``boxes_2D.nc`` contains the following variables:

* ``dGL``: map (on x and y) of shortest distance to respective grounding line [m]
* ``dIF``: map (on x and y) of shortest distance to respective ice front [m] 
* ``box_location``: map (on x and y) masking the location of box 1 to n, depending on the amount of boxes

The resulting netcdf file ``boxes_1D.nc`` contains the following variables:

* ``box_area``: area of the respective box [m^2]
* ``box_depth_below_surface``: mean depth at the top of the box [m]
* ``nD_config``: amount of boxes that can be used in the config levels, according to the criteria that all boxes should have an area of more than 0 and that the box depth below surface has an ascending slope from grounding line to ice front. 

.. _prep_plume_charac:

Preparing the plume characteristics
-----------------------------------

Input data
^^^^^^^^^^

The box and plume characteristics are inferred from the mask file ``'mask_file.nc'`` produced using :func:`multimelt.create_isf_mask_functions.create_mask_and_metadata_isf`. 

.. code-block:: python

    import xarray as xr

    whole_ds = xr.open_dataset(outputpath_mask + 'mask_file.nc')

In the NEMO case, we decide to focus on the ice shelves that are resolved enough on our grid, here the ones larger than 2500 km^2:

.. code-block:: python

    nonnan_Nisf = whole_ds['Nisf'].where(np.isfinite(whole_ds['front_bot_depth_max']), drop=True).astype(int)
    file_isf_nonnan = whole_ds.sel(Nisf=nonnan_Nisf)
    large_isf = file_isf_nonnan['Nisf'].where(file_isf_nonnan['isf_area_here'] >= 2500, drop=True) # only look at ice shelves with area larger than 2500 km2
    file_isf = file_isf_nonnan.sel(Nisf=large_isf)

Running
^^^^^^^

.. code-block:: python

    import xarray as xr
    import multimelt.plume_functions as pf

    plume_param_options = ['cavity','lazero', 'local'] 
    # 'cavity': deepest grounding line, cavity slope
    # 'lazero': grounding line and slope inferred according to Lazeroms et al., 2018
    # 'local': grounding line inferred according to Lazeroms et al., 2018 and local slope

    plume_var_of_int = file_isf[['ISF_mask', 'GL_mask', 'IF_mask', 'dIF', 'dGL_dIF', 'latitude', 'longitude', 'front_ice_depth_avg']]

    # Compute the ice draft
    file_draft = # xr.DataArray containing the actual ice draft depth (not smoothed out through a grid cell mean when the ice concentration is <1)
    ice_draft_pos = file_draft
    ice_draft_neg = -1*ice_draft_pos

    plume_charac = pf.prepare_plume_charac(plume_param_options, 
                                            ice_draft_pos,
                                            plume_var_of_int
                                            )

    print('------ WRITE TO NETCDF -------')
    plume_charac.to_netcdf(outputpath_plumes+'plume_characteristics.nc') 

Output
^^^^^^

The resulting netcdf file ``plume_characteristics.nc`` contains the following variables:  

* ``zGL``: map (on x and y) of grounding line depth (negative downwards) associated to each ice shelf point [m]
* ``alpha``: map (on x and y) of slope associated to each ice shelf point
    



