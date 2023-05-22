from scipy.spatial import cKDTree
from tqdm.notebook import tqdm, trange
import xarray as xr
import numpy as np
import multimelt.useful_functions as uf


# find shortest distance of isf_points to line
def distance_isf_points_from_line(whole_domain,isf_points_da,line_points_da):
    
    """
    Compute the distance between ice shelf points and a line.
    
    This function computes the distance between ice shelf points and a line. This line can be the grounding
    line or the ice shelf front.
    
    Parameters
    ----------
    whole_domain : xarray.DataArray
        ice-shelf mask - all ice shelves are represented by a number, all other points (ocean, land) set to nan
    isf_points_da : xarray.DataArray
        array containing only points from one ice shelf
    line_points_da : xarray.DataArray
        mask representing the grounding line or ice shelf front mask corresponding to the ice shelf selected in ``isf_points_da``
        
    Returns
    -------
    xr_dist_to_line : xarray.DataArray
        distance of the each ice shelf point to the given line of interest
    """
    
    # add a common dimension 'grid' along which to stack
    stacked_isf_points = isf_points_da.stack(grid=['y', 'x'])
    stacked_line = line_points_da.stack(grid=['y', 'x'])
    
    # remove nans
    filtered_isf_points = stacked_isf_points[stacked_isf_points>0]
    #filtered_isf_points = stacked_isf_points.where(stacked_isf_points>0, drop=True)
    #filtered_line = stacked_line.where(stacked_line>0, drop=True)
    filtered_line = stacked_line[stacked_line>0]
    
    # write out the y,x pairs behind the dimension 'grid'
    grid_isf_points = filtered_isf_points.indexes['grid'].to_frame().values.astype(float)
    grid_line = filtered_line.indexes['grid'].to_frame().values.astype(float)
    
    # create tree to line and compute distance
    tree_line = cKDTree(grid_line)
    dist_yx_to_line, _ = tree_line.query(grid_isf_points)
        
    # add the coordinates of the previous variables
    xr_dist_to_line = filtered_isf_points.copy(data=dist_yx_to_line)
    # put 1D array back into the format of the grid and put away the 'grid' dimension
    xr_dist_to_line = xr_dist_to_line.unstack('grid')
    # choose to reindex on initial grid (whole domain)
    xr_dist_to_line = xr_dist_to_line.reindex_like(whole_domain)
    
    return xr_dist_to_line

def check_boxes(box_characteristics,upper_bnd):
    
    """
    Check different box configurations.
    
    This sets the number of boxes that can be applied to the ice shelf cavity depending on ``upper_bnd``
    
    Parameters
    ----------
    box_characteristics : xarray.Dataset
        dataset containing 'box_area_tot' and 'box_depth_below_surface' for a given ice shelf
    upper_bnd : int
        maximum of box number wanted for that configuration

    Returns
    -------
    nD : int
        closest number of boxes to ``upper_bnd`` possible given the box characteristics
    """
    
    subset_a = box_characteristics.sel(box_nb_tot=range(1,upper_bnd+1))
    subset_b = box_characteristics.sel(box_nb_tot=subset_a['box_nb_tot'][1:])
    nD_zero_area = subset_a['box_area_tot'].where(subset_a['box_area_tot']>0)
    nD_area = nD_zero_area['box_nb_tot'].where((nD_zero_area.count('box_nb')/nD_zero_area['box_nb_tot'])==1).max() # if one of the boxes is nan we don't want to keep this number
    nD_diff_slope = subset_b['box_depth_below_surface'].where(subset_b['box_depth_below_surface'] - subset_b['box_depth_below_surface'].shift(box_nb=-1)<=0)
    nD_slope = nD_diff_slope['box_nb_tot'].where((nD_diff_slope.count('box_nb')/(nD_diff_slope['box_nb_tot']-1))==1).max() # if one of the boxes is nan we don't want to keep this number
    nD = int(min(nD_area,nD_slope).values)
    return nD


def prepare_box_charac(kisf, isf_var_of_int, ice_draft, isf_conc, dx, dy, max_nb_box=10):
    
    """
    Prepare file with the box characteristics for one ice shelf.
    
    This function prepares the Dataset containing the relevant geometric information to compute the melt with the box model for an ice shelf of interest.
    
    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    isf_var_of_int : xarray.Dataset
        Dataset containing the variables ``ISF_mask`` with the coordinates ``['x','y']`` and ``isf_name``
    ice_draft : xarray.DataArray
        Ice draft in m. The value must be negative under water.
    isf_conc : floats between 0 and 1
        Concentration of ice shelf in each grid cell
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
     max_nb_box : int
         Amount of boxes in the configuration with the most boxes
    
    Returns
    -------
    ds_2D : xarray.Dataset
        Dataset containing the following variables for the ice shelf of interest: ``dGL``, ``dIF``, ``box_location``
    ds_1D : xarray.Dataset
        Dataset containing the following variables for the ice shelf of interest:``box_area``, ``box_depth_below_surface``, ``nD_config``
    """
    # ice_draft needs to be negative
    
    isf_var_of_int = isf_var_of_int.chunk({'x': 2000, 'y': 2000})
    
    # cut out domain
    isf_area = isf_var_of_int.where(isf_var_of_int['ISF_mask'] == kisf).dropna('x', how='all').dropna('y', how='all')
    ice_draft = ice_draft.sel(x=isf_area.x, y=isf_area.y).where(isf_area['ISF_mask'] == kisf)
    isf_conc_here = isf_conc.sel(x=isf_area.x, y=isf_area.y).where(isf_area['ISF_mask'] == kisf)

    # dataset to save all info (initialized for each ice shelf)
    box_charac = xr.Dataset({'box_location': (['y','x', 'box_nb_tot'], np.zeros((len(isf_area.y),len(isf_area.x), max_nb_box))),
                            'box_area_tot': (['box_nb', 'box_nb_tot'], np.zeros((max_nb_box, max_nb_box))),
                             'box_depth_below_surface': (['box_nb', 'box_nb_tot'], np.zeros((max_nb_box, max_nb_box))),
                             },
                            coords={'y': isf_area.y, 'x': isf_area.x, 'box_nb_tot': range(1, max_nb_box + 1), 'box_nb': range(1,max_nb_box + 1)})

    # relative distance to the grounding line
    r = isf_area['dGL'] / (isf_area['dGL'] + isf_area['dIF'])

    # prepare the dimensions for the cutting of the boxes
    mesh_box_tot, mesh_bbox = xr.broadcast(box_charac['box_nb_tot'], box_charac['box_nb'])

    # conditions how to cut boxes
    difference = mesh_box_tot - mesh_bbox
    # difference = mesh_box_tot-mesh_bbox
    box_charac['lower_bnd'] = 1 - np.sqrt((difference.where(difference >= 0) + 1) / mesh_box_tot)
    box_charac['upper_bnd'] = 1 - np.sqrt((difference.where(difference >= 0)) / mesh_box_tot)

    # geographic location of the boxes, depending on box_nb_tot
    box_charac['box_location'] = box_charac['box_nb'].where((r >= box_charac['lower_bnd']) & (r <= box_charac['upper_bnd'])).min('box_nb')
    box_charac['box_location'] = box_charac['box_location'].where(box_charac['box_location']>0)

    # area of the boxes, depending on box_nb_tot
    box_nb = xr.DataArray(data=np.arange(1,11), dims='box_nb')
    box_charac['box_area_tot'] = isf_conc_here.where(box_charac['box_location'] == box_nb).sum(dim=['x','y']) * abs(dx) * abs(dy)

    if box_charac['box_area_tot'].sum() > 0:

        # mean depth of the boxes, depending on box_nb_tot
        box_charac['box_depth_below_surface'] = uf.weighted_mean(ice_draft.where(box_charac['box_location'] == box_nb), ['x','y'], isf_conc_here.where(box_charac['box_location'] == box_nb))

        nD3 = check_boxes(box_charac, max_nb_box)
        if nD3 < 3:
            nD2 = 1
        else:
            nD2 = check_boxes(box_charac, min(nD3 - 1, max_nb_box // 2))
        if nD2 < 3:
            nD1 = 1
        else:
            nD1 = check_boxes(box_charac, min(nD2 - 1, 2))
                
        nD = np.array([nD3, nD2, nD1])
        
        ds_2D = xr.Dataset(
            {'dGL': (isf_area['dGL'].dims, isf_area['dGL'].values),
             'dIF': (isf_area['dIF'].dims, isf_area['dIF'].values),
             'box_location': (box_charac['box_location'].dims, box_charac['box_location'].values)
         },
        coords={'y': isf_area.y, 'x': isf_area.x, 'box_nb_tot': range(1, max_nb_box + 1), 'box_nb': range(1, max_nb_box + 1),
                'latitude': isf_area['latitude'],
                'longitude': isf_area['longitude'], 'config': range(3, 0, -1)})
            
        ds_1D = xr.Dataset(
            {'box_area': (box_charac['box_area_tot'].dims, box_charac['box_area_tot'].values),
             'box_depth_below_surface': (box_charac['box_depth_below_surface'].dims, box_charac['box_depth_below_surface'].values),
             'nD_config': (['config'], nD),
         },
        coords={'box_nb_tot': range(1, max_nb_box + 1), 'box_nb': range(1, max_nb_box + 1),'config': range(3, 0, -1)})

        return ds_2D, ds_1D
    
def box_nb_like_reese(Nisf_list,dGL_all, dGL_max, n_max, file_isf):
    
    """
    Compute the number of boxes for each ice shelf according to the criterion given in Eq. 9 in Reese et al. 2018.
    
    
    Parameters
    ----------
    Nisf_list: array of int
        List containing the ice shelf IDs for all ice shelves of interest. 
    dGL_all: array
        Distance between each point and the closest grounding line.
    dGL_max: float
        Maximum distance between a point and the closest grounding line in the whole domain. 
    n_max: int
        Maximum possible amount of boxes.
    file_isf: xr.Dataset
        mask_file containing ``ISF_mask``.
    

    Returns
    -------
    nD_all: xr.DataArray
        Corresponding amount of boxes for each ice shelf of Nisf_list.
    """
    
    d_max = dGL_max
    nD_list = []
    for kisf in Nisf_list:
        dGL_max = dGL_all.where(file_isf['ISF_mask']==kisf,drop=True).max()
        nD = 1+np.round(np.sqrt(dGL_max/d_max)*(n_max-1))
        nD_list.append(nD.astype(int))
    
    nD_all = xr.concat(nD_list, dim='Nisf')
    return nD_all

def loop_box_charac(Nisf_list,isf_var_of_int, ice_draft, isf_conc, outputpath_boxes, max_nb_box=10):
    
    """
    Create xarray datasets with the box characteristics for all ice shelves of interest.
    
    This function combines the box characteristics of all individual ice shelves into one dataset.

    Parameters
    ----------
    Nisf_list : list of int
        List of IDs of the ice shelves of interest
    isf_var_of_int : xarray.Dataset
        Dataset containing the variables ``ISF_mask`` with the coordinates ``['x','y']``
    ice_draft : xarray.DataArray
        Ice draft in m. The value must be negative under water.
    isf_conc : floats between 0 and 1
        Concentration of ice shelf in each grid cell
    outputpath_boxes : str
        Path to the fodler where the individual files should be written to.
     max_nb_box : int
         Amount of boxes in the configuration with the most boxes. Default is 10.
    
    Returns
    -------
    out_ds_2D : xarray.Dataset
        Dataset containing the following variables for all ice shelves: ``dGL``, ``dIF``, ``box_location``
    out_ds_1D : xarray.Dataset
        Dataset containing the following variables for all ice shelves:``box_area``, ``box_depth_below_surface``, ``nD_config``
    
    """
    
    
    dx = isf_var_of_int.x[2] - isf_var_of_int.x[1]
    dy = isf_var_of_int.y[2] - isf_var_of_int.y[1]
    
    n = 0
    ds_1D_all = [ ]
    nisf_list_ok = [ ]
    dGL_max_list_ok = [ ]
    # Calculate total area and mean depth of each box for configurations of total nb of boxes from 1 to 10 :
    for kisf in tqdm(Nisf_list):
        
        if ~np.isnan(isf_var_of_int['GL_mask'].where(isf_var_of_int['GL_mask'] == kisf).max()):
            ds_2D, ds_1D = prepare_box_charac(kisf, isf_var_of_int, ice_draft, isf_conc, dx, dy, max_nb_box=10)

            if n == 0:
                ds_box_charac_2D = ds_2D
            else:
                ds_box_charac_2D = ds_box_charac_2D.combine_first(ds_2D)
            n = n+1
            
            ds_1D_all.append(ds_1D)
            nisf_list_ok.append(kisf)
            dGL_max_list_ok.append(isf_var_of_int['dGL'].where(isf_var_of_int['ISF_mask']==kisf).max())
            
    out_ds_2D = ds_box_charac_2D.reindex_like(isf_var_of_int['ISF_mask'])
    out_ds_1D = xr.concat(ds_1D_all, dim='Nisf')
    out_ds_1D = out_ds_1D.assign_coords({'Nisf': nisf_list_ok})
    
    dGL_max_ok = xr.concat(dGL_max_list_ok, dim='Nisf')
    dGL_max_all = dGL_max_ok.max()

    # add Reese box config
    box_config_4 = box_nb_like_reese(Nisf_list,out_ds_2D['dGL'],dGL_max_all,5,isf_var_of_int).rename('nD_config')
    box_config_4_da = xr.DataArray(data=box_config_4, dims=['Nisf'])
    box_config_4_da = box_config_4_da.assign_coords({'Nisf': box_config_4.Nisf, 'config': 4})
    
    new_config = xr.concat([out_ds_1D['nD_config'],box_config_4_da], dim='config').astype(int)
    new_box_1D = out_ds_1D[['box_area','box_depth_below_surface']].merge(new_config)
    
    return out_ds_2D, new_box_1D
        
            #if outfile:
                #outfile.to_netcdf(outputpath_boxes + name_file + '_box_isf' + '{0:03}'.format(kisf.values) + '.nc', 'w')
            
def box_charac_file(Nisf_list,isf_var_of_int, ice_draft, isf_conc, outputpath_boxes, max_nb_box=10):
    
    """
    Compute box characteristics for all ice shelves of interest and prepare dataset to be written to netcdf (with all attributes).
    
    This function writes two files containting the 1D and 2D relevant geometric information to compute the melt with the box model for all ice shelves of interest.

    Parameters
    ----------
    Nisf_list : list of int
        List of IDs of the ice shelves of interest
    isf_var_of_int : xarray.Dataset
        Dataset containing the variables ``GL_mask`` and ``ISF_mask`` with the coordinates ``['x','y']``
    ice_draft : xarray.DataArray
        Ice draft in m. The value must be negative under water.
    isf_conc : floats between 0 and 1
        Concentration of ice shelf in each grid cell
    outputpath_boxes : str
        Path to the fodler where the individual files should be written to.
     max_nb_box : int
         Amount of boxes in the configuration with the most boxes. Default is 10.
    
    Returns
    -------
    ds_2D : xarray.Dataset
        Dataset containing the following variables for all ice shelves: ``dGL``, ``dIF``, ``box_location``
    ds_1D : xarray.Dataset
        Dataset containing the following variables for all ice shelves:``box_area``, ``box_depth_below_surface``, ``nD_config``
        
    
    """
    
    ds_2D, ds_1D = loop_box_charac(Nisf_list,isf_var_of_int, ice_draft, isf_conc, outputpath_boxes, max_nb_box=10)

    ds_2D['dGL'].attrs['units'] = 'm'
    ds_2D['dGL'].attrs['long_name'] = 'Shortest distance to respective grounding line'
    ds_2D['dGL'].attrs['standard_name'] = 'distance_to_grounding_line'
    ds_2D['dIF'].attrs['units'] = 'm'
    ds_2D['dIF'].attrs['long_name'] = 'Shortest distance to respective ice shelf front'
    ds_2D['dIF'].attrs['standard_name'] = 'distance_to_isf_front'
    ds_2D['box_location'].attrs['long_name'] = 'Location of the different boxes'
    ds_2D['box_location'].attrs['standard_name'] = 'box_location'


    # Global attributes
    ds_2D.attrs['history'] = 'Created with box_functions.py by C. Burgard'
    ds_2D.attrs['projection'] = 'Polar Stereographic South (71S,0E)'
    ds_2D.attrs['proj4'] = '+init=epsg:3031'
    ds_2D.attrs[
        'Note'] = 'isf ID can be found in *_isf_masks_and_info_and_distance.nc'
    
    ds_1D['box_depth_below_surface'].attrs['units'] = 'm'
    ds_1D['box_depth_below_surface'].attrs['long_name'] = 'Mean depth at which the box starts'
    ds_1D['box_depth_below_surface'].attrs['standard_name'] = 'box_depth_below_surface'
    ds_1D['nD_config'].attrs['long_name'] = 'Different amount of boxes that can be used'
    ds_1D['nD_config'].attrs['standard_name'] = 'box_amount'
    ds_1D['box_area'].attrs['units'] = 'm**2'
    ds_1D['box_area'].attrs['long_name'] = 'Area of the boxes'
    ds_1D['box_area'].attrs['standard_name'] = 'box_area'
            
            
    ds_1D['config'].attrs['standard_name'] = 'box_config_levels'
    ds_1D['box_nb_tot'].attrs['standard_name'] = 'total_box_number'
    ds_1D['box_nb_tot'].attrs['long_name'] = 'Total amount of boxes'
    ds_1D['box_nb'].attrs['standard_name'] = 'box_number'
    ds_1D['box_nb'].attrs['long_name'] = 'Box number within a total amount of boxes'
            
    # Global attributes
    ds_1D.attrs['history'] = 'Created with box_functions.py by C. Burgard'
    ds_1D.attrs['projection'] = 'Polar Stereographic South (71S,0E)'
    ds_1D.attrs['proj4'] = '+init=epsg:3031'
    ds_1D.attrs[
        'Note'] = 'isf ID can be found in *_isf_masks_and_info_and_distance.nc'
        
    return ds_2D, ds_1D

