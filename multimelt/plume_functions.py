import xarray as xr
import scipy.ndimage
import scipy.signal
import numpy as np
from tqdm.notebook import tqdm

def nd_corr_sig(input1,weights):
    
    """
    Correlation using a filter (scipy.signal library).
        
    Parameters
    ----------
    input1 : xr.DataArray
        Array to be filtered.
    weights : xr.DataArray
        Filter to be applied.

    Returns
    -------
    Filtered data (sum according to the weights)
    
    """
    
    return scipy.signal.correlate(input1, weights, method='auto', mode='same')

def nd_corr(input1,weights):
    
    """
    Correlation using a filter (scipy.ndimage library).
        
    Parameters
    ----------
    input1 : xr.DataArray
        Array to be filtered.
    weights : xr.DataArray
        Filter to be applied.

    Returns
    -------
    Filtered data (sum according to the weights)
    
    """
    
    return scipy.ndimage.correlate(input1, weights, mode='constant')

def xr_nd_corr_sig(data,weights):
    
    """
    Correlation using a filter (scipy.signal library).
        
    Parameters
    ----------
    data : xr.DataArray
        Array to be filtered with at least dimensions ['y','x'].
    weights : xr.DataArray
        Filter to be applied with at least dimensions ['y0','x0'].

    Returns
    -------
    Filtered data (sum according to the weights)
    
    """
    
    return xr.apply_ufunc(nd_corr_sig,
                          data,
                          weights,
                          input_core_dims=[['y','x'],['y0','x0']], 
                          output_core_dims=[['y','x']], 
                          vectorize=True,
                          dask='parallelized')

def xr_nd_corr(data,weights):

    """
    Correlation using a filter (scipy.ndimage library).
        
    Parameters
    ----------
    data : xr.DataArray
        Array to be filtered with at least dimensions ['y','x'].
    weights : xr.DataArray
        Filter to be applied with at least dimensions ['y0','x0'].

    Returns
    -------
    Filtered data (sum according to the weights)
    
    """
    
    return xr.apply_ufunc(nd_corr,
                          data,
                          weights,
                          input_core_dims=[['y','x'],['y0','x0']], 
                          output_core_dims=[['y','x']], 
                          vectorize=True,
                          dask='parallelized')

def xr_nd_corr_v2(data,weights):

    """
    Correlation using a filter (scipy.ndimage library).
        
    Parameters
    ----------
    data : xr.DataArray
        Array to be filtered with at least dimensions ['y','x'].
    weights : xr.DataArray
        Filter to be applied with at least dimensions ['y0','x0'].

    Returns
    -------
    Filtered data (sum according to the weights)
    
    """
    
    return xr.apply_ufunc(nd_corr,
                          data,
                          weights,
                          input_core_dims=[['y','x'],['y0','x0']], 
                          output_core_dims=[['y','x']], 
                          dask='parallelized')

def check_slope_one_dimension(input_da, shifted_plus, shifted_minus, dx):

    """
    Compute the basal slope at each point.
        
    Parameters
    ----------
    input_da : xr.DataArray
        Array where slope needs to be checked. For example: ice draft.
    shifted_plus : xr.DataArray
        Shifted version (positive direction) of input_da.
    shifted_minus : xr.DataArray
        Shifted version (negative direction) of input_da.
    dx : float
        Step in the coordinate along which input_da was shifted

    Returns
    -------
    slope: xr.DataArray
        slope along that coordinate, is 0 if nan
    """
    
    # check the direction in both dim directions
    slope_both = (shifted_minus - shifted_plus) / np.sqrt((2 * dx) ** 2)
    # if x+1 is nan, only take x - (x-1)
    slope_right = (input_da - shifted_plus) / np.sqrt(dx ** 2)
    # if x-1 is nan, only take x+1 - x
    slope_left = (shifted_minus - input_da) / np.sqrt(dx ** 2)
    # combine all of the above
    slope = slope_both.combine_first(slope_right).combine_first(slope_left)
    # set rest to 0
    slope = slope.where(np.isfinite(slope), 0)
    return slope


def add_GL_max(plume_var_of_int, ice_draft_pos):

    """
    Add the information about the deepest grounding line point to the info-dataset.
        
    Parameters
    ----------
    plume_var_of_int : xr.Dataset
        Dataset containing at least ``'GL_mask'`` and ``'dGL_dIF'``.
    ice_draft_pos : xr.DataArray
        Ice draft depth in m. Positive downwards.

    Returns
    -------
    plume_var_of_int : xr.Dataset
        Dataset extended with ``'GL_max'`` and ``'dGL_max_dIF'``.
    """
    
    # Compute maximum grounding line depth and distance between max grounding line depth and ice front
    plume_var_of_int['GL_max'] = xr.DataArray(data=np.zeros(len(plume_var_of_int['Nisf'])) * np.nan, dims=['Nisf'],
                                              coords={'Nisf': plume_var_of_int['Nisf'].values})
    plume_var_of_int['dGL_max_dIF'] = xr.DataArray(data=np.zeros(len(plume_var_of_int['Nisf'])) * np.nan, dims=['Nisf'],
                                                  coords={'Nisf': plume_var_of_int['Nisf'].values})
    for kisf in tqdm(plume_var_of_int['Nisf']):

        if ~np.isnan(plume_var_of_int['GL_mask'].where(plume_var_of_int['GL_mask'] == kisf).max()):

            GL_thick = ice_draft_pos.where(plume_var_of_int['GL_mask'] == kisf)
            # maximum grounding line depth
            plume_var_of_int['GL_max'].loc[dict(Nisf=kisf)] = GL_thick.max(dim=['x', 'y'])
            # shortest distance between max grounding line depth and ice front (assuming there is only one point where maximum depth is reached, if not: mean over all of them)
            plume_var_of_int['dGL_max_dIF'].loc[dict(Nisf=kisf)] = plume_var_of_int['dGL_dIF'].where(GL_thick == plume_var_of_int['GL_max'].sel(Nisf=kisf), drop=True).mean().squeeze().values
    
    return plume_var_of_int

def prepare_plume_dataset(plume_var_of_int,plume_param_options):

    """
    Prepare plume dataset to include zGL and alpha.
    
    This initializes the grounding line depth and the angle at the origin of the plume.
        
    Parameters
    ----------
    plume_var_of_int : xr.DataArray or xr.Dataset
        Dataset or DataArray representing the domain.
    plume_param_options : list of str
        Parametrization options (typically 'cavity', 'lazero' and 'local').

    Returns
    -------
    plume_var_of_int : xr.Dataset
        Dataset extended with ``'alpha'`` and ``'zGL'``.
    """
    
    # initialization
    plume_var_of_int['alpha'] = xr.DataArray(data=np.zeros((len(plume_var_of_int.y), len(plume_var_of_int.x), len(plume_param_options))), dims=['y', 'x', 'option'],
                                             coords={'y': plume_var_of_int.y, 'x': plume_var_of_int.x, 'option': plume_param_options})
    plume_var_of_int['zGL'] = xr.DataArray(data=np.zeros((len(plume_var_of_int.y), len(plume_var_of_int.x), len(plume_param_options))), dims=['y', 'x', 'option'],
                                           coords={'y': plume_var_of_int.y, 'x': plume_var_of_int.x, 'option': plume_param_options})
    return plume_var_of_int

def compute_alpha_cavity(plume_var_of_int):

    """
    Compute alpha with a very simple approach (angle between horizontal and ice front) => cavity slope
        
    Parameters
    ----------
    plume_var_of_int : xr.DataArray or xr.Dataset
        Dataset containing ``'GL_max'``, ``'front_ice_depth_avg'`` and ``'dGL_max_dIF'``

    Returns
    -------
    alphas : xr.DataArray
        Angle at origin of plume in rad.
    """
    
    # compute angle using arctan
    alphas = np.arctan(
        (plume_var_of_int['GL_max'] - plume_var_of_int['front_ice_depth_avg']) / plume_var_of_int['dGL_max_dIF'])
 
    return alphas

def create_16_dir_weights():
    
    """
    Prepare correlation filter in 16 directions.
    
    This function prepares the correlation filter in 16 directions, following the methodology in Lazeroms et al;, 2018.

    Returns
    -------
    ds_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions.
    """
    
    #print('prepare correlation mask to check the 16 defined directions')
    #prepare correlation mask to check the 16 defined directions (1 in the middle and -1 in all interesting directions)
    dir_x = np.arange(-2, 3)
    dir_y = np.arange(-2, 3)

    weights_gradients = np.zeros((5, 5, 16))
    weights_gradients[2, 2, :] = 1
    layer_x = []
    layer_y = []

    n = 0
    for shift_x in dir_x:
        for shift_y in dir_y:

            cond_0 = (shift_x == 0 and shift_y == 0) | (shift_x == 0 and abs(shift_y) == 2) | (
                        abs(shift_x) == 2 and shift_y == 0) | (abs(shift_x) == 2 and abs(shift_y) == 2)

            if not cond_0:
                layer_x.append(shift_x)
                layer_y.append(shift_y)
                weights_gradients[2 + shift_y, 2 + shift_x, n] = -1 # y first, x then because of the initial grid being like this
                n = n + 1

    # make a dataset with it to remember the associated shift_x and shift_y
    xr_weights = xr.DataArray(data=weights_gradients, dims=['y0', 'x0', 'direction'], coords={'direction': np.arange(16)})
    ds_weights = xr.Dataset(data_vars={'weights': xr_weights})
    ds_weights['shift_x'] = xr.DataArray(data=np.array(layer_x), dims=['direction'])
    ds_weights['shift_y'] = xr.DataArray(data=np.array(layer_y), dims=['direction'])
    return ds_weights

def create_4_dir_weights():
    
    """
    Prepare correlation filter in 8 directions.
    
    Returns
    -------
    ds_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions.
    """
    
    #print('prepare correlation mask to check the 16 defined directions')
    #prepare correlation mask to check the 16 defined directions (1 in the middle and -1 in all interesting directions)
    dir_x = np.arange(-1, 2)
    dir_y = np.arange(-1, 2)

    weights_gradients = np.zeros((3, 3, 4))
    weights_gradients[1, 1, :] = 1
    layer_x = []
    layer_y = []

    n = 0
    for shift_x in dir_x:
        for shift_y in dir_y:

            cond_0 = (shift_x == 0 and shift_y == 0) | (shift_x == 1 and shift_y == 1) | (shift_x == -1 and shift_y == -1) | (shift_x == 1 and shift_y == -1) | (shift_x == -1 and shift_y == 1)

            if not cond_0:

                layer_x.append(shift_x)
                layer_y.append(shift_y)
                weights_gradients[1 + shift_y, 1 + shift_x, n] = -1 # y first, x then because of the initial grid being like this
                n = n + 1

    # make a dataset with it to remember the associated shift_x and shift_y
    xr_weights = xr.DataArray(data=weights_gradients, dims=['y0', 'x0', 'direction'], coords={'direction': np.arange(4)})
    ds_weights = xr.Dataset(data_vars={'weights': xr_weights})
    ds_weights['shift_x'] = xr.DataArray(data=np.array(layer_x), dims=['direction'])
    ds_weights['shift_y'] = xr.DataArray(data=np.array(layer_y), dims=['direction'])
    return ds_weights


def first_criterion_lazero(kisf, plume_var_of_int, ice_draft_neg_isf, isf_and_GL_mask, ds_weights, dx, dy):

    """
    Define first criterion for the plume parameters.
    
    This function computes the basal slope and identifies the first criterion, following the methodology in Lazeroms et al;, 2018.

    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` and ``'GL_mask'``
    ice_draft_neg_isf : xr.DataArray
        Ice draft depth for the given ice shelf in m. Negative downwards.
    isf_and_GL_mask : xr.DataArray
        Mask of the domain covered by the ice shelf and the grounding line (this extra mask is needed if the grounding line is defined on ground points)
    ds_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions.
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
        
    Returns
    -------
    GL_depth : xr.DataArray
        Depth of the grounding line points (negative downwards).
    sn_isf : xr.DataArray
        Basal slope in all 16 directions
    first_crit : xr.DataArray
        Boolean where sn_sf > 0
    draft_depth : 
        Ice draft depth in m (negative downwards) extended through the 'direction' dimension.
    """
    
    # add dimension for directions to the ice_draft array
    other = xr.DataArray(np.zeros(16), coords=[('direction', np.arange(16))])
    ice_draft_neg_dirs, other2 = xr.broadcast(ice_draft_neg_isf, other)

    # draft depth only on the ice shelf
    draft_depth = ice_draft_neg_dirs.where(isf_and_GL_mask).where((plume_var_of_int['ISF_mask'] == kisf))

    # grounding line depth only where grounding line
    GL_depth = ice_draft_neg_dirs.where(isf_and_GL_mask).where(plume_var_of_int['GL_mask'] == kisf)
    GL_depth = GL_depth.where(GL_depth < 0, 0)

    # apply the correlation filter to compute gradients in the 16 directions (xr_nd_corr_sig does not work for whatever reason :( ))
    gradients = xr_nd_corr(draft_depth, ds_weights['weights'])

    # compute the sn - basal slope - to be consistent with the origin of the plumes, we cut the basal slopes after ice shelves as well - but might need to think about what happens when several ice shelves are touching each other
    sn_isf = gradients / np.sqrt((ds_weights['shift_x'] * np.abs(dx)) ** 2 + (ds_weights['shift_y'] * np.abs(dy)) ** 2)
    # 1st criterion: sn > 0
    first_crit = sn_isf>0

    return GL_depth, sn_isf, first_crit, draft_depth

def prepare_filter_16dir_isf(isf_and_GL_mask, GL_depth, x_size, y_size, ds_weights):

    """
    Prepare the filter to check grounding line in all 16 directions.
    
    This function computes the basal slope and identifies the first criterion, following the methodology in Lazeroms et al;, 2018.

    Parameters
    ----------
    isf_and_GL_mask : xr.DataArray
        Mask of the domain covered by the ice shelf and the grounding line (this extra mask is needed if the grounding line is defined on ground points)
    GL_depth : xr.DataArray
        Depth of the grounding line points (negative downwards).
    x_size : int
        Size of the domain in the x-coordinate.
    y_size : int
        Size of the domain in the y-coordinate
    ds_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions.

    Returns
    -------
    ds_isf_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions adapted to the given ice shelf domain.
    mid_coord : int
        Maximum dimension of the domain and therefore middle of the filter.
    """
        
    # prepare the filter 
    mid_coord = max(x_size, y_size)

    # double the size of the domain so that the search works also for extreme points
    weights_gline_depth = np.zeros((mid_coord * 2 + 1, mid_coord * 2 + 1, 16))
    weights_gline_gradient = np.zeros((mid_coord * 2 + 1, mid_coord * 2 + 1, 16))

    for dd in GL_depth['direction']:
        # enlarge the filter with each iteration
        for n in range(1, mid_coord + 1):
            # we can only go as far as mid_coord in all directions
            if (abs(n * ds_weights['shift_y'].sel(direction=dd)) <= mid_coord) and (abs(n * ds_weights['shift_x'].sel(direction=dd)) <= mid_coord):
                weights_gline_depth[mid_coord + n * ds_weights['shift_y'].sel(direction=dd), mid_coord + n * ds_weights['shift_x'].sel(direction=dd), dd] = 1

    # write filter to dataset to keep track of related shift_x and shift_y
    xr_weights_gl_depth = xr.DataArray(data=weights_gline_depth, dims=['y0', 'x0', 'direction'],
                                       coords={'direction': np.arange(16)})
    ds_isf_weights = xr.Dataset(data_vars={'weights_gl_depth': xr_weights_gl_depth})
    ds_isf_weights['shift_x'] = ds_weights['shift_x']
    ds_isf_weights['shift_y'] = ds_weights['shift_y']
    return ds_isf_weights, mid_coord

def second_criterion_lazero(kisf, plume_var_of_int, ds_isf_weights,GL_depth, mid_coord, draft_depth):

    """
    Define second criterion for the plume parameters.
    
    This function looks for the grounding line depth at the plume origin and identifies the second criterion, following the methodology in Lazeroms et al., 2018.

    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'``
    ds_isf_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions adapted to the given ice shelf domain.
    GL_depth : xr.DataArray
        Depth of the grounding line points (negative downwards).
    mid_coord : int
        Maximum dimension of the domain and therefore middle of the filter.
    draft_depth : 
        Ice draft depth in m (negative downwards) extended through the 'direction' dimension.
        
    Returns
    -------
    ds_isf_lazeroms: xr.Dataset
        Dataset containing the grounding line depth at the origin of the plume (``'gl_depth'``) and the gradient between the ice draft depth and the grounding line (``'gl_gradient'``)
    second_crit : xr.DataArray
        Boolean where the gradient between the ice draft depth and the grounding line is negative.
    """
       
    # add dimension for iteration to the GL depth dataarray
    other_2 = xr.DataArray(np.arange(1, 3), coords=[('iteration', np.arange(1, 3))])
    xr_gl_depth_iter = xr.broadcast(GL_depth, other_2)[0].copy()
    ds_isf_lazeroms = xr.Dataset(data_vars={'gl_depth_iter': xr_gl_depth_iter})

    for n in range(1, mid_coord):
        # for each iteration, enlarge the mask
        weight_mask = ds_isf_weights['weights_gl_depth'].sel(x0=range(mid_coord - n, mid_coord + n + 1),
                                                             y0=range(mid_coord - n, mid_coord + n + 1))
        # use the correlation function to sum over all relevant points in that direction
        if n == 1:
            ds_isf_lazeroms['gl_depth_iter'].loc[dict(iteration=1)] = xr_nd_corr_sig(GL_depth,weight_mask).transpose("y","x","direction")
        else:
            # checks if new iteration is lower than the one before (for each direction). If yes, it means that we already found what we want
            ds_isf_lazeroms['gl_depth_iter'].loc[dict(iteration=2)] = xr_nd_corr_sig(GL_depth,weight_mask).transpose("y","x","direction")
            ds_isf_lazeroms['gl_depth_iter'].loc[dict(iteration=1)] = ds_isf_lazeroms['gl_depth_iter'].where(np.abs(ds_isf_lazeroms['gl_depth_iter']) > 0.1).max('iteration')

    # cut out only the ice shelf (no grounding line)
    ds_isf_lazeroms['gl_depth'] = ds_isf_lazeroms['gl_depth_iter'].sel(iteration=1).where(plume_var_of_int['ISF_mask'] == kisf)
    # difference between zGL and zb - when checking results, please note that the y coordinate is in reverse order so y=-1 will look up and y=1 will look down
    ds_isf_lazeroms['gl_gradient'] = ds_isf_lazeroms['gl_depth'] - draft_depth
    # 2nd criterion: grounding line must be deeper than the point
    second_crit = ds_isf_lazeroms['gl_gradient'] < 0
    
    return ds_isf_lazeroms, second_crit

def summarize_alpha_zGL_lazero(kisf, ds_isf_lazeroms, sn_isf, first_crit, second_crit, plume_var_of_int, draft_depth):
    
    """
    Summarize all criteria to compute zGL and alphas in the Lazeroms approach.
    
    This function summarizes the first and second criteria to infer zGl and alphas following the methodology in Lazeroms et al., 2018.

    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    ds_isf_lazeroms: xr.Dataset
        Dataset containing the grounding line depth at the origin of the plume (``'gl_depth'``) and the gradient between the ice draft depth and the grounding line (``'gl_gradient'``)
    sn_isf : xr.DataArray
        Basal slope in all 16 directions
    first_crit : xr.DataArray
        Boolean where sn_sf > 0
    second_crit : xr.DataArray
        Boolean where the gradient between the ice draft depth and the grounding line is negative.
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` and ``'GL_mask'``
    draft_depth : 
        Ice draft depth in m (negative downwards) extended through the 'direction' dimension.
        
    Returns
    -------
    go_back_to_whole_grid_alpha: xr.DataArray
        Mean angle at the plume origin in rad for each point.
    go_back_to_whole_grid_zgl: xr.DataArray
        Mean depth of the grounding line in m at the plume origin for each point (negative downwards). 
    """
    
    xx = plume_var_of_int.x
    dx = xx[1] - xx[0]
    
    # make a mean over all directions where 1st and 2nd criterion are valid
    ds_isf_lazeroms['gl_depth_mean'] = ds_isf_lazeroms['gl_depth'].where(first_crit & second_crit).mean('direction')
    # set nans within an ice shelf to local draft depth
    ds_isf_lazeroms['gl_depth_mean'] = ds_isf_lazeroms['gl_depth_mean'].where(np.abs(ds_isf_lazeroms['gl_depth_mean']) > 0, draft_depth.isel(direction=0).drop('direction')).where(plume_var_of_int['ISF_mask'] == kisf)
    
    # put back on the whole grid and write into summary dataset
    go_back_to_whole_grid_zgl = ds_isf_lazeroms['gl_depth_mean'].reindex_like(plume_var_of_int['ISF_mask'])
    
    #avoid strong frontal slope    
    dIF_isf = plume_var_of_int['dIF'].where(plume_var_of_int['ISF_mask'] == kisf)
    dIF_isf_corr = dIF_isf.where(dIF_isf/(abs(dx)/2) < 1,1) #check again with Nico, if I understood it right (MIN to avoid strong frontal slope)

    # for the alphas, same procedure, 2 criteria
    ds_isf_lazeroms['alphas_mean'] = np.arctan(sn_isf.where(first_crit & second_crit).mean('direction')) * dIF_isf_corr
    # set nans within an ice shelf to 0
    ds_isf_lazeroms['alphas_mean'] = ds_isf_lazeroms['alphas_mean'].where(np.abs(ds_isf_lazeroms['alphas_mean']) > 0, 0).where(plume_var_of_int['ISF_mask'] == kisf)   

    # put back on the whole grid and write into summary dataset
    go_back_to_whole_grid_alpha = ds_isf_lazeroms['alphas_mean'].reindex_like(plume_var_of_int['ISF_mask'])
     
    return go_back_to_whole_grid_alpha, go_back_to_whole_grid_zgl

def compute_zGL_alpha_lazero(kisf, plume_var_of_int, ice_draft_neg, dx, dy):
    
    """
    Compute zGL and alphas in the Lazeroms approach.
    
    This function computes zGl and alphas following the methodology in Lazeroms et al., 2018.

    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` and ``'GL_mask'``
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards.
    ds_isf_lazeroms: xr.Dataset
        Dataset containing the grounding line depth at the origin of the plume (``'gl_depth'``) and the gradient between the ice draft depth and the grounding line (``'gl_gradient'``)
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
        
    Returns
    -------
    alpha: xr.DataArray
        Mean angle at the plume origin in rad for each point.
    zGL: xr.DataArray
        Mean depth of the grounding line in m at the plume origin for each point (negative downwards). 
    """
    
    ds_weights = create_16_dir_weights()
         
    # prepare mask for whole domain (GL + ice shelf)
    plume_var_of_int['GL_and_ISF_mask'] = plume_var_of_int['GL_mask'].combine_first(plume_var_of_int['ISF_mask'])
    isf_and_GL_mask = plume_var_of_int['GL_and_ISF_mask'].where(
        (plume_var_of_int['ISF_mask'] == kisf) | (plume_var_of_int['GL_mask'] == kisf)).dropna(how='all',dim='x').dropna(how='all', dim='y')
    ice_draft_neg_isf = ice_draft_neg.where(isf_and_GL_mask == kisf)

    # boundary size
    x_size = len(isf_and_GL_mask.x)
    y_size = len(isf_and_GL_mask.y)

    GL_depth, sn_isf, first_crit, draft_depth = first_criterion_lazero(kisf, plume_var_of_int, ice_draft_neg_isf, isf_and_GL_mask, ds_weights, dx, dy)
    ds_isf_weights, mid_coord = prepare_filter_16dir_isf(isf_and_GL_mask, GL_depth, x_size, y_size, ds_weights)
    ds_isf_lazeroms, second_crit = second_criterion_lazero(kisf, plume_var_of_int, ds_isf_weights,GL_depth, mid_coord, draft_depth)
    alpha, zGL = summarize_alpha_zGL_lazero(kisf, ds_isf_lazeroms, sn_isf, first_crit, second_crit, plume_var_of_int, draft_depth)
    
    return alpha, zGL

def create_8_dir_weights():
    
    """
    Prepare correlation filter in 8 directions.
    
    Returns
    -------
    ds_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions.
    """
    
    #print('prepare correlation mask to check the 16 defined directions')
    #prepare correlation mask to check the 16 defined directions (1 in the middle and -1 in all interesting directions)
    dir_x = np.arange(-1, 2)
    dir_y = np.arange(-1, 2)

    weights_gradients = np.zeros((3, 3, 8))
    weights_gradients[1, 1, :] = 1
    layer_x = []
    layer_y = []

    n = 0
    for shift_x in dir_x:
        for shift_y in dir_y:

            cond_0 = (shift_x == 0 and shift_y == 0) 

            if not cond_0:

                layer_x.append(shift_x)
                layer_y.append(shift_y)
                weights_gradients[1 + shift_y, 1 + shift_x, n] = -1 # y first, x then because of the initial grid being like this
                n = n + 1

    # make a dataset with it to remember the associated shift_x and shift_y
    xr_weights = xr.DataArray(data=weights_gradients, dims=['y0', 'x0', 'direction'], coords={'direction': np.arange(8)})
    ds_weights = xr.Dataset(data_vars={'weights': xr_weights})
    ds_weights['shift_x'] = xr.DataArray(data=np.array(layer_x), dims=['direction'])
    ds_weights['shift_y'] = xr.DataArray(data=np.array(layer_y), dims=['direction'])
    return ds_weights



def first_criterion_lazero_general(kisf, plume_var_of_int, ice_draft_neg_isf, isf_and_GL_mask, ds_weights, dx, dy, dir_nb=16, grad_corr=0, extra_shift=2):

    """
    Define first criterion for the plume parameters using a smoother version of the 16 directions and permitting to use a different amount of directions
    
    This function computes the basal slope and identifies the first criterion, following the methodology in Lazeroms et al;, 2018.

    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` and ``'GL_mask'``
    ice_draft_neg_isf : xr.DataArray
        Ice draft depth for the given ice shelf in m. Negative downwards.
    isf_and_GL_mask : xr.DataArray
        Mask of the domain covered by the ice shelf and the grounding line (this extra mask is needed if the grounding line is defined on ground points)
    ds_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions.
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
    dir_nb: int
        Amount of directions used. I tried with 8, 16, 24. Decided to stay with 16.
    grad_corr: int
        If we want to add some uncertainty in the slopes (adds grad_corr to the gradient) => makes it easier to have positive slopes when the differences are tiny.
    extra_shift: int
        Should be 2 if you do the smooth version, otherwise 1.
        
        
    Returns
    -------
    GL_depth : xr.DataArray
        Depth of the grounding line points (negative downwards).
    sn_isf : xr.DataArray
        Basal slope in all 16 directions
    first_crit : xr.DataArray
        Boolean where sn_sf > 0
    draft_depth : 
        Ice draft depth in m (negative downwards) extended through the 'direction' dimension.
    """
    
    # add dimension for directions to the ice_draft array
    other = xr.DataArray(np.zeros(dir_nb), coords=[('direction', np.arange(dir_nb))])
    ice_draft_neg_dirs, other2 = xr.broadcast(ice_draft_neg_isf, other)

    # draft depth only on the ice shelf
    draft_depth = ice_draft_neg_dirs.where(isf_and_GL_mask).where((plume_var_of_int['ISF_mask'] == kisf))

    # grounding line depth only where grounding line
    GL_depth = ice_draft_neg_dirs.where(isf_and_GL_mask).where(plume_var_of_int['GL_mask'] == kisf)
    GL_depth = GL_depth.where(GL_depth < 0, 0)

    # apply the correlation filter to compute gradients in the 16 directions (xr_nd_corr_sig does not work for whatever reason :( ))
    gradients = xr_nd_corr(draft_depth, ds_weights['weights'])

    # compute the sn - basal slope - to be consistent with the origin of the plumes, we cut the basal slopes after ice shelves as well - but might need to think about what happens when several ice shelves are touching each other
    sn_isf = gradients / np.sqrt((ds_weights['shift_x'] * extra_shift * np.abs(dx)) ** 2 + (ds_weights['shift_y'] * extra_shift * np.abs(dy)) ** 2)
    # adding correction for criterion
    sn_isf_corr = (gradients +  grad_corr) / np.sqrt((ds_weights['shift_x'] * extra_shift * np.abs(dx)) ** 2 + (ds_weights['shift_y'] * extra_shift * np.abs(dy)) ** 2)
    # 1st criterion: sn > 0
    first_crit = sn_isf > 0
    first_crit_corr = sn_isf_corr > 0

    return sn_isf, sn_isf_corr, first_crit, first_crit_corr

def create_16_dir_weights_across():
    
    """
    Prepare correlation filter in 16 directions in a smooth way.
    
    This function prepares the correlation filter in 16 directions, following the methodology in Lazeroms et al;, 2018, BUT making it across the point (instead of finishing at the point).

    Returns
    -------
    ds_weights : xr.Dataset
        Weights for the filter and information about the x- and y-shift in the 16 directions.
    """
    
    #print('prepare correlation mask to check the 16 defined directions')
    #prepare correlation mask to check the 16 defined directions (1 in the middle and -1 in all interesting directions)
    dir_x = np.arange(-2, 3)
    dir_y = np.arange(-2, 3)

    weights_gradients = np.zeros((5, 5, 16))
    #weights_gradients[2, 2, :] = 1
    layer_x = []
    layer_y = []

    n = 0
    for shift_x in dir_x:
        for shift_y in dir_y:

            cond_0 = (shift_x == 0 and shift_y == 0) | (shift_x == 0 and abs(shift_y) == 2) | (
                        abs(shift_x) == 2 and shift_y == 0) | (abs(shift_x) == 2 and abs(shift_y) == 2)

            if not cond_0:
                layer_x.append(shift_x)
                layer_y.append(shift_y)
                weights_gradients[2 + shift_y, 2 + shift_x, n] = -1 # y first, x then because of the initial grid being like this
                weights_gradients[2 - shift_y, 2 - shift_x, n] = 1
                n = n + 1

    # make a dataset with it to remember the associated shift_x and shift_y
    xr_weights = xr.DataArray(data=weights_gradients, dims=['y0', 'x0', 'direction'], coords={'direction': np.arange(16)})
    ds_weights = xr.Dataset(data_vars={'weights': xr_weights})
    ds_weights['shift_x'] = xr.DataArray(data=np.array(layer_x), dims=['direction'])
    ds_weights['shift_y'] = xr.DataArray(data=np.array(layer_y), dims=['direction'])
    return ds_weights

def lazero_GL_alpha_kisf_newmethod(kisf, ice_draft_neg_isf, GL_mask, isf_and_GL_mask, dist_incl, weights8_0, weights16_0, mid_coord, sn_isf, first_crit):
    """
    
    This function computes the plume departing grounding line depth and the local angle in a smoother manner than Lazeroms et al. 2018. 
    Remains heavily inspired from Lazeroms et al. 2018 (using the 16 directions).
    Includes an option to extend the grounding line to neighboring points in case the original grounding line is weirdly shallow.
    This will produce fields with potential regions of nans because there is an obstacle with too many negative slopes. If you want to get rid of these obstacles,
    use the newmethod2 below.
    
    kisf : int
        ID of the ice shelf of interest
    ice_draft_neg : xr.DataArray
        Ice draft depth (Negative with depth!)
    GL_mask : xr.DataArray
        Mask of the Antarctic grounding lines
    isf_and_GL_mask : xr.DataArray  
        Mask of the isf and associated GL
    dist_incl : int
        Distance, in grid cells, to count within the grounding line
    weights8_0 : xr.Dataset
        Contains the weights (0,1) to look at the 8 neighbours of a point
    weights16_0 : xr.Dataset
        Contains the weights (0,1) to look in the 16 directions, starting at the point
    mid_coord : int
        Indication on how many times to propagate the grounding line
    sn_isf : xr.DataArray
        Slopes
    first_crit : xr.DataArray
       First criterion

        
    Returns
    -------
    alpha: xr.DataArray
        Slopes to use for Lazeroms version of plume parameterisation for one ice shelf.
    zGL: xr.DataArray
        Grounding line depth to use for Lazeroms version of plume parameterisation for one ice shelf.
    
    """
    
    # Enlarge GL mask to dist_incl rows (e.g. if your initial GL is shallow)
    GL_mask1_0 = (GL_mask == kisf)
    GL_2_mask = GL_mask1_0
    for n in range(dist_incl):
        GL_2 = xr_nd_corr(GL_2_mask, weights8_0['weights'])
        GL_2_sum = GL_2.sum('direction').where(isf_and_GL_mask == kisf)
        GL_2_mask = (GL_2_sum > 0).astype(int)
        
    # Cut out the GL band in draft depth
    GL_depth_isf = -1*(ice_draft_neg_isf.where(GL_2_mask))
    
    # Propagate GL depth in the whole ice shelf
    
    # Initialise the field at grounding line
    GL_neighbors_new = GL_depth_isf
    sn_new = sn_isf.mean('direction')
    sn_new = sn_new.where(sn_new >= 0,0).where(GL_2_mask > 0)
    
    second_crit_all = GL_depth_isf* 0 + 1
    
    # Iterate to advace the propagation
    for n in range(mid_coord):

        GL_neighbors = xr_nd_corr(GL_neighbors_new, weights16_0['weights'])
        
        # cut out the newly formed data strip
        GL_neighbors_step = GL_neighbors.where(np.isnan(GL_neighbors_new))
        GL_neighbors_step = GL_neighbors_step.where(isf_and_GL_mask == kisf)

        # check if the propagated GL is deeper than point
        diff_base_GL = (-1*ice_draft_neg_isf - GL_neighbors_step)
        second_crit_n = diff_base_GL <= 0 #<=
        
        # combine this criterion and the slope criterion
        all_crit =  first_crit & second_crit_n #
        
        # make a mean over all valid GL depths
        GL_mean = GL_neighbors_step.where(all_crit).mean('direction')
        GL_neighbors_new = GL_neighbors_new.where(GL_neighbors_new > 0,GL_mean)    
        
        # make a mean over all valid slopes
        sn_mean = sn_isf.where(GL_neighbors_step, drop=True).where(all_crit).mean('direction')
        sn_new = sn_new.where(sn_new >= 0,sn_mean)
        
        second_crit_all = second_crit_all.where(second_crit_all > 0,second_crit_n)   
        
    return np.arctan(sn_new), -1*GL_neighbors_new


def lazero_GL_alpha_kisf_newmethod2(kisf, ice_draft_neg_isf, GL_mask, isf_and_GL_mask, gl_mask_isl, dist_incl, weights8_0, weights16_0, mid_coord, sn_isf, first_crit, sn_isf_corr, first_crit_corr):
    """
    
    This function computes the plume departing grounding line depth and the local angle in a smoother manner than Lazeroms et al. 2018. 
    Remains heavily inspired from Lazeroms et al. 2018 (using the 16 directions).
    Includes an option to extend the grounding line to neighboring points in case the original grounding line is weirdly shallow.
    
    kisf : int
        ID of the ice shelf of interest
    ice_draft_neg : xr.DataArray
        Ice draft depth (Negative with depth!)
    GL_mask : xr.DataArray
        Mask of the Antarctic grounding lines
    isf_and_GL_mask : xr.DataArray  
        Mask of the isf and associated GL
    dist_incl : int
        Distance, in grid cells, to count within the grounding line
    weights8_0 : xr.Dataset
        Contains the weights (0,1) to look at the 8 neighbours of a point
    weights16_0 : xr.Dataset
        Contains the weights (0,1) to look in the 16 directions, starting at the point
    mid_coord : int
        Indication on how many times to propagate the grounding line
    sn_isf : xr.DataArray
        Slopes
    first_crit : xr.DataArray
       First criterion
    
    """
    
    # Enlarge GL mask to dist_incl rows (e.g. if your initial GL is shallow)
    GL_mask1_0 = (GL_mask == kisf)
    GL_2_mask = GL_mask1_0
    for n in range(dist_incl):
        GL_2 = xr_nd_corr(GL_2_mask, weights8_0['weights'])
        GL_2_sum = GL_2.sum('direction').where(isf_and_GL_mask == kisf)
        GL_2_mask = (GL_2_sum > 0).astype(int)
        
    # Cut out the GL band in draft depth
    GL_depth_isf = -1*(ice_draft_neg_isf.where(GL_2_mask))
    


    # Initialise the field at grounding line
    GL_neighbors_new = GL_depth_isf
    sn_new = sn_isf.where(first_crit).mean('direction')
    sn_new = sn_new.where(sn_new > 0,0).where(GL_2_mask > 0)

    second_crit_all = GL_depth_isf* 0 + 1

    diff_masks = 1
    i = 0
    diff_stop = 0

    while diff_stop < 3:

        mask_old_domain = np.isnan(GL_neighbors_new)

        GL_neighbors = xr_nd_corr(GL_neighbors_new, weights16_0['weights'])

        # cut out the newly formed data strip
        GL_neighbors_step = GL_neighbors.where(np.isnan(GL_neighbors_new))
        GL_neighbors_step = GL_neighbors_step.where(isf_and_GL_mask == kisf)

        # check if the propagated GL is deeper than point
        diff_base_GL = (-1*ice_draft_neg_isf - GL_neighbors_step)
        second_crit_n = diff_base_GL < 0 #<=

        # combine this criterion and the slope criterion
        all_crit =  first_crit & second_crit_n #

        if diff_masks != 0:

            # make a mean over all valid GL depths
            GL_mean = GL_neighbors_step.where(all_crit).mean('direction')
            GL_neighbors_new = GL_neighbors_new.where(GL_neighbors_new > 0,GL_mean)    

            # make a mean over all valid slopes
            sn_mean = sn_isf.where(all_crit).mean('direction')
            sn_new = sn_new.where(sn_new > 0,sn_mean)

            second_crit_all = second_crit_all.where(second_crit_all > 0,second_crit_n)   

            diff_stop = 0

        else:

            #print('Entering obstacle option')

            # insert corrected sn and first crit
            first_crit_corr2 = first_crit.where((all_crit.sum('direction') > 0), first_crit_corr)
            all_crit_corr = (first_crit_corr2 & second_crit_n).where(np.isfinite(GL_neighbors_step))
            sn_isf_corr2 = sn_isf.where((all_crit.sum('direction') > 0), sn_isf_corr).where(np.isfinite(GL_neighbors_step))

            # make a mean over all valid GL depths
            GL_mean = GL_neighbors_step.where(all_crit_corr).mean('direction')
            GL_neighbors_new = GL_neighbors_new.where(GL_neighbors_new > 0,GL_mean)    

            # make a mean over all valid slopes
            sn_mean = sn_isf_corr2.where(all_crit_corr).mean('direction')
            sn_new = sn_new.where(sn_new > 0,sn_mean)

            second_crit_all = second_crit_all.where(second_crit_all > 0,second_crit_n)   

        if diff_stop == 2:
            
            # cut out areas that are nan and potentially are near a grounding line of an island
            start_new_GL = (gl_mask_isl) & ~(GL_neighbors_new > 0) & ~(GL_mask == kisf) & (isf_and_GL_mask == kisf)
            
            GL_neighbors_new2 = -1*(ice_draft_neg_isf.where(start_new_GL))
            sn_new2 = sn_isf.where(first_crit).mean('direction')
            sn_new2 = sn_new2.where(sn_new2 > 0,0).where(start_new_GL > 0)
            second_crit_all2 = GL_neighbors_new2 * 0 + 1
            
            mask_old_domain2 = np.isnan(GL_neighbors_new2)
            
            diff_masks2 = 1

            for n in range(50):

                GL_neighbors = xr_nd_corr(GL_neighbors_new2, weights16_0['weights'])

                # cut out the newly formed data strip
                GL_neighbors_step = GL_neighbors.where(np.isnan(GL_neighbors_new2))
                GL_neighbors_step = GL_neighbors_step.where(isf_and_GL_mask == kisf)

                # check if the propagated GL is deeper than point
                diff_base_GL = (-1*ice_draft_neg_isf - GL_neighbors_step)
                second_crit_n = diff_base_GL < 0 #<=

                # combine this criterion and the slope criterion
                all_crit =  first_crit & second_crit_n #
                
                if diff_masks2 != 0 :
                    # make a mean over all valid GL depths
                    GL_mean = GL_neighbors_step.where(all_crit).mean('direction')
                    GL_neighbors_new2 = GL_neighbors_new2.where(GL_neighbors_new2 > 0,GL_mean)    

                    # make a mean over all valid slopes
                    sn_mean = sn_isf.where(all_crit).mean('direction')
                    sn_new2 = sn_new2.where(sn_new2 > 0,sn_mean)

                    second_crit_all2 = second_crit_all2.where(second_crit_all2 > 0,second_crit_n)   
                    mask_new_domain2 = np.isnan(GL_neighbors_new2)
                    diff_masks2 = (mask_new_domain2.astype(int) - mask_old_domain2.astype(int)).sum().values
                    
                else:
                
                    # insert corrected sn and first crit
                    first_crit_corr2 = first_crit.where((all_crit.sum('direction') > 0), first_crit_corr)
                    all_crit_corr = (first_crit_corr2 & second_crit_n).where(np.isfinite(GL_neighbors_step))
                    sn_isf_corr2 = sn_isf.where((all_crit.sum('direction') > 0), sn_isf_corr).where(np.isfinite(GL_neighbors_step))

                    # make a mean over all valid GL depths
                    GL_mean = GL_neighbors_step.where(all_crit_corr).mean('direction')
                    GL_neighbors_new2 = GL_neighbors_new2.where(GL_neighbors_new2 > 0,GL_mean)    

                    # make a mean over all valid slopes
                    sn_mean = sn_isf_corr2.where(all_crit_corr).mean('direction')
                    sn_new2 = sn_new2.where(sn_new2 > 0,sn_mean)

                    second_crit_all2 = second_crit_all2.where(second_crit_all2 > 0,second_crit_n)   
                
            # fill nans with this new product
            GL_neighbors_new = GL_neighbors_new.where(np.isfinite(GL_neighbors_new), GL_neighbors_new2)
            sn_new = sn_new.combine_first(sn_new2)
            
        # check if we still have obstacles
        mask_new_domain = np.isnan(GL_neighbors_new)
        diff_masks = (mask_new_domain.astype(int) - mask_old_domain.astype(int)).sum().values

        # check if we have reached the maximum
        diff_mask_isf = np.isnan(GL_neighbors_new) & (isf_and_GL_mask == kisf)
        
        if diff_masks == 0:
            #print('mask did not change', diff_stop)
            diff_stop = diff_stop+1
                    
        i = i+1
        
        if i == 500:
            return  np.arctan(sn_new), -1*GL_neighbors_new
            break
        
            
    return  np.arctan(sn_new), -1*GL_neighbors_new

def compute_zGL_alpha_lazero_newmethod(kisf, plume_var_of_int, ice_draft_neg, dx, dy, dir_nb, grad_corr, extra_shift, dist_incl):

    """
    Compute zGL and alphas with a revisitation of the Lazeroms approach.
    
    This function computes zGl and alphas following a slightly different methodology than Lazeroms et al., 2018 but inspired by it.

    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` and ``'GL_mask'``
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards.
    ds_isf_lazeroms: xr.Dataset
        Dataset containing the grounding line depth at the origin of the plume (``'gl_depth'``) and the gradient between the ice draft depth and the grounding line (``'gl_gradient'``)
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
        
    Returns
    -------
    alpha: xr.DataArray
        Mean angle at the plume origin in rad for each point.
    zGL: xr.DataArray
        Mean depth of the grounding line in m at the plume origin for each point (negative downwards). 
    """
    
    weights8 = create_8_dir_weights()
    weights16 = create_16_dir_weights()
    
    if dir_nb == 16:
        if extra_shift == 2:
            weights_across = create_16_dir_weights_across()
        elif extra_shift == 1:
            weights_across = create_16_dir_weights()
    elif dir_nb == 8:
        if extra_shift == 2:
            weights_across = create_8_dir_weights_across()
        elif extra_shift == 1:
            weights_across = create_8_dir_weights()   


    weights8_0 = weights8.where(weights8 < 0,0) * -1
    weights8_0 = weights8_0.where(weights8_0 > 0,0)

    weights16_0 = weights16.where(weights16 < 0,0) * -1
    weights16_0 = weights16_0.where(weights16_0 > 0,0)

         
    # prepare mask for whole domain (GL + ice shelf)
    plume_var_of_int['GL_and_ISF_mask'] = plume_var_of_int['GL_mask'].combine_first(plume_var_of_int['ISF_mask'])
    isf_and_GL_mask = plume_var_of_int['GL_and_ISF_mask'].where(
        (plume_var_of_int['ISF_mask'] == kisf) | (plume_var_of_int['GL_mask'] == kisf)).dropna(how='all',dim='x').dropna(how='all', dim='y')
    ice_draft_neg_isf = ice_draft_neg.where(isf_and_GL_mask == kisf)

    # first crit        
    sn_isf, sn_isf_corr, first_crit, first_crit_corr = first_criterion_lazero_general(kisf, plume_var_of_int, ice_draft_neg_isf, isf_and_GL_mask, weights_across, dx, dy, dir_nb=dir_nb, grad_corr=grad_corr, extra_shift=extra_shift) 

    # second crit and zGL and alpha
    alpha_kisf, zGL_kisf = lazero_GL_alpha_kisf_newmethod2(kisf, ice_draft_neg_isf, plume_var_of_int['GL_mask'], isf_and_GL_mask, plume_var_of_int['GL_mask_with_isl'], dist_incl, weights8_0, weights16_0, 200, sn_isf, first_crit, sn_isf_corr, first_crit_corr)
    #alpha_kisf = alpha_kisf.where(alpha_kisf < 0, 0).where(np.isfinite(isf_and_GL_mask))
    #zGL_kisf = zGL_kisf.where(np.isfinite(zGL_kisf), ice_draft_neg_isf).where(np.isfinite(isf_and_GL_mask))
    
    #alpha_kisf = alpha_kisf.where(np.isfinite(alpha_kisf), 0)
    #zGL_kisf = zGL_kisf.where(np.isfinite(zGL_kisf), ice_draft_neg_isf)
    
    go_back_to_whole_grid_alpha = alpha_kisf.reindex_like(plume_var_of_int['ISF_mask'])
    go_back_to_whole_grid_zgl = zGL_kisf.reindex_like(plume_var_of_int['ISF_mask'])   
    
    return go_back_to_whole_grid_alpha,  go_back_to_whole_grid_zgl


def compute_alpha_local(kisf, plume_var_of_int, ice_draft_neg, dx, dy):   

    """
    Compute alphas like in Appendix B of Favier et al., 2019 TCDiscussions.
    
    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` and ``'dIF'``
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards.
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
        
    Returns
    -------
    go_back_to_whole_grid_local_alpha: xr.DataArray
        Local slope angle in rad for each point.
    """

    # cut out the ice shelf of interest
    draft_isf = ice_draft_neg.where(plume_var_of_int['ISF_mask'] == kisf, drop=True)

    shiftedx_minus = draft_isf.shift(x=-1)
    shiftedx_plus = draft_isf.shift(x=1)
    xslope = check_slope_one_dimension(draft_isf, shiftedx_plus, shiftedx_minus, dx)

    shiftedy_minus = draft_isf.shift(y=-1)
    shiftedy_plus = draft_isf.shift(y=1)
    yslope = check_slope_one_dimension(draft_isf, shiftedy_plus, shiftedy_minus, dy)

    dIF_isf = plume_var_of_int['dIF'].where(plume_var_of_int['ISF_mask'] == kisf)
    dIF_isf_corr = dIF_isf.where(dIF_isf/(abs(dx)/2) < 1,1) #check again with Nico, if I understood it right (MIN to avoid strong frontal slope)

    local_alpha = np.arctan(np.sqrt(xslope ** 2 + yslope ** 2)) * dIF_isf_corr

    go_back_to_whole_grid_local_alpha = local_alpha.reindex_like(plume_var_of_int['ISF_mask'])

    return go_back_to_whole_grid_local_alpha

def compute_zGL_alpha_all(plume_var_of_int, opt, ice_draft_neg, grad_corr=0, dir_nb=16, extra_shift=2, dist_incl=0):

    """
    Compute grounding line and angle for the plume for all ice shelves.
    
    Parameters
    ----------
    plume_var_of_int : xr.Dataset
        Dataset containing 'ISF_mask', 'GL_mask', 'IF_mask', 'dIF', 'dGL_dIF', 'latitude', 'longitude', 'front_ice_depth_avg'
    opt : str
        Method after which to compute the depth and angle. Can be:
            cavity : Zgl and Alpha are found between draft point and deepest GL point.
            lazero: original from Lazeroms et al. 2018
            local: local slope
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards.
        
    Returns
    -------
    plume_alpha: xr.DataArray
        Angle in rad for each point.
    plume_zGL : xr.DataArray
        Depth of the grounding line in m at the plume origin for each point (negative downwards). 
    """

    # create a map with the angles
    empty_map = xr.DataArray(data=np.zeros((len(plume_var_of_int.y), len(plume_var_of_int.x))), dims=['y', 'x'],
                                         coords={'y': plume_var_of_int.y, 'x': plume_var_of_int.x})
    plume_alpha = empty_map.copy()
    plume_zGL = empty_map.copy()    
    
    dx = plume_var_of_int.x[2] - plume_var_of_int.x[1]
    dy = plume_var_of_int.y[2] - plume_var_of_int.y[1]
    
    if opt == 'cavity':
        print('----------- PREPARATION OF ZGL AND ALPHA WITH CAVITY APPROACH -----------')
        alpha = compute_alpha_cavity(plume_var_of_int)
        zGL = -1*plume_var_of_int['GL_max']

    elif opt == 'lazero':
        print('----------- PREPARATION OF ZGL AND ALPHA WITH LAZEROMS 2018 -----------')

    elif opt == 'new_lazero':
        print('----------- PREPARATION OF ZGL AND ALPHA WITH MODIFIED LAZEROMS 2018 -----------')
    
    elif opt == 'local':     
        print('----------- PREPARATION OF ZGL AND ALPHA WITH APPENDIX B FAVIER-----------')
    
    
    weights4 = create_4_dir_weights()
    mask_0_1_2 = plume_var_of_int['ISF_mask'].where(plume_var_of_int['ISF_mask'] < 2,2)
    corr_mask = xr_nd_corr(mask_0_1_2, weights4['weights'])
    corr_mask_max = np.abs(corr_mask).max('direction')
    plume_var_of_int['GL_mask_with_isl'] = (corr_mask_max == 2)
        
    for kisf in tqdm(plume_var_of_int['Nisf']):
        if ~np.isnan(plume_var_of_int['GL_mask'].where(plume_var_of_int['GL_mask'] == kisf).max()):
            
            if opt == 'lazero':
                alpha, zGL = compute_zGL_alpha_lazero(kisf, plume_var_of_int, ice_draft_neg, dx, dy)
            elif opt == 'new_lazero':
                alpha, zGL = compute_zGL_alpha_lazero_newmethod(kisf, plume_var_of_int, ice_draft_neg, dx, dy, dir_nb, grad_corr, extra_shift, dist_incl)
            elif opt == 'local':
                alpha0, zGL = compute_zGL_alpha_lazero(kisf, plume_var_of_int, ice_draft_neg, dx, dy)
                alpha = compute_alpha_local(kisf, plume_var_of_int, ice_draft_neg, dx, dy)
            
            if opt == 'cavity':
                plume_alpha = plume_alpha.where(plume_var_of_int['ISF_mask'] != kisf, alpha.sel(Nisf=kisf))
                plume_zGL = plume_zGL.where(plume_var_of_int['ISF_mask'] != kisf, zGL.sel(Nisf=kisf))
            else:
                plume_alpha = plume_alpha.where(plume_var_of_int['ISF_mask'] != kisf, alpha)
                plume_zGL = plume_zGL.where(plume_var_of_int['ISF_mask'] != kisf, zGL)
    
    return plume_alpha, plume_zGL


    
def prepare_plume_charac(plume_param_options, ice_draft_pos, plume_var_of_int, grad_corr=0, dir_nb=16, extra_shift=2, dist_incl=0):

    """
    Overall function to compute the plume characteristics depending on geometry.
    
    Parameters
    ----------
    plume_param_options : list of str
        Parametrization options (typically 'cavity', 'lazero' and 'local').
    ice_draft_pos : xr.DataArray
        Ice draft depth in m. Positive downwards.
    plume_var_of_int : xr.Dataset
        Dataset containing 'ISF_mask', 'GL_mask', 'IF_mask', 'dIF', 'dGL_dIF', 'latitude', 'longitude', 'front_ice_depth_avg'
        
    Returns
    -------
    outfile : xr.Dataset
        Dataset containing plume characteristics depending on geometry needed for plume parametrization.
    """
    
    print('----------------- PREPARATION GL MAX --------------')
    ice_draft_neg = -1*ice_draft_pos
    plume_var_of_int = add_GL_max(plume_var_of_int, ice_draft_pos)
    plume_var_of_int = prepare_plume_dataset(plume_var_of_int,plume_param_options)
    
    for opt in plume_param_options: 
        alpha, zGL = compute_zGL_alpha_all(plume_var_of_int, opt, ice_draft_neg, grad_corr, dir_nb, extra_shift, dist_incl)
        plume_var_of_int['alpha'].loc[dict(option=opt)] = alpha
        plume_var_of_int['zGL'].loc[dict(option=opt)] = zGL
           
    
    #####
    ##### WRITE INTO NETCDF
    #####

    print('----------- PREPARE OUTPUT DATASET --------------')

    # write to netcdf
    outfile = xr.Dataset(
        {'zGL': (plume_var_of_int['zGL'].dims, plume_var_of_int['zGL'].values),
         'alpha': (plume_var_of_int['alpha'].dims, plume_var_of_int['alpha'].values)
         },
        coords={'y': plume_var_of_int.y.values, 'x': plume_var_of_int.x.values, 'option': plume_param_options,
                'latitude': (plume_var_of_int['latitude'].dims, plume_var_of_int['latitude'].values),
                'longitude': (plume_var_of_int['longitude'].dims, plume_var_of_int['longitude'].values)})

    outfile['zGL'].attrs['standard_name'] = 'effective_grounding_line_depth'
    outfile['zGL'].attrs['long_name'] = 'Depth of possible grounding line points where plume starts'
    outfile['alpha'].attrs['standard_name'] = 'effective_slope_angle'
    outfile['alpha'].attrs['long_name'] = 'Slope angle'
    outfile['option'].attrs['standard_name'] = 'zGL_alpha_compute_option'
    outfile['option'].attrs['long_name'] = 'Computation option for computing zGL and alpha'

    # Global attributes
    outfile.attrs['history'] = 'Created with prepare_plume_charac() by C. Burgard'
    outfile.attrs['projection'] = 'Polar Stereographic South (71S,0E)'
    outfile.attrs['proj4'] = '+init=epsg:3031'
    outfile.attrs['Note'] = 'isf ID can be found in BedMachineAntarctica_2020-07-15_v02_reduced5_isf_masks_and_info.nc'

    return outfile
        

    
        
        





