import pandas as pd
import xarray as xr
import numpy as np

def slope_zonal_merid(kisf, plume_var_of_int, ice_draft_neg, dx, dy):
    

    """
    Compute lon and lat slope in x and slope in y direction
    
    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` , ``'longitude'``, ``'latitude'``
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards. Or other variable on which we want the slope (e.g. bathymetry)
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
    
    Returns
    -------
    merid_slope_whole_grid: xr.DataArray
        Slope (dvar/dlon) for each point.
    zonal_slope_whole_grid: xr.DataArray
        Slope (dvar/dlat) for each point.
        
    """
    
    # cut out the ice shelf of interest
    draft_isf = ice_draft_neg.where(plume_var_of_int['ISF_mask'] == kisf, drop=True)
    lon_isf = plume_var_of_int['longitude'].where(plume_var_of_int['ISF_mask'] == kisf, drop=True)
    
    shiftedx_minus = draft_isf.shift(x=-1)
    shiftedx_plus = draft_isf.shift(x=1)
    xslope = check_slope_one_dimension(draft_isf, shiftedx_plus, shiftedx_minus, dx)

    shiftedy_minus = draft_isf.shift(y=-1)
    shiftedy_plus = draft_isf.shift(y=1)
    yslope = check_slope_one_dimension(draft_isf, shiftedy_plus, shiftedy_minus, dy)
        
    dr = np.pi/180.
    #deltaphi = 90. - GEddxx !Geddxx= 90° in my config
    deltaphi = 0

    phi = (-1)*(lon_isf+deltaphi)*dr #to turn positively
    cphi = np.cos( phi )
    sphi = np.sin( phi )
    #
    ux = cphi*xslope.values + sphi*(-1)*yslope.values #because the y-axis is in the other direction
    vy = - sphi*xslope.values + cphi*(-1)*yslope.values #because the y-axis is in the other direction
    u_lon = xr.DataArray(ux, coords=lon_isf.coords).transpose('y','x').rename('slope_lon')
    v_lat = xr.DataArray(vy, coords=lon_isf.coords).transpose('y','x').rename('slope_lat')
    
    merid_slope_whole_grid = u_lon.reindex_like(plume_var_of_int['ISF_mask'])
    zonal_slope_whole_grid = v_lat.reindex_like(plume_var_of_int['ISF_mask'])
    
    return merid_slope_whole_grid, zonal_slope_whole_grid

