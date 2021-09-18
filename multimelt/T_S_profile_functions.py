"""
This is a script to collect the functions to prepare the T and S profiles

@author: Clara Burgard
"""

import numpy as np
import xarray as xr
from tqdm.notebook import tqdm
import gsw
import itertools
import multimelt.useful_functions as uf

def cut_out_contshelf_offshore(input_fld, mask_ocean, offshore, contshelf):
    
    """
    Function to select the ocean region types (ocean, offshore, continental shelf) over which we want to take the mean profiles.
        
    Parameters
    ----------
    input_fld : xr.Dataset
        Dataset containing the variables 'temperature' and 'salinity' over the complete domain.
    mask_ocean : xr.DataArray
        Mask for open ocean (excluding ice shelf cavities).
    offshore : xr.DataArray
        Mask for ocean deeper than the continental shelf threshold.
    contshelf : xr.DataArray
        Mask for ocean shallower than the continental shelf threshold.

    Returns
    -------
    input_fld : xr.Dataset
        input_fld now also contains the temperature and salinity cut out for the different ocean region types (ocean, offshore, continental shelf) and a description of the attributes.
    """ 

    print('MASK THE DATA')

    for var in ['temperature','salinity']:
        input_fld[var+'_ocean'] = input_fld[var].where(mask_ocean)
        input_fld[var+'_cont_shelf'] = input_fld[var+'_ocean'].where(contshelf)
        input_fld[var+'_offshore'] = input_fld[var+'_ocean'].where(offshore)
    
    output_fld = input_fld[['temperature_cont_shelf', 'salinity_cont_shelf', 'temperature_offshore', 'salinity_offshore']]
    
    print('SET NETCDF ATTRIBUTES')

    # Attributes
    for app in ['cont_shelf','offshore']:
        for var in ['temperature','salinity']:
            output_fld[var+'_'+app].attrs['coordinates'] = 'lon lat time'
            output_fld[var+'_'+app].attrs['long_name'] = 'Values for sea_water_'+var+'_'+app+' over depth masked over different region types (continental shelf VS offshore)'
            output_fld[var+'_'+app].attrs['standard_name'] = 'sea_water_'+var+'_'+app
        output_fld['salinity_'+app].attrs['units'] = 'psu'
        output_fld['temperature_'+app].attrs['units'] = 'degrees_celsius'
        
    # Global attributes
    output_fld.attrs['history'] = 'Created with cut_out_contshelf_offshore() by C. Burgard'
    output_fld.attrs['projection'] = 'Polar Stereographic South (71S,0E)'
    output_fld.attrs['proj4'] = '+init=epsg:3031'
    output_fld.attrs['Note'] = 'The limit between offshore and continental shelf is -1500 m.'
    
    return output_fld

def mask_boxes_around_IF_new(lon, lat, mask_domains, front_min_lon, front_max_lon, front_min_lat, front_max_lat, lon_box, lat_box, isf_name):

    """
    Function to select the domains around the ice shelf front over which we want to take the mean profiles.
        
    Parameters
    ----------
    lon : xr.DataArray of floats
        Longitude
    lat : xr.DataArray of floats
        Latitude
    mask_domains : xr.DataArray of Bool
        Different types of domains (e.g. continental shelf/offshore)
    front_min_lon : xr.DataArray of floats
        Minimum longitude of the ice front for all ice shelves of interest.
    front_max_lon : xr.DataArray of floats
        Maximum longitude of the ice front for all ice shelves of interest.
    front_min_lat : xr.DataArray of floats
        Minimum latitude of the ice front for all ice shelves of interest.
    front_max_lat : xr.DataArray of floats
        Maximum latitude of the ice front for all ice shelves of interest.
    lon_box : xr.DataArray of floats
        Different ranges of longitude extending around the limit of the ice front.
    lat_box : xr.DataArray of floats
        Different ranges of latitude extending around the limit of the ice front.
    isf_name : xr.DataArray of str
        Name of the ice shelves of interest 

    Returns
    -------
    mask_complete : list of xr.DataArrays
        Gives back a list of mask for the different regions of interest around the ice front for the different ice shelves. Has the combined dimensions of mask_domains/lon_box/lat_box and front_min/max_lon/lat
    """     
    
    front_sup_lon = front_max_lon + lon_box
    front_inf_lon = front_min_lon - lon_box
    
    front_sup_lat = front_max_lat + lat_box
    front_inf_lat = front_min_lat - lat_box

    mask = (front_inf_lon <= lon) & (lon <= front_sup_lon) & (front_inf_lat <= lat) & (lat <= front_sup_lat) 
    
    # combine both Ross halfs
    mask_Ross = (((front_inf_lon <= lon) & (lon <= 180)) |  ((-180 <= lon) & (lon <= front_sup_lon))) & (front_inf_lat <= lat) & (lat <= front_sup_lat)
    
    # avoid spill on the other side of the Peninsula
    mask_east_peninsula_ok = ((lon > -66) & (lon < -44) & (lat > -74) & (lat < -67)) \
                            | ((lon > -65) & (lon < -44) & (lat > -67) & (lat < -66)) \
                            | ((lon > -63) & (lon < -44) & (lat > -66) & (lat < -65)) \
                            | ((lon > -61) & (lon < -44) & (lat > -65) & (lat < -64)) \
                            | ((lon > -58) & (lon < -44) & (lat > -64) & (lat < -59))
    
    nisf_east_peninsula = (front_min_lon > -66) & (front_max_lon < -46) & (front_min_lat > -74)
    
    mask_complete = mask.where(isf_name != 'Ross', mask_Ross).where(~nisf_east_peninsula, mask_east_peninsula_ok) & mask_domains
    
    return mask_complete

def estimate_S_abs(S_psu, depth, lon, lat):
    
    """
    Function to apply gsw.SA_from_SP to an xr.DataArray.
        
    Parameters
    ----------
    S_psu : xr.DataArray
        Practical salinity in psu.
    depth :
        Depth levels of S_psu in m. Depth must be positive.
        = Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : xr.DataArray of floats
        Longitude
    lat : xr.DataArray of floats
        Latitude
        
    Returns
    -------
    S_abs : xr.DataArray
        Absolute salinity in g/kg
    """   
    
    S_abs = xr.apply_ufunc(gsw.SA_from_SP,
                            S_psu,
                            depth,
                            lon,
                            lat, 
                            dask='parallelized',
                          output_dtypes=['float']
                )
    
    return S_abs

def estimate_theta(S_abs, T_insitu, depth):

    """
    Function to apply gsw.pt0_from_t to an xr.DataArray.
        
    Parameters
    ----------
    S_abs : xr.DataArray
        Absolute salinity in g/kg.
    T_insitu : xr.DataArray
        In-situ temperature in °C.
    depth :
        Depth levels of S_psu in m. Depth must be positive.
        = Sea pressure (absolute pressure minus 10.1325 dbar), dbar
        
    Returns
    -------
    theta : xr.DataArray
        Potential temperature in °C.
    """   
    
    theta = xr.apply_ufunc(gsw.pt0_from_t,
                           S_abs,
                           T_insitu,
                           depth,
                           dask='parallelized',
                          output_dtypes=['float'])
    
    return theta

def estimate_theta_through_S_abs(T_insitu, S_psu, depth, lon, lat):
    
    """
    Function to compute potential temperature from physical temperature in xr.DataArray.
        
    Parameters
    ----------
    T_insitu : xr.DataArray
        In-situ temperature in °C.
    S_psu : xr.DataArray
        Practical salinity in psu.
    depth :
        Depth levels of S_psu in m. Depth must be positive.
        = Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : xr.DataArray of floats
        Longitude
    lat : xr.DataArray of floats
        Latitude
        
    Returns
    -------
    theta : xr.DataArray
        Potential temperature in °C.
    """ 
    
    S_abs = estimate_S_abs(S_psu, depth, lon, lat)
    theta = estimate_theta(S_abs, T_insitu, depth)
    return theta


def split_by_chunks(dataset, dim):
    
    """
    Function to split a dataset in chunks over a given dimension. 
    
    Inspired from https://ncar.github.io/xdev/posts/writing-multiple-netcdf-files-in-parallel-with-xarray-and-dask/
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to be splitted
    dim : str
        Dimension along which to chunk
    
    Returns
    -------
    
    """
    
    chunk_slices = {}
    slices = []
    start = 0
    for chunk in dataset.chunks[dim]:
        if start >= dataset.sizes[dim]:
            break
        stop = start + chunk
        slices.append(slice(start, stop))
        start = stop
    chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]
        
def create_filepath(ds, prefix, outputpath, year):
    """
    Generate a filepath when given an xarray dataset.
    
    Inspired from https://ncar.github.io/xdev/posts/writing-multiple-netcdf-files-in-parallel-with-xarray-and-dask/
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of interest
    prefix : str
        Beginning of the name of the file
    outputpath : str
        Output path, where the files should be written.
    year : str
        End of the file (preferably explaining the splitted chunks). If it is chunked by time, it can be the year.
        
    Returns
    -------
    filepath : str
        Complete name of the output file (incl. path)
    
    """
    filepath = f'{outputpath}{prefix}_{year}.nc'
    return filepath
