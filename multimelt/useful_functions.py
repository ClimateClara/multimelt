from pyproj import Transformer
from tqdm.notebook import trange, tqdm
import xarray as xr

# check if x and y are in range
def in_range(in_xy,txy):
    return ((in_xy >= min(txy)) & (in_xy < max(txy)))

# compute shortest distance on sphere between lat and lon
def dist_sphere(lon1,lon2,lat1,lat2):
    R = 6.371e6
    a =   np.sin(np.deg2rad(lat1-lat2)*0.5)**2 + np.cos(np.deg2rad(lat2)) * np.cos(np.deg2rad(lat1)) * np.sin(np.deg2rad(lon1-lon2)*0.5)**2    
    return abs( 2 * R * np.atan2( np.sqrt(a), np.sqrt(1-a) ) ) # distance in meters

def change_coord_latlon_to_stereo(meshlon,meshlat):
    ### Transformation from latlon to stereo
    trans_tostereo = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    meshx, meshy = trans_tostereo.transform(meshlon,meshlat)
    return meshx, meshy

def change_coord_stereo_to_latlon(meshx,meshy):
    ### Transformation from latlon to stereo
    trans_tolonlat = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    meshlon,meshlat = trans_tolonlat.transform(meshx,meshy)
    return meshlon, meshlat

def cut_domain_stereo(var_to_cut, map_lim_x, map_lim_y):
    var_cutted = var_to_cut.sel(x=var_to_cut.x.where(in_range(var_to_cut.x,map_lim_x),drop=True), y=var_to_cut.y.where(in_range(var_to_cut.y,map_lim_y),drop=True))
    return var_cutted

def weighted_mean(data, dims, weights):
    weight_sum = weights.sum(dim=dims) # to avoid dividing by zero
    return (data*weights).sum(dim=dims)/weight_sum.where(weight_sum != 0)

def create_stacked_mask(isfmask_2D, nisf_list, dims_to_stack, new_dim):
    # create stacked indices to select the different ice shelves
    # based on https://xarray.pydata.org/en/stable/indexing.html#more-advanced-indexing
    stacked_mask = isfmask_2D.stack(z=(dims_to_stack))
    
    new_coord_mask = stacked_mask.z.where(stacked_mask==nisf_list).dropna(how='all',dim='z')  
    new_coord_mask = new_coord_mask.rename({'z': new_dim})
    
    return new_coord_mask

def choose_isf(var, isf_stacked_mask, kisf):
    # choose a given ice shelf based on stacked coordinate
    return var.stack(mask_coord=['y','x']).sel(mask_coord=isf_stacked_mask.sel(Nisf=kisf).dropna('mask_coord'))

def bring_back_to_2D(stacked_da):
    # unstack to plot
    return stacked_da.unstack().sortby('y').sortby('x')