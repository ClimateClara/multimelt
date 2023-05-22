"""
This is a script to collect the masking steps with functions

@author: Clara Burgard
"""

import xarray as xr
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange
import cc3d
#from tqdm import tqdm, trange
import multimelt.plume_functions as pf
import multimelt.useful_functions as uf
import multimelt.box_functions as bf

def read_isfmask_info(infile):
    
    """
    Read the ice shelf limits and info from infile and convert it to an array.
    
    This function reads the ice shelf limits (min/max lon, min/max lat, ice shelf ID) from infile and coverts it to a np.array. This function is tailored to the format of ``lonlat_masks.txt``.
    
    Parameters
    ----------
    infile : str
        path to the csv-file containing the info. This function is tailored to the format of ``lonlat_masks.txt``.

    Returns
    -------
    res : np.array
        array containing minlon,maxlon,minlat,maxlat,is_nb
    """
    
    def_ismask = [ ]
    file1 = open(infile, 'r')
    Lines = file1.readlines()
    for ll in Lines:
        if ll[0] != '!':
            minlon = float(ll[18:25])
            maxlon = float(ll[43:51])
            if ll[51] == ')':
                minlat = -90
                maxlat = -50
                is_nb = int(ll[70:71])
                is_name = str(ll[73:100])
            else:
                minlat = float(ll[71:78])
                maxlat = float(ll[98:105])
                is_nb = int(ll[124:127])
                is_name = str(ll[130:200])
            def_ismask.append([minlon,maxlon,minlat,maxlat,is_nb])
    res = np.array(def_ismask)
    return res


def def_isf_mask(arr_def_ismask, file_msk, file_conc, lon, lat, FRIS_one=True, 
                 mouginot_basins=False, variable_geometry=False, connectivity = 4, threshold = 4):
    
    """
    Define a mask for the individual ice shelves. 
    
    This function defines a mask for the individual ice shelves. I think it works for both stereographic and latlon grids but I have not tried the latter.
    
    Parameters
    ----------
    arr_def_ismask : np.array
        Array containing minlon,maxlon,minlat,maxlat,is_nb or xr.Dataset with drainage basins
    file_msk : xr.DataArray
        Mask separating ocean (0), ice shelves (between 0 and 2, excluding 0 and 2), grounded ice (2) 
    file_conc : xr.DataArray
        Ice shelf concentration for each point (between 0 and 1)
    lon : xr.DataArray
        Longitude (depends on x,y for stereographic)
    lat : xr.DataArray
        Latitude (depends on x,y for stereographic)
    FRIS_one : Boolean 
        If True, Filchner-Ronne are considered as one ice-shelf
    mouginot_basins : Boolean 
        If True, arr_def_ismask is an xr.DataArray with drainage basins
    variable_geometry : Boolean 
        If True, arr_def_ismask
    connectivity : int
        4 or 8 for 2D, defines what is considered a "connected" point
    threshold : int
        Size of lonely pixel areas to remove
        
    Returns
    -------
    new_mask : xr.DataArray
        Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0
    """    
    
    if mouginot_basins:
        
        isf_mask = file_msk.copy()
        # only ice shelves
        isf_only_mask = file_conc > 0
        
        #find connected components
        dusted = cc3d.dust(isf_only_mask.values.astype(np.int64), 
                   threshold = threshold, 
                   connectivity = connectivity, 
                   in_place = False)
        
        labels_out = cc3d.connected_components(dusted, 
                                       connectivity = connectivity)
        
        labelled = xr.DataArray(labels_out, 
                        coords = {"y": file_conc.y, "x": file_conc.x}, 
                        dims = ["y", "x"],
                        name = "labels")
        
        # assign ID for basins
        isf_mask_basins = arr_def_ismask['ID_isf'].where(isf_only_mask > 0)
        # cut connected areas to area covered by basin stuff
        labelled_isf = labelled.where(np.isfinite(isf_mask_basins))
        
        # creating the mask
        new_mask = isf_mask_basins.copy()
        
        new_mask = new_mask.where(
            new_mask != 58, 57).where(
            new_mask != 151, 99).where(
            new_mask != 109, 107).where(
            new_mask != 116, 5).where(
            new_mask != 143, 97).where(
            new_mask != 137, 99)
        
                    
        arr_def_ismask['name_isf'].loc[{'Nisf': 57}] = 'Ross'
        arr_def_ismask['name_isf'].loc[{'Nisf': 58}] = np.nan
        
        if FRIS_one:
            new_mask = new_mask.where(new_mask != 104, 103)
            arr_def_ismask['name_isf'].loc[{'Nisf': 103}] = 'Filchner-Ronne'
            arr_def_ismask['name_isf'].loc[{'Nisf': 104}] = np.nan

        arr_def_ismask['name_isf'] = arr_def_ismask['name_isf'].dropna('Nisf')
        
        # do some fine-tuning for overlapping ice shelves   
        problem_regions = [2,3,8,9,10,13,23,26,27,28,29,32,34,38,44,46,50,57,59,60,
                   63,70,71,72,73,74,76,77,78,83,84,85,89,91,96,103]
        
        for conn_label in range(1,labels_out.max()):
            basins_conn_domain = arr_def_ismask['ID_isf'].where(labelled_isf == conn_label, drop=True)
            max_label = basins_conn_domain.max().values
            min_label = basins_conn_domain.min().values
            
            # for areas with two labels in problem regions, take the one with the most points
            if max_label != min_label:
                groups_isf = basins_conn_domain.groupby(basins_conn_domain)
                groups_labels = groups_isf.groups.keys()
                if groups_isf.count().ID_isf.count() > 1:
                    if any(x in problem_regions for x in list(groups_labels)):
                        #print(conn_label)
                        #print(min_label,max_label)
                        dominant_isf = groups_isf.count().idxmax().values
                        if dominant_isf == 12:
                            dominant_isf = 14
                        #print(dominant_isf)
                        new_mask = new_mask.where(labelled_isf != conn_label, dominant_isf)
            
        # other fine-tuning: if an ice shelf is split, keep the largest connected domain
        dx = abs(file_conc.x[1] - file_conc.x[0])
        dy = abs(file_conc.y[1] - file_conc.y[0])

        split_regions = [70,77,83,89,103] 

        for rreg in split_regions:
            # look where there are the same labels in several unconnected domains
            labels_same = list(new_mask.groupby(labelled_isf).groups) * (new_mask.groupby(labelled_isf).median() == rreg)
            labels_same = labels_same[labels_same>0]

            area_before = 0
            for conn_label in labels_same:
                # compute the area of the different unconnected areas
                conc_for_area = file_conc.where(labelled_isf == conn_label, drop=True)
                area_now = (conc_for_area * dx * dy).sum()
                if area_now >= area_before:
                    area_before = area_now
                    largest_label = conn_label

            # set the smaller areas to 159
            for small_label in (labels_same.where(labels_same != largest_label).dropna('labels')):
                new_mask = new_mask.where(labelled_isf != small_label, 159)

        new_mask = new_mask + 1
        new_mask_info = arr_def_ismask.copy()
        new_mask_info['Nisf'] = new_mask_info['Nisf'] + 1
        
        new_mask = new_mask.where(file_msk != 0, 1).where(file_msk != 2, 0)
    
    else:
        
        arr_def_general = arr_def_ismask[arr_def_ismask[:, 3] == -50]
        arr_def_detail = arr_def_ismask[arr_def_ismask[:, 3] != -50]

        isf_yes = (file_msk > 0) & (file_msk < 2)
        isf_mask = file_msk.copy()
        # is_mask0.plot()
        for i, mm in enumerate(arr_def_general):
            #print('general ' + str(i))
            isf_mask = isf_mask.where(~(uf.in_range(lon, mm[0:2]) & uf.in_range(lat, mm[2:4])), int(mm[4]))
        for i, mm in enumerate(arr_def_detail):
            #print('detail ' + str(i))
            isf_mask = isf_mask.where(~(uf.in_range(lon, mm[0:2]) & uf.in_range(lat, mm[2:4])), int(mm[4]))
        isf_mask = isf_mask.where(isf_yes)

        if FRIS_one:
            isf_mask = isf_mask.where(isf_mask != 21, 11) # Filchner (21) and Ronne (11) are combined
        
        remaining_isf = (isf_mask > 0) & (isf_mask <= 1)
        new_mask = isf_mask.where(~remaining_isf, 4)
        new_mask = new_mask.where(file_msk != 0, 1).where(file_msk != 2, 0)
        
        if variable_geometry:
            
            print('YOU CHOSE VARIABLE GEOMETRY SO I NEED TO WORK A BIT MORE')
            new_mask = new_mask.where(~((new_mask > 1) & (new_mask < 10)), 4)


            dx = abs(new_mask.x[1] - new_mask.x[0]).values.astype(int)
            dy = abs(new_mask.y[1] - new_mask.y[0]).values.astype(int)

            ### SPECIAL REGIONS
            new_mask = new_mask.where(new_mask != 102, 75)
            new_mask = new_mask.where(new_mask != 103, 75)
            new_mask = new_mask.where(new_mask != 114, 26)
            new_mask = new_mask.where(new_mask != 81, 22)

            print('I am separating splitted ice shelves')
            ###### THIS BLOCK IS TO SEPARATE SPLIT REGIONS
            threshold = 1
            connectivity = 4

            #find connected components
            dusted = cc3d.dust(new_mask.values.astype(np.int64), 
                       threshold = threshold, 
                       connectivity = connectivity, 
                       in_place = False)

            labels_out = cc3d.connected_components(dusted, 
                                           connectivity = connectivity)

            labelled_isf = xr.DataArray(labels_out, 
                            coords = {"y": file_conc.y, "x": file_conc.x}, 
                            dims = ["y", "x"],
                            name = "labels")

            all_isf_list = np.array(list(new_mask.groupby(new_mask).groups))
            isf_labels = all_isf_list[all_isf_list>9]

            for rreg in isf_labels:
                #print(rreg)
                # look is one ice shelf is present in disconnected regions
                isf_group = new_mask.where(new_mask==rreg)
                label_group = labelled_isf.where(np.isfinite(isf_group))
                label_group_list = np.array(list(label_group.groupby(label_group).groups))
                label_group_list = label_group_list[label_group_list > 1]
                if label_group_list.size > 0:
                    if label_group_list.min() != label_group_list.max():
                        area_before = 0
                        for conn_label in label_group_list:
                            # compute the area of the different unconnected areas
                            conc_for_area = file_conc.where(labelled_isf == conn_label, drop=True)
                            area_now = (conc_for_area * dx * dy).sum()
                            if area_now >= area_before:
                                area_before = area_now
                                largest_label = conn_label

                        # set the smaller areas to 4 
                        for llabel in label_group_list:
                            if llabel != largest_label:
                                new_mask = new_mask.where(labelled_isf != llabel, 4)

            print('I am filling ice-shelf regions that did not fit the initial limits')
            ###### THIS BLOCK IS TO FILL THE "NEWLY ICE SHELF REGIONS"

            #for n in range(4):
            threshold = 1
            connectivity = 4

            scattered_reg_all_conc = file_conc.where(new_mask == 4) 
            scattered_reg_all_mask = scattered_reg_all_conc > 0

            #find connected components
            dusted = cc3d.dust(scattered_reg_all_mask.values.astype(np.int64), 
                       threshold = threshold, 
                       connectivity = connectivity, 
                       in_place = False)

            labels_out_conc = cc3d.connected_components(dusted, 
                                           connectivity = connectivity)

            labelled = xr.DataArray(labels_out_conc, 
                            coords = {"y": file_conc.y, "x": file_conc.x}, 
                            dims = ["y", "x"],
                            name = "labels")

            # filter that checks the point around
            weights_filter = np.zeros((3,3))
            weights_filter[0,1] = 1
            weights_filter[1,0] = 1
            weights_filter[1,2] = 1
            weights_filter[2,1] = 1

            weights_da = xr.DataArray(data=weights_filter,dims=['y0','x0'])


            for conn_label in range(1,labels_out_conc.max()+1):
                dom_region = labelled.where(labelled == conn_label, drop=True)
                dom_bounds_plus1 = np.array([dom_region.x.min().values - dx,dom_region.x.max().values + dx,dom_region.y.min().values - dy,dom_region.y.max().values + dy]).astype(int)
                dom_plus1_mask = scattered_reg_all_mask.sel(x=range(dom_bounds_plus1[0],dom_bounds_plus1[1]+1,dx), y=range(dom_bounds_plus1[2],dom_bounds_plus1[3]+1,dy))
                corr = pf.xr_nd_corr_v2(dom_plus1_mask, weights_filter)
                only_contour = (corr ^ dom_plus1_mask)
                neighboring_pixels = new_mask.where(only_contour)
                if neighboring_pixels.max() > 9:
                    neighbor_max = neighboring_pixels.where(neighboring_pixels > 9).max()
                    neighbor_min = neighboring_pixels.where(neighboring_pixels > 9).min()
                    if neighbor_max == neighbor_min:
                        #print(neighbor_min.values)
                        new_mask = new_mask.where(labelled != conn_label, neighbor_min)
                    else:
                        isf_cont = neighboring_pixels.where(neighboring_pixels > 1)
                        isf_around = xr.DataArray(data=np.array(list(only_contour.groupby(isf_cont).groups))).assign_coords({'dim_0': np.array(list(only_contour.groupby(isf_cont).groups))})
                        count_isf = (isf_cont == isf_around).sum(['x','y'])
                        new_kisf = count_isf.dim_0.where(count_isf == count_isf.max(), drop=True).values[0]
                        new_mask = new_mask.where(labelled != conn_label, new_kisf)
    
    
    if mouginot_basins:
        mask_file = xr.merge([new_mask.rename('ISF_mask'), 
                              new_mask_info['name_isf'], 
                              new_mask_info['name_reg'], 
                              new_mask_info['Nisf_orig']])
    else:
        mask_file = new_mask
    
    return mask_file

def def_ground_mask(file_msk, dist, add_fac):

    """
    Define a mask for the Antarctic continent as such (not the islands). 
    
    This function defines the points that are part of the Antarctic continent as such (not the islands). 
    
    Parameters
    ----------
    file_msk : xr.DataArray
        Mask separating ocean (0), ice shelves (between 0 and 2, excluding 0 and 2), grounded ice (2)
    dist : int
        Defines the size of the starting square - should be small if the resolution is coarse and high if the resolution is fine. Default is currently 150 but you can play around. A good indicator to see if it is too high is if you see the small upper tail of the Ross ice shelf or if it is masked as ground.
    add_fac : int
       Defines additional iterations. Was introduced to get to the end of the Antarctic Peninsula, sometimes it would not get there otherwise. Current default is 100 but you are welcome to play around with it.
    Returns
    -------
    mask_ground : xr.DataArray
        Array showing the coverage of the Antarctic continent (0 for islands, 1 for ocean and ice shelves, 2 for mainland)
    """  
    
    mask_10 = file_msk.where(file_msk == 2, 0).where(file_msk != 2,1) #set all ice shelves and open ocean to 0, set all grounded ice to 1

    mask_gnd = mask_10.where(mask_10>0, drop=True)
    mask_gnd = mask_gnd.where(mask_gnd>0,0)
    
    meshx_gnd, meshy_gnd = np.meshgrid(mask_gnd.x,mask_gnd.y)
    meshx_gnd_da = mask_gnd.copy(data=np.broadcast_to(meshx_gnd, mask_gnd.shape))
    meshy_gnd_da = mask_gnd.copy(data=np.broadcast_to(meshy_gnd, mask_gnd.shape))
    
    dx = abs(meshx_gnd_da.x[2] - meshx_gnd_da.x[1])
    dy = abs(meshx_gnd_da.y[2] - meshx_gnd_da.y[1])
    
    max_len_xy = max(len(meshx_gnd_da.x),len(meshx_gnd_da.y))
    half_range = round(max_len_xy/2)
    
    mask_core = mask_gnd.where(~((uf.in_range(meshx_gnd_da, [-2*dist*dx,2*dist*dx]) # assuming that the South Pole is in the center of the projection
                          & uf.in_range(meshy_gnd_da, [-2*dist*dy,2*dist*dy]))), 5)
    
    # filter that checks the point around
    weights_filter = np.zeros((3,3))
    weights_filter[0,1] = 1
    weights_filter[1,0] = 1
    weights_filter[1,2] = 1
    weights_filter[2,1] = 1
    
    weights_da = xr.DataArray(data=weights_filter,dims=['y0','x0'])
    
    if 'time' in mask_gnd.dims:

        iter_list = []
        for tt,timet in enumerate(tqdm(mask_gnd.time)):

            mask_core_tt = mask_core.isel(time=tt)
            iter_mask = mask_core_tt.copy()
            for n in range(half_range+2*dist+add_fac):
                corr = pf.xr_nd_corr_v2(iter_mask, weights_filter)
                iter_mask = iter_mask.where(~((corr >= 5) & (mask_core_tt == 1)),5)

            iter_list.append(iter_mask)

        iter_mask_all = xr.concat(iter_list,dim='time')
        iter_mask_all = iter_mask_all.assign_coords({'time': mask_gnd.time})

        mask_ground = iter_mask_all.where(iter_mask_all !=5, 2).reindex_like(mask_10)
    
    else:
        
        iter_mask = mask_core.copy()
        for n in tqdm(range(half_range+2*dist+add_fac)):
            corr = pf.xr_nd_corr_v2(iter_mask, weights_filter)
            iter_mask = iter_mask.where(~((corr >= 5) & (mask_core == 1)),5)

        mask_ground = iter_mask.where(iter_mask !=5, 2).reindex_like(mask_10)
        
    mask_ground = mask_ground.where(mask_ground>0,0)
    
    return mask_ground

def def_grounding_line(new_mask, mask_ground, ground_point, add_fac, dx, dy):
    
    """
    Identify grounding line points and assign ice shelf ID to these points. 
    
    This function draws the grounding line of the different ice shelves. You can decide if they are the points on the ground or on the ice shelf side. 
    CAREFUL: I would recommend you choose setting ground_point = 'no'. In that case Alexander Island is taken into account. I have not been able to arrange that for ground_point = 'yes'.
    
    Parameters
    ----------
    new_mask : xr.DataArray
        Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0.
    mask_ground : xr.DataArray
        Array showing the coverage "mainland" Antarctica. Mainland is 2, islands are 1, all else is 0.
    ground_point : str
       ``yes`` or ``no``. If ``yes``, the grounding line is defined on the ground points at the border to the ice shelf. If ``no``, the grounding line is defined on the ice shelf points at the border to the ground.
    add_fac : int
       Defines additional iterations for the propagation for the ground mask. Was introduced to get to the end of the Antarctic Peninsula, sometimes it would not get there otherwise. Now useful to also propagate ground in Alexander Island.
    dx : float
        Grid size in x direction, step from left to right (can be positive or negative depending on the initial coordinate).
    dy : float
        Grid size in x direction, step from left to right (can be positive or negative depending on the initial coordinate).

    Returns
    -------
    mask_gline_final : xr.DataArray
        Array showing the grounding line with the ID of the corresponding ice shelf.
    """  
    
    if ground_point == 'yes': # grounding line is first cell with ground
        is_mask0 = new_mask.where(np.isnan(new_mask)==False,0)
        # this could be optimized with filters as well - this is using too much memory for large files so I should apply filters but not straightforward so needs more thinking
        shifted = xr.concat([is_mask0.shift(x=1), is_mask0.shift(x=-1), is_mask0.shift(y=1), is_mask0.shift(y=-1)], "shift")
        #sum of neighbours
        limits = abs(shifted).sum(dim='shift')
        #if sum is above 0, there is at least one ice shelf point near, so we take the max of all 4 (this is an "easy" solution)
        limits = limits.where(limits<=0,shifted.max(dim='shift'))
        #we cut out the points with numbers, on the continent
        mask_gline = limits.where(limits>0).where(mask_ground==2)
    
    else: 
        
        mask_10 = mask_ground.where(mask_ground==2, 0) * 0.5
        weights_neighbors = np.array(([0, 1, 0], [1, 1, 1], [0, 1, 0]))
        xr_weights = xr.DataArray(data=weights_neighbors, dims=['y', 'x'])

        if 'time' in mask_10.dims:

            iter_list = []
            for tt,timet in enumerate(tqdm(mask_10.time)):

                mask_10_tt = mask_10.isel(time=tt)
                xr_corr_neighbors = mask_10_tt.copy(data=pf.nd_corr(mask_10_tt,xr_weights))

                new_mask_tt = new_mask.isel(time=tt)
                mmask = (new_mask_tt>1).astype(int) + (xr_corr_neighbors>0).astype(int)
                cut_gline = xr_corr_neighbors.where(mmask==2)
                mask_gline_tt = new_mask_tt.where(cut_gline>0)#.load()

                iter_list.append(mask_gline_tt)

            mask_gline = xr.concat(iter_list,dim='time')
            mask_gline = mask_gline.assign_coords({'time': mask_10.time})
        
        else:
            
            xr_corr_neighbors = mask_10.copy(data=pf.nd_corr(mask_10,xr_weights))

            mmask = (new_mask>1).astype(int) + (xr_corr_neighbors>0).astype(int)
            cut_gline = xr_corr_neighbors.where(mmask==2)
            mask_gline = new_mask.where(cut_gline>0)#.load()
        
        #################
        # fix the problems around Alexander Island (ice shelves with grounding line only on the island)
        mask_gline_orig = mask_gline.copy()

        larger_region = new_mask.sel(x=np.arange(-2998000.,0.5,dx),y=np.arange(2998000.,0,dy))
        mask_10_isl = larger_region.where(larger_region == 0, 5).where(larger_region != 0, 1)
        mask_isl = mask_10_isl.where(mask_10_isl == 1, 0) #set all ice shelves and open ocean to 0, set all grounded ice to 1


        core = mask_isl.sel(x=np.arange(-1938000.,-1900000., dx),y=np.arange(718000.,680000., dy)).reindex_like(mask_isl)
        mask_core = mask_isl.where(np.isnan(core),5)

        # filter that checks the point around
        weights_filter = np.zeros((3,3))
        weights_filter[0,1] = 1
        weights_filter[1,0] = 1
        weights_filter[1,2] = 1
        weights_filter[2,1] = 1

        weights_da = xr.DataArray(data=weights_filter,dims=['y0','x0'])

        if 'time' in mask_10.dims:
            
            iter_list = []
            for tt,timet in enumerate(tqdm(mask_gline_orig.time)):

                    mask_core_tt = mask_core.isel(time=tt)
                    iter_mask = mask_core_tt.copy()
                    for n in range(add_fac):
                        corr = pf.xr_nd_corr_v2(iter_mask, weights_filter)
                        iter_mask = iter_mask.where(~((corr >= 5) & (mask_core_tt == 1)),5)

                    iter_list.append(iter_mask)

            iter_mask_all = xr.concat(iter_list,dim='time')
            iter_mask_all = iter_mask_all.assign_coords({'time': mask_gline_orig.time})

            mask_island = iter_mask_all.where(iter_mask_all !=5, 2)
            
        else:
            
            iter_mask = mask_core.copy()
            for n in range(add_fac):
                corr = pf.xr_nd_corr_v2(iter_mask, weights_filter)
                iter_mask = iter_mask.where(~((corr >= 5) & (mask_core == 1)),5)

            mask_island = iter_mask.where(iter_mask !=5, 2)
            
        mask_island = mask_island.where(mask_island>0,0)

        ##########################################

        mask_10_island = mask_island.where(mask_island==2, 0) * 0.5
        weights_neighbors = np.array(([0, 1, 0], [1, 1, 1], [0, 1, 0]))
        xr_weights = xr.DataArray(data=weights_neighbors, dims=['y', 'x'])
        
        if 'time' in mask_10.dims:
    
            iter_list = []
            for tt,timet in enumerate(tqdm(mask_10_island.time)):

                mask_10_island_tt = mask_10_island.isel(time=tt)
                xr_corr_neighbors = mask_10_island_tt.copy(data=pf.nd_corr(mask_10_island_tt,xr_weights))

                cut_gline_tt = xr_corr_neighbors.where((larger_region.isel(time=tt)>1) & (xr_corr_neighbors>0))

                iter_list.append(cut_gline_tt)

            cut_gline = xr.concat(iter_list,dim='time')
            cut_gline = cut_gline.assign_coords({'time': mask_10_island.time})
            
        else:
            
            xr_corr_neighbors = mask_10_island.copy(data=pf.nd_corr(mask_10_island,xr_weights))

            cut_gline = xr_corr_neighbors.where((larger_region>1) & (xr_corr_neighbors>0))
            
    
        mask_gline_new = larger_region.where(cut_gline>0).reindex_like(mask_gline_orig)
        mask_gline_final = mask_gline_orig.where(mask_gline_new != 54, mask_gline_new) # file_isf.Nisf.where(file_isf['isf_name'] == 'Bach', drop=True).values[0] (54)
        mask_gline_final = mask_gline_final.where(mask_gline_new != 75, mask_gline_new) # file_isf.Nisf.where(file_isf['isf_name'] == 'Wilkins', drop=True).values[0] (75)
        #mask_gline_final = mask_gline_final.where(mask_gline_new != 9, mask_gline_new) # only with Nico mask
        #mask_gline_final = mask_gline_final.where(mask_gline_new != 98, mask_gline_new) # only with Nico mask - Verdi
        #mask_gline_final = mask_gline_final.where(mask_gline_new != 99, mask_gline_new) # only with Nico mask - Brahms
        #mask_gline_final = mask_gline_final.where(mask_gline_new != 100, mask_gline_new) # only with Nico mask - Mendelssohn

    return mask_gline_final

def def_ice_front(new_mask, file_msk):

    """
    Identify ice front points and assign ice shelf ID to these points. 

    This function draws the ice front of the different ice shelves. They are the points on the ice shelf side.

    Parameters
    ----------
    new_mask : xr.DataArray
       Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0.
    file_msk : xr.DataArray
        Mask separating ocean (0), ice shelves (between 0 and 2, excluding 0 and 2), grounded ice (2)

    Returns
    -------
    mask_front : xr.DataArray
        Array showing the ice front with the ID of the corresponding ice shelf.
    """  

    is_mask0 = new_mask.where(np.isnan(new_mask)==False,0)
    # set all ice shelves to 3
    mask_front = file_msk.where((file_msk == 0) | (file_msk == 2), 3).copy()
    # check all directions and set points at border between ocean and ice shelf (0-3) to 5
    mask_front = mask_front.where((mask_front.shift(x=-1)-mask_front)!=-3,5)
    mask_front = mask_front.where((mask_front.shift(x=1)-mask_front)!=-3,5)
    mask_front = mask_front.where((mask_front.shift(y=-1)-mask_front)!=-3,5)
    mask_front = mask_front.where((mask_front.shift(y=1)-mask_front)!=-3,5)
    # cut out all front points
    mask_front = mask_front.where(mask_front==5)
    # set the ice shelf number
    mask_front = mask_front.where(mask_front!=5,is_mask0)
    
    return mask_front


def def_pinning_points(new_mask, lon, lat, mask_ground):
    
    """
    Identify pinning points and assign the negative value of the ice shelf ID to these points. 
    
    This function identifies islands within or at the limit of the different ice shelves. 
    
    Parameters
    ----------
    new_mask : xr.DataArray
       Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0.
    lon : xr.DataArray
        Longitude (depends on x,y for stereographic)
    lat : xr.DataArray
        Latitude (depends on x,y for stereographic)
    mask_ground : xr.DataArray
        Array showing the coverage of the Antarctic continent (0 for islands, 1 for ocean and ice shelves, 2 for mainland)

    Returns
    -------
    mask_pin : xr.DataArray
        Array showing the pinning points with the negative value of the ID of the corresponding ice shelf.
    """  
    
    is_mask = new_mask.where(new_mask>1)
    mask_pin = is_mask.where(np.isnan(is_mask)==False,0)
    half_range = round(len(is_mask.x)/2)

    for n in tqdm(range(25)):
        mask_pin = mask_pin.where((mask_pin.shift(x=1) >= 0) & (mask_pin>0),mask_pin.shift(x=1))
        mask_pin = mask_pin.where((mask_pin.shift(x=-1) >= 0) & (mask_pin>0),mask_pin.shift(x=-1))
        mask_pin = mask_pin.where((mask_pin.shift(y=1) >= 0) & (mask_pin>0),mask_pin.shift(y=1))
        mask_pin = mask_pin.where((mask_pin.shift(y=-1) >= 0) & (mask_pin>0),mask_pin.shift(y=-1))

    # manual adjustement for Gertz and Borchgrevink - from Nico
    mask_pin = mask_pin.where(~(uf.in_range(lon,[-130,-120]) & uf.in_range(lat,[-74.6,-50])),22)
    mask_pin = mask_pin.where(~(uf.in_range(lon,[23.5,24.1]) & uf.in_range(lat,[-70.39,-70.22])),71)

    mask_pin = mask_pin.where(mask_ground==1)*-1
    return mask_pin

def def_pinning_point_boundaries(mask_pin, new_mask):
    
    """
    Identify the boundaries of pinning points and assign the negative value of the ice shelf ID to these points. 
    
    This function draws the boundaries of the pinning points of the different ice shelves. 
    
    Parameters
    ----------
    mask_pin : xr.DataArray
        Array showing the pinning points with the negative value of the ID of the corresponding ice shelf.
    new_mask : xr.DataArray
       Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0.

    Returns
    -------
    mask_pin2 : xr.DataArray
        Array showing the boundaries of the pinning points with the negative value of the ID of the corresponding ice shelf.
    """  
    
    mask_pin2 = mask_pin.where(np.isnan(mask_pin),3)
    mask_pin2 = mask_pin2.where(new_mask<2,0)
    mask_pin2 = mask_pin2.where((mask_pin2.shift(x=-1)-mask_pin2)!=-3,5)
    mask_pin2 = mask_pin2.where((mask_pin2.shift(x=1)-mask_pin2)!=-3,5)
    mask_pin2 = mask_pin2.where((mask_pin2.shift(y=-1)-mask_pin2)!=-3,5)
    mask_pin2 = mask_pin2.where((mask_pin2.shift(y=1)-mask_pin2)!=-3,5)
    mask_pin2 = mask_pin2.where(mask_pin2==5)
    mask_pin2 = mask_pin2.where(mask_pin2!=5,-1*mask_pin)
    
    return mask_pin2

#latlonboundaryfile inputpath_metadata+'lonlat_masks.txt'
#whole_ds.to_netcdf(outputpath+'BedMachineAntarctica_2020-07-15_v02_reduced5_isf_masks_and_info.nc')
#outfile.to_netcdf(outputpath+'all_masks.nc','w')


def create_isf_masks(file_map, file_msk, file_conc, xx, yy, latlonboundary_file, outputpath, chunked, dx, dy, FRIS_one=True, mouginot_basins=False, variable_geometry=False, ground_point='yes', write_ismask = 'yes', write_groundmask = 'yes', dist=150, add_fac=100, connectivity=4, threshold=4):
 
    """
    Identify the location of ice shelves, Antarctic mainland, grounding lines, ice fronts and pinning points. 
    
    This function creates masks identifying the location of ice shelves, Antarctic mainland, grounding lines, ice fronts and pinning points
    
    Parameters
    ----------
    file_map : xr.Dataset
        Dataset containing information about the grid
    file_msk : xr.DataArray
        Mask separating ocean (0), ice shelves (between 0 and 2, excluding 0 and 2), grounded ice (2)
    file_conc : xr.DataArray
        Ice shelf concentration for each point (between 0 and 1)
    xx : xr.DataArray
        x-coordinates for the domain.
    yy : xr.DataArray
        y-coordinates for the domain.
    latlonboundary_file : str 
         Path to the csv-file containing the info. This function is tailored to the format of ``lonlat_masks.txt``. It takes the path to a netcdf-file if mouginot_basins is True.
    outputpath : str
        Path where the intermediate masks should be written to.
    chunked : int or False
        Size of chunks for dask when opening a netcdf into a dataset, if no need to chunk: False.
    dx : float
        Grid size in x direction, step from left to right (can be positive or negative depending on the initial coordinate).
    dy : float
        Grid size in x direction, step from left to right (can be positive or negative depending on the initial coordinate).
    FRIS_one : boolean
        True if Filchner-Ronne should be treated as one ice shelf, False if Filchner and Ronne should be separated.
    mouginot_basins : Boolean 
        If True, arr_def_ismask
    ground_point : str
        ``yes`` or ``no``. If ``yes``, the grounding line is defined on the ground points at the border to the ice shelf. If ``no``, the grounding line is defined on the ice shelf points at the border to the ground.
    write_ismask : str
        ``yes`` or ``no``. If ``yes``, compute the mask of the different ice shelves. If ``no``, read in the already existing file ``outputpath + 'preliminary_mask_file.nc'``.
    write_groundmask : str
        ``yes`` or ``no``. If ``yes``, compute the mask of mainland Antarctica. If ``no``, read in the already existing file ``outputpath + 'mask_ground.nc'``.
    dist : int
        Defines the size of the starting square for the ground mask - should be small if the resolution is coarse and high if the resolution is fine. Default is currently 150 but you can play around. A good indicator to see if it is too high is if you see the small upper tail of the Ross ice shelf or if it is masked as ground.
    add_fac : int
       Defines additional iterations for the propagation for the ground mask. Was introduced to get to the end of the Antarctic Peninsula, sometimes it would not get there otherwise. Current default is 100 but you are welcome to play around with it.
    connectivity : int
        4 or 8 for 2D, defines what is considered a "connected" point when looking at ice-shelves
    threshold : int
        Size of lonely pixel areas to remove for ice-shelf mask.


    Returns
    -------
    outfile : xr.Dataset
        Dataset containing all the produced masks: ISF_mask, GL_mask, IF_mask, PP_mask, ground_mask
    new_mask : xr.DataArray
        Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0.
    mask_gline : xr.DataArray
        Array showing the grounding line with the ID of the corresponding ice shelf.
    mask_front : xr.DataArray
        Array showing the ice front with the ID of the corresponding ice shelf.
    """  
    
    #####
    ##### COORDINATES
    #####

    print('handling coordinates')

    ### Create latlon coordinates
    meshx, meshy = np.meshgrid(xx,yy)
    meshlon,meshlat = uf.change_coord_stereo_to_latlon(meshx,meshy)
    file_map['longitude'] = (['y', 'x'],  meshlon)
    file_map['latitude'] = (['y', 'x'],  meshlat)
    lon = file_map['longitude']
    lat = file_map['latitude']

    #####
    ##### PREPARE THE NETCDF WITH ICE SHELF NUMBERS AND THEIR FRONT, GROUNDING LINE AND PINNING POINT
    #####

    ### Read in the latlon boundaries of the ice shelves

    print('Reading in latlon boundaries')
    if mouginot_basins:
        arr_def_ismask = xr.open_dataset(latlonboundary_file)
    else:
        arr_def_ismask = read_isfmask_info(latlonboundary_file)


    ### Define the regional masks

    print('Define the regional masks')

    if write_ismask == 'yes':
        new_mask = def_isf_mask(arr_def_ismask, file_msk, file_conc, lon, lat, 
                                FRIS_one, mouginot_basins, variable_geometry, connectivity, threshold)
        new_mask.to_netcdf(outputpath + 'preliminary_mask_file.nc', 'w')
        
        new_mask_info = new_mask
        if mouginot_basins:
            new_mask = new_mask['mask']
    else:
        print('read in from netcdf')
        if not chunked:
            new_mask_file = xr.open_mfdataset(outputpath + 'preliminary_mask_file.nc')
            #new_mask = new_mask_file['ls_mask012']
            new_mask = new_mask_file['mask']
        else:
            new_mask = xr.open_mfdataset(outputpath + 'preliminary_mask_file.nc', chunks={'x': chunked, 'y': chunked})
            #new_mask = new_mask_file['ls_mask012']
            new_mask = new_mask_file['mask']
    
    
    ### Find out what grounded ice is on the main continent (2) and what is islands and pinning points (0)

    print('Define ground_mask')

    if write_groundmask == 'yes':
        print('Distance to South Pole for initial ground square =',dist)
        print('Additional number of iterations =', add_fac)
        mask_ground = def_ground_mask(file_msk, dist, add_fac)
        mask_ground.to_netcdf(outputpath+'mask_ground.nc')
    else:
        print('read in from netcdf')
        if not chunked:
            mask_ground_file = xr.open_mfdataset(outputpath + 'mask_ground.nc')
            #mask_ground = mask_ground_file['ls_mask012']
            mask_ground = mask_ground_file['mask']
        else:
            mask_ground_file = xr.open_mfdataset(outputpath + 'mask_ground.nc', chunks={'x': chunked, 'y': chunked})
            #mask_ground = mask_ground_file['ls_mask012']
            mask_ground = mask_ground_file['mask']

    
    ### Define the grounding line

    print('Define grounding line')
    print('Grounding line is the first point on the ground:', ground_point)

    mask_gline = def_grounding_line(new_mask, mask_ground, ground_point, add_fac, dx, dy)

    ### Define the front

    print('Define front')

    mask_front = def_ice_front(new_mask, file_msk)
    
    ### Define the pinning points (to -1*ice shelf number)

    print('Define pinning points')

    mask_pin = def_pinning_points(new_mask, lon, lat, mask_ground)
    mask_pin.to_netcdf(outputpath+'mask_pinning_points.nc','w')

    ### Write out the contours of the pinning points (I have the feeling this part could be inferred earlier already
    ### but at least like this I am sure that it does what it should)
    print('Define pinning point boundaries')

    mask_pin2 = def_pinning_point_boundaries(mask_pin, new_mask)
    mask_pin2.to_netcdf(outputpath+'mask_pin_lines.nc','w')
        
    print('Merge into one netcdf')

    outfile = new_mask.to_dataset(name='ISF_mask')
    outfile['GL_mask'] = mask_gline
    outfile['IF_mask'] = mask_front
    outfile['PP_mask'] = mask_pin2
    outfile['ground_mask'] = mask_ground
    
#    outfile = xr.Dataset(
#                     {'ISF_mask': (new_mask.dims, new_mask.values),
#                     'GL_mask': (mask_gline.dims, mask_gline.values),
#                     'IF_mask': (mask_front.dims, mask_front.values),
#                    'PP_mask': (mask_pin2.dims, mask_pin2.values),
#                     'ground_mask': (mask_ground.dims, mask_ground.values),
#                     },
#                    coords = new_mask.coords)

    outfile['longitude'].attrs['standard_name'] = 'longitude'
    outfile['longitude'].attrs['units'] = 'degrees_east'
    outfile['longitude'].attrs['long_name'] = 'longitude coordinate'
    outfile['latitude'].attrs['standard_name'] = 'latitude'
    outfile['latitude'].attrs['units'] = 'degrees_north'
    outfile['latitude'].attrs['long_name'] = 'latitude coordinate'
    outfile['ISF_mask'].attrs['standard_name'] = 'ice shelf mask (0 for grounded, 1 for ocean, isf ID for ice shelves)'
    outfile['ISF_mask'].attrs['units'] = '-'
    outfile['GL_mask'].attrs['standard_name'] = 'grounding zone mask (isf ID for grounding line, NaN elsewhere)'
    outfile['GL_mask'].attrs['units'] = '-'
    outfile['IF_mask'].attrs['standard_name'] = 'ice-shelf front mask (isf ID for grounding line, NaN elsewhere)'
    outfile['IF_mask'].attrs['units'] = '-'
    outfile['PP_mask'].attrs['standard_name'] = 'pinning point zone mask (isf ID for pinning line, NaN elsewhere)'
    outfile['PP_mask'].attrs['units'] = '-'
    outfile['ground_mask'].attrs['standard_name'] = 'mainland vs islands mask (0 for islands, 1 for ocean and ice shelves, 2 for mainland)'
    outfile['ground_mask'].attrs['units'] = '-'

    # Global attributes
    outfile.attrs['history'] = 'Created with create_isf_masks() by C. Burgard'
    outfile.attrs['projection'] = 'Polar Stereographic South (71S,0E)'
    outfile.attrs['proj4'] = '+init=epsg:3031'
    outfile.attrs['Note'] = 'isf ID and individual isf characteristics can be found in ice_shelf_metadata_complete.csv'
    if mouginot_basins:
        return outfile, new_mask_info # new_mask, mask_gline, mask_front
    else:
        return outfile

def prepare_csv_metadata(file_metadata, file_metadata_GL_flux, file_conc, dx, dy, new_mask, FRIS_one, mouginot_basins):
    
    """
    Prepare the metadata info (ice shelf names and observed melt rates). 
    
    This function creates a dataframe with the metadata for each ice shelf (name, area, observed melt rates from Rignot)
    
    Parameters
    ----------
    file_metadata : str
        Path to ``iceshelves_metadata_Nico.txt``
    file_metadata_GL_flux : str
        Path to ``GL_flux_rignot13.csv``
    file_conc : float between 0 and 1
        Concentration of ice shelf in each grid cell
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
    new_mask : xr.DataArray
        Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0.
    FRIS_one : boolean
        True if Filchner-Ronne should be treated as one ice shelf, False if Filchner and Ronne should be separated.

    Returns
    -------
    df1 : pandas.DataFrame
        DataFrame containing the following columns for each ice shelf: columns=['isf_name', 'region', 'isf_melt', 'melt_uncertainty','isf_area_rignot']
    """  
    
    #if mouginot_basins:
    #    new_mask = new_mask['mask']
        
    ### Name, Area, Numbers from Rignot et al. 2013
    ismask_info = []
    file1 = open(file_metadata, 'r')
    Lines = file1.readlines()
    for ll in Lines:
        # print(ll)
        if ll[0] != '!':
            is_nb0 = int(ll[9:12])
            is_name = ll[17:36].strip()
            is_region = ll[56:75].strip()
            is_melt_rate = float(ll[94:100])
            is_melt_unc = float(ll[124:129])
            if ll[-2] == ';' or ll[132] == '!' or ll[132] == ' ':
                is_area = np.nan
            else:
                is_area = float(ll[148:155])
            if is_area == 1.:
                is_area = np.nan
                
            if mouginot_basins:
                if new_mask.Nisf.where(new_mask['Nisf_orig'] == is_nb0, drop=True):
                    is_nb1 = new_mask.Nisf.where(new_mask['Nisf_orig'] == is_nb0, drop=True).values[0].astype(int)
                    ismask_info.append([is_nb1, is_name, is_region, is_melt_rate, is_melt_unc, is_area])
                elif is_name == 'Ross':
                    is_nb1 = 59
                    ismask_info.append([is_nb1, is_name, is_region, is_melt_rate, is_melt_unc, is_area])
                elif is_name == 'Filchner':
                    is_nb1 = 105
                    ismask_info.append([is_nb1, is_name, is_region, is_melt_rate, is_melt_unc, is_area]) 
                elif is_name == 'Ronne':
                    is_nb1 = 104
                    ismask_info.append([is_nb1, is_name, is_region, is_melt_rate, is_melt_unc, is_area]) 
                #else:
                #    ismask_info.append([is_nb0, is_name, is_region, is_melt_rate, is_melt_unc, is_area])
            else:
                ismask_info.append([is_nb0, is_name, is_region, is_melt_rate, is_melt_unc, is_area])

    arr_ismask_info = np.array(ismask_info)
    
    df = pd.DataFrame(arr_ismask_info[:, 1:6], index=arr_ismask_info[:, 0].astype(int),
                      columns=['isf_name', 'region', 'isf_melt', 'melt_uncertainty',
                               'isf_area_rignot'])        

    df['isf_melt'] = df['isf_melt'].astype(float)
    df['melt_uncertainty'] = df['melt_uncertainty'].astype(float)
    df['isf_area_rignot'] = df['isf_area_rignot'].astype(float)
    
    if FRIS_one and not mouginot_basins:

        df['isf_name'].loc[11] = 'Filchner-Ronne'
        df['isf_melt'].loc[11] = df['isf_melt'].loc[11] + df['isf_melt'].loc[21]
        df['melt_uncertainty'].loc[11] = df['melt_uncertainty'].loc[11] + df['melt_uncertainty'].loc[21] # this might be a bit dodgy to add the uncertainties?
        df['isf_area_rignot'].loc[11] = df['isf_area_rignot'].loc[11] + df['isf_area_rignot'].loc[21] 
        df = df.drop(21)
    
    elif mouginot_basins:        
        
        df['isf_name'].loc[104] = 'Filchner-Ronne'
        df['isf_melt'].loc[104] = df['isf_melt'].loc[104] + df['isf_melt'].loc[105]
        df['melt_uncertainty'].loc[104] = df['melt_uncertainty'].loc[104] + df['melt_uncertainty'].loc[105] # this might be a bit dodgy to add the uncertainties?
        df['isf_area_rignot'].loc[104] = df['isf_area_rignot'].loc[104] + df['isf_area_rignot'].loc[105] 
        df = df.drop(105)
    
    ### Compute area from our data
    if mouginot_basins:
        is_mask = new_mask['mask'].where(new_mask['mask']>1)
    else:
        is_mask = new_mask.where(new_mask>1)
        
    idx_da = xr.DataArray(data=df.index, dims=['Nisf']).chunk({'Nisf':1})
    
    
    if 'time' not in file_conc.dims:    
        df['isf_area_here'] = file_conc.chunk({'x':1000,'y':1000}).where(is_mask == idx_da).sum(['x','y']) * abs(dx) * abs(dy) * 10 ** -6

    ### Correct melt numbers using our area
    df1 = df.sort_index()
        #print('here 1')
        
    if 'time' not in file_conc.dims: 
        df1['isf_melt'] = df1['isf_melt'] * df1['isf_area_here'] / df1['isf_area_rignot']
        #print('here 2')
        df1['melt_uncertainty'] = df1['melt_uncertainty'] * df1['isf_area_here'] / df1['isf_area_rignot']
        #print('here 3')
        df1['ratio_isf_areas'] = df1['isf_area_here'] / df1['isf_area_rignot']
        #print('here 4')
    
    # add grounding line flux as given by Rignot et al 2013
    GL_flux_pd = pd.read_csv(file_metadata_GL_flux, delimiter=';').dropna(how='all',axis=1).dropna(how='any',axis=0)
    df1['GL_flux'] = np.nan
    if not mouginot_basins:
        for idx_gl in GL_flux_pd.index:
            for idx in df1.index:
                if df1['isf_name'].loc[idx] == GL_flux_pd['Ice Shelf Name'].loc[idx_gl]:
                    df1['GL_flux'].loc[idx] = float(GL_flux_pd['Grounding line flux'].loc[idx_gl])

    return df1

    
def compute_dist_front_bot_ice(mask_gline, mask_front, file_draft, file_bed, df1, lon, lat, dx=False, dy=False, new_mask=False, file_conc=False):

    """
    Compute the depth of the bedrock and of the ice draft at the ice front. 
    
    This function computes the average and maximum depth of the bedrock and of the ice draft at the ice front.
    
    Parameters
    ----------
    mask_gline : xr.DataArray
        Array showing the grounding line with the ID of the corresponding ice shelf.
    mask_front : xr.DataArray
        Array showing the ice front with the ID of the corresponding ice shelf.
    file_draft : xr.DataArray
        Array containing the ice draft depth at at least at each ocean/ice shelf. Ice draft depth should be negative when below sea level!
    file_bed : xr.DataArray
        Array containing the bedrock topography at least at each ocean/ice shelf point. Bedrock depth should be negative when below sea level!
    df1 : pd.DataFrame
        DataFrame containing the following columns for each ice shelf: columns=['isf_name', 'region', 'isf_melt', 'melt_uncertainty','isf_area_rignot']
    lon : xr.DataArray
        Longitude (depends on x,y for stereographic)
    lat : xr.DataArray
        Latitude (depends on x,y for stereographic)
        
    Returns
    -------
    df1 : pd.DataFrame
        DataFrame containing the following columns for each ice shelf: columns=['isf_name', 'region', 'isf_melt', 'melt_uncertainty','isf_area_rignot'] AND ['front_bot_depth_max','front_bot_depth_avg','front_ice_depth_min','front_ice_depth_avg','front_min_lat','front_max_lat','front_min_lon','front_max_lon']
    """ 
    
    file_draft = file_draft.where(file_draft<0,0)
    
    idx_da = xr.DataArray(data=df1.index, dims=['Nisf']).chunk({'Nisf': 1})

    if 'time' in mask_front.dims:

        ds1 = xr.Dataset()
        
        is_mask = new_mask.where(new_mask>1)
        ds1['isf_area_here'] = file_conc.chunk({'x':1000,'y':1000}).where(is_mask == idx_da).sum(['x','y']) * abs(dx) * abs(dy) * 10 ** -6

        #print('here 7')
        ds1['front_bot_depth_max'] = -1*file_bed.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).min(['x','y'])
        ds1['front_bot_depth_avg'] = -1*file_bed.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).mean(['x','y'])
        ds1['front_ice_depth_min'] = -1*file_draft.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).max(['x','y'])
        ds1['front_ice_depth_avg'] = -1*file_draft.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).mean(['x','y'])

        #print('here 8')
        ds1['front_min_lat'] = lat.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).min(['x','y'])
        ds1['front_max_lat'] = lat.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).max(['x','y'])
        ds1['front_min_lon'] = lon.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).min(['x','y'])
        ds1['front_max_lon'] = lon.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).max(['x','y'])

        #print('here 9')
        # special treatment for Ross
        ds1['front_max_lon'].loc[dict({'Nisf': 10})] = lon.where(mask_front == 10).where(lon < -100).max()
        ds1['front_min_lon'].loc[dict({'Nisf': 10})] = lon.where(mask_front == 10).where(lon > 100).min()

        ds_clean = ds1.where(ds1['isf_area_here'] > 0, drop= True)
        
        return ds_clean

    else:

        #print('here 7')
        df1['front_bot_depth_max'] = -1*file_bed.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).min(['x','y'])
        df1['front_bot_depth_avg'] = -1*file_bed.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).mean(['x','y'])
        df1['front_ice_depth_min'] = -1*file_draft.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).max(['x','y'])
        df1['front_ice_depth_avg'] = -1*file_draft.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).mean(['x','y'])

        #print('here 8')
        df1['front_min_lat'] = lat.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).min(['x','y'])
        df1['front_max_lat'] = lat.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).max(['x','y'])
        df1['front_min_lon'] = lon.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).min(['x','y'])
        df1['front_max_lon'] = lon.chunk({'x': 1000, 'y': 1000}).where(mask_front==idx_da).max(['x','y'])

        #print('here 9')
        # special treatment for Ross
        df1['front_max_lon'].loc[10] = lon.where(mask_front == 10).where(lon < -100).max()
        df1['front_min_lon'].loc[10] = lon.where(mask_front == 10).where(lon > 100).min()

        #print('here 10')
        df_clean = df1[df1['isf_area_here'] > 0]
    
        return df_clean
    
def prepare_metadata(file_metadata, file_metadata_GL_flux, dx, dy, new_mask, mask_gline, mask_front, file_draft, file_bed, file_conc, lon, lat, outputpath, write_metadata = 'yes', FRIS_one=True, mouginot_basins=True):

    """
    Prepare the metadata info into a csv. 
    
    This function creates a panda DataFrame with all metadata (1D) info about the individual ice shelves.
    
    Parameters
    ----------
    file_metadata : str
        Path to ``iceshelves_metadata_Nico.txt``
    file_metadata_GL_flux : str
        Path to ``GL_flux_rignot13.csv``
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
    new_mask : xr.DataArray
        Array showing the coverage of each ice shelf with the respective ID, open ocean is 1, land is 0.
    mask_gline : xr.DataArray
        Array showing the grounding line with the ID of the corresponding ice shelf.
    mask_front : xr.DataArray
        Array showing the ice front with the ID of the corresponding ice shelf.
    file_draft : xr.DataArray
        Array containing the ice draft depth at at least at each ocean/ice shelf. Ice draft depth should be negative when below sea level!
    file_bed : xr.DataArray
        Array containing the bedrock topography at least at each ocean/ice shelf point. Bedrock depth should be negative when below sea level!
    file_conc : xr.DataArray, float between 0 and 1
        Concentration of ice shelf in each grid cell
    lon : xr.DataArray
        Longitude (depends on x,y for stereographic)
    lat : xr.DataArray
        Latitude (depends on x,y for stereographic)
    outputpath : str
        Path where the metadata csv-file should be written to.
    write_metadata : str
        ``yes`` or ``no``. If ``yes``, prepare the metadata csv-file. If ``no``, read in the already existing file ``outputpath + 'ice_shelf_metadata_complete.csv'``.
    FRIS_one : boolean
        True if Filchner-Ronne should be treated as one ice shelf, False if Filchner and Ronne should be separated.
        
    Returns
    -------
    df1 : pandas.DataFrame
        DataFrame containing the following columns for each ice shelf: columns=['isf_name', 'region', 'isf_melt', 'melt_uncertainty','isf_area_rignot','front_bot_depth_max','front_bot_depth_avg','front_ice_depth_min','front_ice_depth_avg','front_min_lat','front_max_lat','front_min_lon','front_max_lon']
    ds1 : xr.Dataset
        xarray Dataset containing ['front_bot_depth_max','front_bot_depth_avg','front_ice_depth_min','front_ice_depth_avg','front_min_lat','front_max_lat','front_min_lon','front_max_lon'] if time dependent geometry.
    """  

    #####
    ##### PREPARE THE CSV WITH THE METADATA OF EACH ICE SHELF
    #####

    print('Prepare csv with metadata')

    ### Name, Area, Numbers from Rignot et al. 2013
    if write_metadata == 'yes':
        df1 = prepare_csv_metadata(file_metadata, file_metadata_GL_flux, file_conc, dx, dy, new_mask, 
                                   FRIS_one, mouginot_basins) 
        if 'time' not in new_mask.dims:
            df1 = compute_dist_front_bot_ice(mask_gline, mask_front, file_draft, file_bed, df1, lon, lat)
        df1.to_csv(outputpath + "ice_shelf_metadata_complete.csv")
        
        if 'time' in new_mask.dims:
            ds1 = compute_dist_front_bot_ice(mask_gline, mask_front, file_draft, file_bed, df1, lon, lat, dx, dy, new_mask, file_conc)
            ds1.to_netcdf(outputpath + "ice_shelf_metadata_time_stuff.nc")
        else:
            ds1 = None

    else:
        print('read complete metadata from csv')
        df1 = pd.read_csv(outputpath + "ice_shelf_metadata_complete.csv", index_col=0)
        if 'time' in new_mask.dims:
            ds1 = xr.open_dataset(outputpath + "ice_shelf_metadata_time_stuff.nc")
        else:
            ds1 = None
            
    return df1, ds1

def combine_mask_metadata(df1, outfile, ds_time=False):

    """
    Combine the metadata and the mask info into one netcdf. 
    
    This function combines the metadata and the mask info into one netcdf.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame containing 1D info for each ice shelf
    outfile : xr.Dataset
        Dataset containing all the produced masks: ISF_mask, GL_mask, IF_mask, PP_mask, ground_mask
    ds_time : xr.Dataset
        Dataset containing metadata for each time step in the case there is a time evolution of the geometry
        
    Returns
    -------
    whole_ds : xr.Dataset
        Dataset containing all information from df1 and outfile 
    """  
    
    df_ds = df1.to_xarray()
    whole_ds = outfile.merge(df_ds)
    whole_ds = whole_ds.rename({"index": "Nisf"})
    
    if 'time' in outfile.dims:
        whole_ds = whole_ds.merge(ds_time)

    whole_ds['Nisf'].attrs['standard_name'] = 'ice shelf ID'
    whole_ds['isf_name'].attrs['standard_name'] = 'ice shelf name'
    whole_ds['region'].attrs['standard_name'] = 'region'
    whole_ds['region'].attrs['long_name'] = 'Region name'
    whole_ds['isf_melt'].attrs['standard_name'] = 'ice shelf melt'
    whole_ds['isf_melt'].attrs['units'] = 'Gt/yr'
    whole_ds['melt_uncertainty'].attrs['standard_name'] = 'melt uncertainty'
    whole_ds['melt_uncertainty'].attrs['units'] = 'Gt/yr'
    whole_ds['isf_area_rignot'].attrs['standard_name'] = 'ice shelf area Rignot'
    whole_ds['isf_area_rignot'].attrs['units'] = 'km^2'
    whole_ds['isf_area_rignot'].attrs['long_name'] = 'Ice shelf area in Rignot et al 2013'
    whole_ds['isf_area_here'].attrs['standard_name'] = 'our ice shelf area'
    whole_ds['isf_area_here'].attrs['units'] = 'km^2'
    whole_ds['isf_area_here'].attrs['long_name'] = 'Ice shelf area computed from our mask'
    #whole_ds['ratio_isf_areas'].attrs['standard_name'] = 'ratio isf area here/Rignot'
    #whole_ds['ratio_isf_areas'].attrs['units'] = '-'
    #whole_ds['ratio_isf_areas'].attrs['long_name'] = 'Ratio between ice shelf area computed from our mask and area from Rignot et al 2013'
    whole_ds['GL_flux'].attrs['standard_name'] = 'grounding_line_flux'
    whole_ds['GL_flux'].attrs['units'] = 'Gt/yr'
    whole_ds['GL_flux'].attrs['long_name'] = 'Flux across grounding line from Rignot et al 2013'
    whole_ds['front_bot_depth_max'].attrs['standard_name'] = 'max depth between isf front and ocean bottom'
    whole_ds['front_bot_depth_max'].attrs['units'] = 'm'
    whole_ds['front_bot_depth_max'].attrs['long_name'] = 'Maximum depth between the ice shelf front and ocean bottom'
    whole_ds['front_bot_depth_avg'].attrs['standard_name'] = 'avg depth between isf front and ocean bottom'
    whole_ds['front_bot_depth_avg'].attrs['units'] = 'm'
    whole_ds['front_bot_depth_avg'].attrs['long_name'] = 'Average depth between the ice shelf front and ocean bottom'
    whole_ds['front_ice_depth_min'].attrs['standard_name'] = 'min distance between sea surface and isf front depth'
    whole_ds['front_ice_depth_min'].attrs['units'] = 'm'
    whole_ds['front_ice_depth_min'].attrs['long_name'] = 'Minimum distance between sea surface and ice shelf front depth'
    whole_ds['front_ice_depth_avg'].attrs['standard_name'] = 'avg distance between sea surface and isf front depth'
    whole_ds['front_ice_depth_avg'].attrs['units'] = 'm'
    whole_ds['front_ice_depth_avg'].attrs['long_name'] = 'Average distance between sea surface and ice shelf front depth'
    whole_ds['front_min_lat'].attrs['standard_name'] = 'Min latitude isf front'
    whole_ds['front_min_lat'].attrs['units'] = 'degrees_north'
    whole_ds['front_min_lat'].attrs['long_name'] = 'Minimum latitude of the ice shelf front'
    whole_ds['front_max_lat'].attrs['standard_name'] = 'Max latitude isf front'
    whole_ds['front_max_lat'].attrs['units'] = 'degrees_north'
    whole_ds['front_max_lat'].attrs['long_name'] = 'Maximum latitude of the ice shelf front'
    whole_ds['front_min_lon'].attrs['standard_name'] = 'Min longitude isf front'
    whole_ds['front_min_lon'].attrs['units'] = 'degrees_east'
    whole_ds['front_min_lon'].attrs['long_name'] = 'Minimum longitude of the ice shelf front'
    whole_ds['front_max_lon'].attrs['standard_name'] = 'Max longitude isf front'
    whole_ds['front_max_lon'].attrs['units'] = 'degrees_east'
    whole_ds['front_max_lon'].attrs['long_name'] = 'Maximum longitude of the ice shelf front'

    # Global attributes
    whole_ds.attrs['history'] = 'Created with combine_mask_metadata() by C. Burgard'
    whole_ds.attrs['projection'] = 'Polar Stereographic South (71S,0E)'
    whole_ds.attrs['proj4'] = '+init=epsg:3031'
    return whole_ds

def compute_distance_GL_IF_ISF(whole_ds):

    """
    Compute the distance from each point to the grounding line and the ice front. 
    
    This function computes the distance between each point and the grounding line on the one hand and the ice front on the other hand.
    
    Parameters
    ----------
    whole_ds : xr.Dataset
        Dataset containing at least ``'ISF_mask'``, ``'GL_mask'``, ``'IF_mask'``
        
    Returns
    -------
    whole_ds : xr.Dataset
       whole_ds extended with the variables ``'dGL'``, ``'dIF'`` and ``'dGL_dIF'``
    """      
    
    ######
    ###### DO THE COMPUTATION AND ADD THE DISTANCE TO THE DATASET
    ######

    whole_ds['dGL'] = whole_ds['ISF_mask'] * np.nan
    whole_ds['dIF'] = whole_ds['ISF_mask'] * np.nan
    whole_ds['dGL_dIF'] = whole_ds['ISF_mask'] * np.nan
    
    

    for kisf in tqdm(whole_ds['Nisf']):

        
            
        if 'time' in whole_ds.dims:
            
            whole_ds0 = whole_ds.isel(time=0)

            if ~np.isnan(whole_ds0['GL_mask'].where(whole_ds0['GL_mask'] == kisf).max()):

                #print(whole_ds.dims)

                iter_list = []
                for tt,timet in enumerate(whole_ds.time):

                    whole_ds_tt = whole_ds.isel(time=tt).copy()

                    domain = whole_ds_tt['ISF_mask'].where(whole_ds_tt['ISF_mask'] >= 0)
                    isf_area = whole_ds_tt['ISF_mask'].where(whole_ds_tt['ISF_mask'] == kisf).dropna('x', how='all').dropna('y', how='all')
                    isf_gl = whole_ds_tt['GL_mask'].where(whole_ds_tt['GL_mask'] == kisf).dropna('x', how='all').dropna('y', how='all')
                    isf_if = whole_ds_tt['IF_mask'].where(whole_ds_tt['IF_mask'] == kisf).dropna('x', how='all').dropna('y', how='all')

                    # Compute distance from grounding line
                    xr_dist_to_gl = bf.distance_isf_points_from_line(domain, isf_area, isf_gl)
                    whole_ds_tt['dGL'] = whole_ds_tt['dGL'].where(np.isnan(xr_dist_to_gl), xr_dist_to_gl)

                    # Compute distance from ice front (for ice shelf points)
                    xr_dist_to_if = bf.distance_isf_points_from_line(domain, isf_area, isf_if)
                    whole_ds_tt['dIF'] = whole_ds_tt['dIF'].where(np.isnan(xr_dist_to_if), xr_dist_to_if)

                    domain_gl = whole_ds_tt['GL_mask'].where(whole_ds_tt['GL_mask'] >= 0)
                    # Compute distance from ice front (for grounding line)
                    xr_dist_dgl_if = bf.distance_isf_points_from_line(domain_gl, isf_gl, isf_if)
                    whole_ds_tt['dGL_dIF'] = whole_ds_tt['dGL_dIF'].where(np.isnan(xr_dist_dgl_if), xr_dist_dgl_if)

                    iter_list.append(whole_ds_tt.chunk({'x': 500, 'y': 500}))

                whole_ds_new = xr.concat(iter_list, dim='time')
                whole_ds_new = whole_ds_new.assign_coords({'time': whole_ds.time})
                whole_ds = whole_ds_new

        else:
            
            if ~np.isnan(whole_ds['GL_mask'].where(whole_ds['GL_mask'] == kisf).max()):

                domain = whole_ds['ISF_mask'].where(whole_ds['ISF_mask'] >= 0)
                isf_area = whole_ds['ISF_mask'].where(whole_ds['ISF_mask'] == kisf).dropna('x', how='all').dropna('y',
                                                                                                                  how='all')
                isf_gl = whole_ds['GL_mask'].where(whole_ds['GL_mask'] == kisf).dropna('x', how='all').dropna('y', how='all')
                isf_if = whole_ds['IF_mask'].where(whole_ds['IF_mask'] == kisf).dropna('x', how='all').dropna('y', how='all')

                # Compute distance from grounding line
                xr_dist_to_gl = bf.distance_isf_points_from_line(domain, isf_area, isf_gl)
                whole_ds['dGL'] = whole_ds['dGL'].where(np.isnan(xr_dist_to_gl), xr_dist_to_gl)

                # Compute distance from ice front (for ice shelf points)
                xr_dist_to_if = bf.distance_isf_points_from_line(domain, isf_area, isf_if)
                whole_ds['dIF'] = whole_ds['dIF'].where(np.isnan(xr_dist_to_if), xr_dist_to_if)

                domain_gl = whole_ds['GL_mask'].where(whole_ds['GL_mask'] >= 0)
                # Compute distance from ice front (for grounding line)
                xr_dist_dgl_if = bf.distance_isf_points_from_line(domain_gl, isf_gl, isf_if)
                whole_ds['dGL_dIF'] = whole_ds['dGL_dIF'].where(np.isnan(xr_dist_dgl_if), xr_dist_dgl_if)

    ######
    ###### UPDATE ATTRIBUTES
    ######

    # Variable attributes
    whole_ds['dGL'].attrs['units'] = 'm'
    whole_ds['dGL'].attrs['long_name'] = 'Shortest distance to respective grounding line'
    whole_ds['dGL'].attrs['standard_name'] = 'distance_to_grounding_line'
    whole_ds['dIF'].attrs['units'] = 'm'
    whole_ds['dIF'].attrs['long_name'] = 'Shortest distance to respective ice shelf front (ice shelf points)'
    whole_ds['dIF'].attrs['standard_name'] = 'distance_to_isf_front'
    whole_ds['dGL_dIF'].attrs['units'] = 'm'
    whole_ds['dGL_dIF'].attrs['long_name'] = 'Shortest distance to respective ice shelf front (grounding line)'
    whole_ds['dGL_dIF'].attrs['standard_name'] = 'distance_gl_to_isf_front'

    # Global attributes
    whole_ds.attrs['history'] = 'Created with combine_mask_metadata() by C. Burgard. dGL, dIF and dGL_dIF added by compute_distance_GL_IF_ISF()'
    
    return whole_ds


def create_mask_and_metadata_isf(file_map, file_bed, file_msk, file_draft, file_conc, chunked, latlonboundary_file, outputpath, file_metadata, file_metadata_GL_flux, ground_point, FRIS_one=True, mouginot_basins=False, variable_geometry=False, write_ismask = 'yes', write_groundmask = 'yes', write_outfile='yes', dist=150, add_fac=100, connectivity=4, threshold=4, write_metadata = 'yes'):
    
    """
    Create mask and metadata file for all ice shelves. 
    
    This function creates a dataset containing masks and metadata for the ice shelves in Antarctica. The input must be on a stereographic grid centered around Antarctica. 
    
    Parameters
    ----------
    file_map : xr.Dataset
        Dataset containing information about the grid
    file_bed : xr.DataArray
        Array containing the bedrock topography at least at each ocean/ice shelf point. Bedrock depth should be negative when below sea level!
    file_msk : xr.DataArray
        Mask separating ocean (0), ice shelves (3), grounded ice (2) (also can contain ice free land (1), and lake Vostok (4)).
    file_draft : xr.DataArray
        Array containing the ice draft depth at at least at each ocean/ice shelf. Ice draft depth should be negative when below sea level!
    file_conc : float between 0 and 1
        Concentration of ice shelf in each grid cell
    chunked : int or False
        Size of chunks for dask when opening a netcdf into a dataset, if no need to chunk: False.
    latlonboundary_file : str
         Path to the csv-file containing the info. This function is tailored to the format of ``lonlat_masks.txt``.
    outputpath : str
        Path where the intermediate files should be written to.
    file_metadata : str
        Path to ``iceshelves_metadata_Nico.txt``
    file_metadata_GL_flux : str
        Path to ``GL_flux_rignot13.csv``
    ground_point : str
        ``yes`` or ``no``. If ``yes``, the grounding line is defined on the ground points at the border to the ice shelf. If ``no``, the grounding line is defined on the ice shelf points at the border to the ground.
    FRIS_one : boolean
        True if Filchner-Ronne should be treated as one ice shelf, False if Filchner and Ronne should be separated.
    write_ismask : str
        ``yes`` or ``no``. If ``yes``, compute the mask of the different ice shelves. If ``no``, read in the already existing file ``outputpath + 'preliminary_mask_file.nc'``.
    write_groundmask : str
        ``yes`` or ``no``. If ``yes``, compute the mask of mainland Antarctica. If ``no``, read in the already existing file ``outputpath + 'mask_ground.nc'``.
    write_outfile : str
        ``yes`` or ``no``. If ``yes``, go through the mask file. If ``no``, read in the already existing file ``outputpath + 'outfile.nc'``.
    dist : int
        Defines the size of the starting square for the ground mask - should be small if the resolution is coarse and high if the resolution is fine. Default is currently 150 but you can play around. A good indicator to see if it is too high is if you see the small upper tail of the Ross ice shelf or if it is masked as ground.
    add_fac : int
       Defines additional iterations for the propagation for the ground mask. Was introduced to get to the end of the Antarctic Peninsula, sometimes it would not get there otherwise. Current default is 100 but you are welcome to play around with it.
    write_metadata : str
        ``yes`` or ``no``. If ``yes``, prepare the metadata csv-file. If ``no``, read in the already existing file ``outputpath + 'ice_shelf_metadata_complete.csv'``.
         
    Returns
    -------
    whole_ds : xr.Dataset
       Dataset summarizing all masks and info needed to prepare and use the parametrizations.
    """ 
    
    xx = file_map['x']
    yy = file_map['y']
    dx = xx[2] - xx[1]
    dy = yy[2] - yy[1]
    
    print('--------- PREPARE THE MASKS --------------')
    if write_outfile == 'yes':
        if mouginot_basins:
            outfile, new_mask_info = create_isf_masks(file_map, file_msk, file_conc, xx, yy, latlonboundary_file, outputpath, chunked, dx, dy, FRIS_one, mouginot_basins, variable_geometry, ground_point, write_ismask, write_groundmask, dist, add_fac, connectivity, threshold)
            outfile.to_netcdf(outputpath + 'outfile.nc', 'w')
            new_mask_info.to_netcdf(outputpath + 'new_mask_info.nc', 'w')
        else:
            outfile = create_isf_masks(file_map, file_msk, file_conc, xx, yy, latlonboundary_file, outputpath, chunked, dx, dy, FRIS_one, mouginot_basins, variable_geometry, ground_point, write_ismask, write_groundmask, dist, add_fac, connectivity, threshold)
            outfile.to_netcdf(outputpath + 'outfile.nc', 'w')
    else:
        outfile = xr.open_dataset(outputpath + 'outfile.nc')
        if mouginot_basins:
            new_mask_info = xr.open_dataset(outputpath + 'new_mask_info.nc')
    
    print('--------- PREPARE THE METADATA --------------')
    if 'time' in outfile.dims:
        df1, ds1 = prepare_metadata(file_metadata, file_metadata_GL_flux, dx, dy, outfile['ISF_mask'], outfile['GL_mask'], outfile['IF_mask'], file_draft, file_bed, file_conc, outfile['longitude'], outfile['latitude'], outputpath, write_metadata, FRIS_one)
    else:
        if mouginot_basins:
            df1, _ = prepare_metadata(file_metadata, file_metadata_GL_flux, dx, dy, new_mask_info, outfile['GL_mask'], outfile['IF_mask'], file_draft, file_bed, file_conc, outfile['longitude'], outfile['latitude'], outputpath, write_metadata, FRIS_one, mouginot_basins)
        else:
            df1, _ = prepare_metadata(file_metadata, file_metadata_GL_flux, dx, dy, outfile['ISF_mask'], outfile['GL_mask'], outfile['IF_mask'], file_draft, file_bed, file_conc, outfile['longitude'], outfile['latitude'], outputpath, write_metadata, FRIS_one, mouginot_basins)


    print('--------- COMBINE MASK AND METADATA --------------')
    if 'time' in outfile.dims:
        whole_ds = combine_mask_metadata(df1, outfile, ds1)
    else:
        whole_ds = combine_mask_metadata(df1, outfile)
    print('--------- COMPUTE DISTANCE TO GROUNDING LINE AND ICE FRONT --------------')
    whole_ds = compute_distance_GL_IF_ISF(whole_ds)
    
    return whole_ds






