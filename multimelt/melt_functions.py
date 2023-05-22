import numpy as np
from multimelt.constants import *
import multimelt.useful_functions as uf
import xarray as xr
import time
from tqdm.notebook import trange, tqdm
from numpy.polynomial import Polynomial

def freezing_temperature(salinity,depth):
    
    """
    Compute freezing temperature.
    
    This function computes the melting-freezing point Tf at the interface between the ocean and the ice-shelf
    basal surface. Formula and coefficients taken from Favier et al., 2019. Depth must be negative below sea level!
    
    Parameters
    ----------
    salinity : scalar or array
        Practical salinity in psu
    depth : scalar or array
        Depth in m, must be negative below sea level

    Returns
    -------
    T_f : scalar or array
        Melting-freezing point Tf at the interface between the ocean and the ice-shelf basal surface
    """
    
    l_1 = -0.0575 #°C per psu; Reese et al: -0.0572
    l_2 = 0.0832 #°C; Reese et al: 0.0788
    l_3 = 7.59*10**-4 #°C per m; Reese et al: 7.77 * 10**-8 (per Pa)
    T_f = l_1 * salinity + l_2 + l_3 * depth
    return T_f

def find_closest_depth_levels(depth_levels,depth_of_int):
    
    """
    Find closest depth levels in an array to a given depth.
    
    This function looks for nearest neighbors of a given depth in an array of depth levels. Depth must be positive!
    
    Parameters
    ----------
    depth_levels : xarray.DataArray
        DataArray cointaining the dimension 'depth' with the depth levels you want to localise your depth in. Depth must be positive!
    depth_of_int : xarray.DataArray
        Depth of interest (can be more than 1D). Depth must be positive!

    Returns
    -------
    lev_inf : xarray.DataArray
        nearest depth level below depth of interest
    lev_sup : xarray.DataArray
        nearest depth level above depth of interest
    weight_inf : xarray.DataArray
        weight to nearest depth level below depth of interest
    weight_sup : xarray.DataArray
        weight to nearest depth level above depth of interest
    """
    
    depth_diff = depth_levels - depth_of_int
    lev_inf = depth_levels.where(depth_diff>0,0).min('depth')
    lev_sup = depth_levels.where(depth_diff<=0,0).max('depth')
    weight_inf = abs(lev_sup - depth_of_int)/abs(lev_inf-lev_sup)
    weight_sup = abs(depth_of_int - lev_inf)/abs(lev_inf-lev_sup)
    
    # set to 0 to avoid nan-selecting later
    lev_inf = lev_inf.where(lev_inf>0,0)
    lev_sup = lev_sup.where(lev_sup>0,0)

    return lev_inf, lev_sup, weight_inf, weight_sup

def interp_from_levels_to_depth_of_int(var, lev_inf, lev_sup, weight_inf, weight_sup):
    
    """
    Interpolate a variable from the nearest neighbours temperature.
    
    This function computes from a variable weighted from the nearest neighbours 
    of a given depth in an array of depth levels. Depth must be positive!
    
    Parameters
    ----------
    var : xarray.DataArray
        DataArray of the variable of interest cointaining the dimension 'depth'. Depth must be positive!
    lev_inf : xarray.DataArray
        nearest depth level below depth of interest
    lev_sup : xarray.DataArray
        nearest depth level above depth of interest
    weight_inf : xarray.DataArray
        weight to nearest depth level below depth of interest
    weight_sup : xarray.DataArray
        weight to nearest depth level above depth of interest

    Returns
    -------
    new_var: xarray.DataArray
        interpolated value
    """
    
    new_var = weight_inf * var.sel(depth=lev_inf) + weight_sup * var.sel(depth=lev_sup)
    return new_var

def convert_m_ice_to_m_water(melt_rate_m_ice, grid_cell_area):
    
    """
    Convert m ice in m water equivalent.
    
    This function converts an ice thickness into a water equivalent height. 
    
    Parameters
    ----------
    melt_rate_m_ice : scalar or array
        Quantity in m of ice (e.g. melt rate)
    grid_cell_area : scalar or array 
        Area of the grid cells in :math:`\mathtt{m}^2`. if array: same shape as melt_rate_m_ice.

    Returns
    -------
    melt_rate_m_w : scalar or array
        Quantity in m of water equivalent (e.g. melt rate)
    """
    
    melt_rate_kg = melt_rate_m_ice * grid_cell_area * rho_i
    melt_rate_m3_w = melt_rate_kg * (1/rho_fw)
    melt_rate_m_w = melt_rate_m3_w / grid_cell_area
    return melt_rate_m_w

def convert_m_ice_to_Gt(melt_rate_m_ice, grid_cell_area):
    
    """
    Convert m ice in Gt.
    
    This function converts an ice thickness into a mass in Gt. 
    
    Parameters
    ----------
    melt_rate_m_ice : scalar or array
        Quantity in m of ice (e.g. melt rate)
    grid_cell_area : scalar or array 
        Area of the grid cells in :math:`\mathtt{m}^2`. if array: same shape as melt_rate_m_ice.

    Returns
    -------
    melt_rate_m_Gt : scalar or array
        Quantity in Gt (e.g. melt rate)
    """
        
    melt_rate_kg = melt_rate_m_ice * grid_cell_area * rho_i
    melt_rate_Gt = melt_rate_kg * 10**-12 ##10-9 Gt => t 10-3 t => kg
    return melt_rate_Gt

def convert_kg_ice_per_m2_to_Gt(melt_rate_kg_ice_per_m2, grid_cell_area):
    
    """
    Convert kg per m2 ice in Gt.
    
    This function converts an ice kg per m2 into a mass in Gt. 
    
    Parameters
    ----------
    melt_rate_kg_ice_per_m2 : scalar or array
        Quantity in kg of ice per :math:`\mathtt{m}^2` (e.g. melt rate)
    grid_cell_area : scalar or array 
        Area of the grid cells in :math:`\mathtt{m}^2`. if array: same shape as melt_rate_m_ice.

    Returns
    -------
    melt_rate_m_Gt : scalar or array
        Quantity in Gt (e.g. melt rate)
    """
        
    melt_rate_Gt = melt_rate_kg_ice_per_m2 / rho_i * grid_cell_area * 10**-12 ##10-9 Gt => t 10-3 t => kg
    return melt_rate_Gt


def linear_local_param(gamma, melt_factor, thermal_forcing):
    
    """
    Apply the linear local parametrization.
    
    This function computes the basal melt based on a linear local parametrization (see Favier et al., 2019 or Beckmann and Goosse, 2003).
    
    Parameters
    ----------
    gamma : scalar
        Heat exchange velocity in m per second
    melt_factor : scalar 
        Melt factor representing (rho_sw*c_pw) / (rho_i*L_i) in :math:`\mathtt{K}^{-1}`.
    thermal_forcing: scalar or array
        Difference between T and the freezing temperature Tf (T-Tf) in K or degrees C

    Returns
    -------
    melt : scalar or array
        Melt rate in m ice per second
    """
    
    melt = gamma * melt_factor * thermal_forcing
    return melt

def quadratic_local_param(gamma, melt_factor, thermal_forcing, U_factor):
    
    """
    Apply the quadratic local parametrization.
    
    This function computes the basal melt based on a quadratic local parametrization (based on Favier et al., 2019 or DeConto and Pollard, 2016), revisited in Burgard et al. 2021.
    
    Parameters
    ----------
    gamma : scalar
        Heat exchange velocity in m per second
    melt_factor : scalar 
        Melt factor representing (rho_sw*c_pw) / (rho_i*L_i) in :math:`\mathtt{K}^{-1}`.
    thermal_forcing: scalar or array
        Difference between T and the freezing temperature Tf (T-Tf) in K or degrees C
    U_factor : scalar or array
        Factor introduced to emulate the speed of the current, see function calculate_melt_rate_2D_simple_1isf.

    Returns
    -------
    melt : scalar or array
        Melt rate in m ice per second
    """
    
    melt = gamma * melt_factor * U_factor * thermal_forcing * abs(thermal_forcing)
    return melt 


def quadratic_mixed_mean(gamma, melt_factor, thermal_forcing, thermal_forcing_avg, U_factor):
    
    """
    Apply the quadratic local and non-local parametrization.
    
    This function computes the basal melt based on a quadratic local parametrization (see Favier et al., 2019), revisited in Burgard et al. 2021.
    
    Parameters
    ----------
    gamma : scalar
        Heat exchange velocity in m per second
    melt_factor : scalar 
        Melt factor representing (rho_sw*c_pw) / (rho_i*L_i) in :math:`\mathtt{K}^{-1}`.
    thermal_forcing: scalar or array
        Difference between T and the freezing temperature Tf (T-Tf) in K or degrees C
    thermal_forcing_avg: scalar
        Spatial average of the thermal forcing in K or degrees C
    U_factor : scalar or array
        Factor introduced to emulate the speed of the current, see function calculate_melt_rate_2D_simple_1isf.

    Returns
    -------
    melt : scalar or array
        Melt rate in m ice per second
    """
    
    melt = gamma * melt_factor * U_factor * thermal_forcing * abs(thermal_forcing_avg)
    return melt 

def quadratic_mixed_slope(gamma, melt_factor, thermal_forcing, thermal_forcing_avg, U_factor, alpha):
    
    """
    Apply the quadratic local and non-local parametrization taking into account the slope.
    
    This function computes the basal melt based on a quadratic local parametrization and taking into account the slope (see Jourdain et al. 2020), revisited in Burgard et al. 2021.
    
    Parameters
    ----------
    gamma : scalar
        Heat exchange velocity in m per second
    melt_factor : scalar 
        Melt factor representing (rho_sw*c_pw) / (rho_i*L_i) in :math:`\mathtt{K}^{-1}`.
    thermal_forcing: scalar or array
        Difference between T and the freezing temperature Tf (T-Tf) in K or degrees C
    thermal_forcing_avg: scalar
        Spatial average of the thermal forcing in K or degrees C
    U_factor : scalar or array
        Factor introduced to emulate the speed of the current, see function calculate_melt_rate_2D_simple_1isf.
    alpha: scalar or array
        Slope angle in rad (must be positive).

    Returns
    -------
    melt : scalar or array
        Melt rate in m ice per second
    """
    
    melt = gamma * melt_factor * U_factor * thermal_forcing * abs(thermal_forcing_avg) * np.sin(alpha)
    return melt 

def compute_T_S_one_box_PICO(T_prev_box,S_prev_box,box_depth,box_area,box_nb,C,gamma,q):
    
    """
    Compute T and S of a box in PICO.
    
    This function computes the temperature and salinity of a box in the box model PICO (see Reese et al., 2018).
    
    Parameters
    ----------
    T_prev_box : scalar
        Temperature in the previous box (box n-1) in degrees C.
    S_prev_box : scalar
        Salinity in the previous box (box n-1) in psu.
    box_depth : scalar 
        Depth of the box below the ice in m (depth is negative!).
    box_area: scalar
        Area of the box in :math:`\mathtt{m}^{2}`.
    box_nb: scalar
        Number of the current box (n).
    C: scalar
        Circulation parameter C in Sv*:math:`\mathtt{m}^{3}`*:math:`\mathtt{kg}^{-1}` = :math:`\mathtt{m}^{6}`*:math:`\mathtt{kg}^{-1}*:math:`\mathtt{s}^{-1}) in [0.1;9]*1.e6
    gamma: scalar
        Effective turbulent temperature exchange velocity in m per second
    q: scalar
        Overturning flux q in :math:`\mathtt{m}^{3}` per second

    Returns
    -------
    q : scalar
        Overturning flux q in :math:`\mathtt{m}^{3}` per second
    T_cur_box : scalar
        Temperature in the current box (box n) in degrees C. 
    S_cur_box : scalar
        Salinity in the current box (box n) in psu. 
    """
    
    Tf = freezing_temperature(S_prev_box, box_depth) 
    T_star = Tf - T_prev_box # above Eq A6 in Reese et al 2018 
    g1 = box_area * gamma # above Eq A6 in Reese et al 2018

    if box_nb == 1:
        # Eq A12 in Reese et al 2018
        s = S_prev_box * melt_factor
        g1_term = g1 / (C * rho_star_pico * (beta_coeff_pico*s - alpha_coeff_pico))
        sn = (0.5*g1_term)**2 - g1_term*T_star
        sn = sn.where(sn>0,0)
        x = -0.5*g1_term + np.sqrt(sn)

        T_cur_box = T_prev_box - x 
        S_cur_box = S_prev_box - x * S_prev_box * melt_factor 

        q = C * rho_star_pico * (beta_coeff_pico*(S_prev_box - S_cur_box) - alpha_coeff_pico * (T_prev_box - T_cur_box)) 

    else:
        g2 = g1 * melt_factor
        x = (-g1 * T_star) / (q + g1 - g2 * l_1 * S_prev_box)

        T_cur_box = T_prev_box - x 
        S_cur_box = S_prev_box - x * S_prev_box * melt_factor 
    
    return q, T_cur_box, S_cur_box


def compute_c_rho_tau(gamma, S_in):

    """
    Compute the c-constants for plume parameterisation.
    
    This function computes c_rho_1, c_rho_2 and c_tau for the plume parameterisation. They are constants given in Table 1 of Lazeroms et al. 2019.
    
    Parameters
    ----------
    gamma: scalar
        Effective thermal Stanton number. Can be modulated for tuning.
    S_in : scalar (or array?)
        Ambient salinity in psu.

    Returns
    -------
    c_rho_1 : scalar (or array?)
        Constant given in Table 1 in Lazeroms et al. 2019.
    c_rho_2 : scalar
        Constant given in Table 1 in Lazeroms et al. 2019.
    c_tau : scalar (or array?)
        Constant given in Table 1 in Lazeroms et al. 2019.
    """  
    
    c_rho_1 = (L_i * alpha_coeff_lazero) / (c_po * gamma * beta_coeff_lazero * S_in)
    c_rho_2 = -l_1 * alpha_coeff_lazero / beta_coeff_lazero
    c_tau = c_rho_2 / c_rho_1 
    return c_rho_1, c_rho_2, c_tau

def compute_X_hat(ice_draft_depth,zGL,T_in,Tf,E0,c_tau,alpha,gamma):
    
    """
    Compute x_hat (or x_tilda) for plume parameterisation.
    
    This function computes x_hat (or x_tilda) for the plume parameterisation. It is a dimensionless coordinate describing distance from plume origin. This is Equation 28(b) in Lazeroms et al. 2019.
    
    Parameters
    ----------
    ice_draft_depth : scalar (or array?)
        Depth of the ice draft in m (depth is negative!).
    zGL: scalar (or array?)
        Depth of the grounding line where the source of the plume is in m (depth is negative!).
    T_in : scalar (or array?)
        Ambient temperature in degrees C.
    Tf : scalar (or array?)
        Freezing temperature 
    E0: scalar
        Entrainment coefficient. Can be modulated for tuning.
    c_tau : scalar (or array?)
        Constant given in Table 1 in Lazeroms et al. 2019.
    alpha: scalar or array
        Slope angle in rad (must be positive).
    gamma: scalar
        Effective thermal Stanton number. Can be modulated for tuning.

    Returns
    -------
    x_hat : scalar or array
        Dimensionless coordinate describing distance from plume origin. Has to be between 0 and 1
    """    

    # x_tilda in Eq 28b in Lazeroms et al. 2019
    x_hat = l_3 * (ice_draft_depth-zGL)/((T_in-Tf) * (1 + C_eps_lazero * ((E0 * np.sin(alpha))/(gamma + c_tau + E0 * np.sin(alpha)))**(3/4)))
    # all of this derivation is only valid for x in [0,1]
    x_hat = x_hat.where(x_hat<1,1)
    x_hat = x_hat.where(x_hat>0,0)
    x_hat = x_hat.where(zGL<=0)
    return x_hat

def compute_M_hat(X_hat):
    
    """
    Compute M_hat for plume parameterisation.
    
    This function computes the M_hat (or M0) for the plume parameterisation. This is Equation 26 in Lazeroms et al. 2019.
    
    Parameters
    ----------
    X_hat : scalar or array
        Coordinate describing distance from plume origin.

    Returns
    -------
    M_hat : scalar or array
        Dimensionless melt rate, to be multiplied with the Mterm in Eq 28a.
    """
    
    # here M_hat = M0(X_tilde), Eq 26 in Lazeroms et al. 2019
    M_hat = 1 / (2*np.sqrt(2)) * (3*(1 - X_hat)**(4/3) - 1) * np.sqrt(1 - (1 - X_hat)**(4/3))
    return M_hat

def compute_Mterm(T_in, S_in, Tf, c_rho_1, c_tau, gamma, E0, alpha, thermal_forcing_factor):
    
    """
    Compute M-term for plume parameterisation.
    
    This function computes the M-term for the plume parameterisation. This is the beginning of Equation 28(a) in Lazeroms et al. 2019.
    
    Parameters
    ----------
    T_in : scalar (or array?)
        Ambient temperature in degrees C.
    S_in : scalar (or array?)
        Ambient salinity in psu.
    Tf : scalar (or array?)
        Freezing temperature 
    c_rho_1 : scalar
        Constant given in Table 1 in Lazeroms et al. 2019.
    c_tau : scalar (or array?)
        Constant given in Table 1 in Lazeroms et al. 2019.
    gamma: scalar
        Effective thermal Stanton number. Can be modulated for tuning.
    E0: scalar
        Entrainment coefficient. Can be modulated for tuning.
    alpha: scalar or array
        Slope angle in rad (must be positive).
    thermal_forcing_factor : scalar (or array?)
        Factor to be multiplied to T0-Tf in the end of Eq. 28a - either thermal forcing or thermal forcing average.

    Returns
    -------
    Mterm : scalar or array
        Term to be multiplied with M_hat in Eq 28a.
    """

    Mterm = (np.sqrt((beta_coeff_lazero*S_in*g) / (l_3*(L_i/c_po)**3)) 
        * np.sqrt((1 - c_rho_1 * gamma) / (C_d_lazero + E0*np.sin(alpha))) 
        * ((gamma * E0 * np.sin(alpha)) / (gamma + c_tau + E0*np.sin(alpha)))**(3/2) 
        * (T_in - Tf) * thermal_forcing_factor)
    return Mterm

def interp_profile_to_depth_of_interest(var_in, depth_of_int):
    
    """
    Interpolate a profile to a given depth point.
    
    This function interpolates a variable at a given depth from a profile.
    
    Parameters
    ----------
    var_in : xr.DataArray
        Profile of the variable of interest, must have a dimension 'depth'. Make sure that ``depth_of_int`` and the dimension ``depth`` have the same sign.
    depth_of_int : scalar
        Depth at which we want the variable interpolated.

    Returns
    -------
    var_out : xr.DataArray
        Variable of interest at the given depth of interest.
    """
    
    var_out = var_in.interp(depth=depth_of_int).drop('depth')
    return var_out

def g_Mterm(alpha, E0, gamma_T_S):
    
    """
    Compute g-term for M-term in PICOP for plume parameterisation.
    
    This function computes the g-term for M-term in PICOP. This is the beginning of Equation 6 in Pelle et al. 2019.
    
    Parameters
    ----------
    alpha: scalar or array
        Slope angle in rad (must be positive).
    E0 : float
        Entrainment coefficient.
    gamma_T_S : float
        Effective Stanton number.
        
    Returns
    -------
    g_alpha : scalar or array
        Term used to compute the melt in Equation 10 of Pelle et al. 2019.
    """
    
    g_alpha = (np.sqrt((np.sin(alpha))/ (C_d_lazero + E0 * np.sin(alpha)))
                * np.sqrt((E0 * np.sin(alpha)) / (np.sqrt(C_d_lazero) * gamma_T_S + E0*np.sin(alpha)))
                * ((E0 * np.sin(alpha)) / (np.sqrt(C_d_lazero) * gamma_T_S + E0*np.sin(alpha))))
    
    return g_alpha

def f_xterm(stanton_number, E0, alpha):
    
    """
    Compute f-term for x-term in PICOP for plume parameterisation.
    
    This function computes the f-term for x-term in PICOP. This is the right part of Equation A10 in Lazeroms et al. 2018.
    
    Parameters
    ----------
    stanton_number : float
        Effective Stanton number computed based on Equation 5 in Pelle et al. 2019.
    E0 : float
        Entrainment coefficient.
    alpha : scalar or array
        Slope angle in rad (must be positive).
                
    Returns
    -------
    f_alpha : scalar or array
        Term to be used for the length scale l in Equation A10 in Lazeroms et al. 2018.
    """
    
    f_alpha = ((x0 * stanton_number + E0 * np.sin(alpha)) / (x0 * (stanton_number + E0 * np.sin(alpha))))
    
    return f_alpha

def compute_Mterm_picop(T_in, Tf, E0, alpha, gamma_T_S):
    
    """
    Compute M-term for PICOP plume parameterisation.
    
    This function computes the M-term for the PICOP plume parameterisation. This is Equation 10 in Pelle et al. 2019.
    
    Parameters
    ----------
    T_in : scalar (or array?)
        Ambient temperature in degrees C.
    Tf : scalar (or array?)
        Freezing temperature 
    E0: scalar
        Entrainment coefficient. Can be modulated for tuning.
    alpha: scalar or array
        Slope angle in rad (must be positive).
    gamma_T_S : float
        Effective Stanton number.
        
    Returns
    -------
    Mterm : scalar or array
        Term to be multiplied with M_hat in Eq 9 of Pelle et al. (2019).
    """

    Mterm = M0 * g_Mterm(alpha, E0, gamma_T_S) * (T_in - Tf)**2
    
    return Mterm

def compute_X_hat_picop(T_in, Tf, ice_draft_depth, zGL, stanton_number, E0, alpha):
    
    """
    Compute xhat factor for PICOP plume parameterisation.
    
    This function computes the x-hat factor for the plume parameterisation. This is Equation 8 in Pelle et al. 2019.
    
    Parameters
    ----------
    T_in : scalar (or array?)
        Ambient temperature in degrees C.
    Tf : scalar (or array?)
        Freezing temperature 
    ice_draft_depth : scalar (or array?)
        Depth of the ice draft in m. Must be negative.
    zGL : scalar (or array?)
        Depth of the grounding line in m. Must be negative.
    stanton_number : float
        Effective Stanton Number. 
    E0: scalar
        Entrainment coefficient. Can be modulated for tuning.
    alpha: scalar or array
        Slope angle in rad (must be positive).

    
    Returns
    -------
    x_hat : array
        Factor to be used for the polynomial version of M_hat. Dimensionless coordinate.

    """

    l = f_xterm(stanton_number, E0, alpha) * (T_in - Tf) / l_3
    
    x_hat = (ice_draft_depth-zGL) / l
    
    # all of this derivation is only valid for x in [0,1]
    x_hat = x_hat.where(x_hat<1,1)
    x_hat = x_hat.where(x_hat>0,0)
    x_hat = x_hat.where(zGL<=0)
    return x_hat

def compute_M_hat_picop(x_hat):
    
    """
    Compute Mhat factor for PICOP plume parameterisation.
    
    This function computes the Mhat factor for the Lazeroms 2018 plume parameterisation. This is Equation A13 in Lazeroms et al. 2018.
    
    Parameters
    ----------
    x_hat : array
        Factor to be used for the polynomial version of M_hat. Dimensionless coordinate.

    
    Returns
    -------
    M_hat_picop : array
        Factor to be multiplied with Mterm.

    """
    #polynom = Polynomial([1.371 * 10**-1,
    #                      5.528 * 10,
    #                      -8.952 * 10**2,
    #                     8.927 * 10**3,
    #                     -5.564 * 10**4,
    #                     2.219 * 10**5,
    #                     -5.820 * 10**5,
    #                     1.015 * 10**6,
    #                     -1.166 * 10**6,
    #                     8.467 * 10**5,
    #                     -3.521 * 10**5,
    #                     6.388 * 10**4])
    
    polynom = Polynomial([0.1371330075095435,
                          5.527656234709359e1,
                          -8.951812433987858e2,
                          8.927093637594877e3,
                          -5.563863123811898e4,
                          2.218596970948727e5,
                          -5.820015295669482e5,
                          1.015475347943186e6,
                          -1.166290429178556e6,
                          8.466870335320488e5,
                          -3.520598035764990e5,
                          6.387953795485420e4])
    M_hat_picop = polynom(x_hat)
    
    return M_hat_picop

def plume_param(T_in, S_in, ice_draft_depth, zGL, alpha, gamma, E0, picop=False):
    
    """
    Apply the plume parametrization.
    
    This function computes the basal melt based on a plume parametrization (see Lazeroms et al. 2018 and Lazeroms et al. 2019).
    
    Parameters
    ----------
    T_in : scalar (or array?)
        Ambient temperature in degrees C.
    S_in : scalar (or array?)
        Ambient salinity in psu.
    ice_draft_depth : scalar or array
        Depth of the ice draft in m (depth is negative!).
    zGL: scalar or array
        Depth of the grounding line where the source of the plume is in m (depth is negative!).
    alpha: scalar or array
        Slope angle in rad (must be positive).
    gamma: scalar
        Effective thermal Stanton number. Can be modulated for tuning.
    E0: scalar
        Entrainment coefficient. Can be modulated for tuning.
    picop: Boolean
        Option defining which Mterm function to use.

    Returns
    -------
    melt_rate : scalar or array
        Melt rate in m ice per second.
    """
    
    c_rho_1, c_rho_2, c_tau = compute_c_rho_tau(gamma, S_in)
    
    # freezing temperature at the grounding line
    Tf = freezing_temperature(S_in, zGL)
    thermal_forcing = T_in-Tf
    
    if picop:
        # Equation 5 in Pelle et al. 2019
        gamma_T_S = ((C_d_gamma_T / np.sqrt(C_d_lazero)) 
                 * (g_1 + g_2 * ((T_in - Tf) / l_3) * ((E0 * np.sin(alpha)) / (C_d_gamma_T_S0 + E0 * np.sin(alpha)))))
        
        stanton_number = np.sqrt(C_d_lazero) * gamma_T_S
    
    if picop:
        x_hat = compute_X_hat_picop(T_in, Tf, ice_draft_depth, zGL, stanton_number, E0, alpha)
    else:
        x_hat = compute_X_hat(ice_draft_depth,zGL,T_in,Tf,E0,c_tau,alpha,gamma)
    
    if picop:
        M_hat = compute_M_hat_picop(x_hat)
    else:
        M_hat = compute_M_hat(x_hat)
        
    if picop:
        Mterm = compute_Mterm_picop(T_in, Tf, E0, alpha, gamma_T_S)
    else:
        Mterm = compute_Mterm(T_in, S_in, Tf, c_rho_1, c_tau, gamma, E0, alpha, thermal_forcing)
        
    melt_rate = Mterm * M_hat * rho_sw/rho_i
    
    return melt_rate


    
def plume_param_modif(theta_isf,salinity_isf,ice_draft_points,front_bot_dep_max_isf,zGL_isf,alpha_isf,conc_isf,dGL_isf,gamma,E0):
    
    """
    Apply the plume parametrization with modifications (from Burgard et al. 2022).
    
    This function computes the basal melt based on a plume parameterisation (see Lazeroms et al. 2018 and Lazeroms et al. 2019) with modifications from pers. comm. between A. Jenkins and C. Burgard & N. Jourdain.
    
    Parameters
    ----------
    theta_isf : array
        Ambient temperature profile in degrees C.
    salinity_isf : array
        Ambient salinity profile in psu.
    ice_draft_points : array
        Depth of the ice draft in m (depth is positive!).
    front_bot_dep_max_isf : scalar
        Maximum depth of the continental shelf at the entry of the ice shelf (depth is positive!)
    zGL_isf: scalar or array
        Depth of the grounding line in m (depth is negative!), containing 'cavity' and 'lazero' option.
    alpha_isf: scalar or array
        Slope angle in rad (must be positive), containing 'cavity' and 'lazero' option.
    conc_isf: array
        Concentration of grid cell covered by the ice shelf.
    dGL_isf : array
        Distance between local point and nearest grounding line in m.
    Tf_gl : scalar (or array?)
        Freezing temperature at the grounding line in degrees C.
    E0: scalar
        Entrainment coefficient. Can be modulated for tuning.
    gamma: scalar
        Effective thermal Stanton number. Can be modulated for tuning.

    Returns
    -------
    melt_rate : scalar or array
        Melt rate in m ice per second.
    """
    
    zGL_cavity_isf = -1*zGL_isf.sel(option='cavity').max()
    zGL_lazero_points = -1*zGL_isf.sel(option='lazero')
    
    depth_GL_to_interp = zGL_cavity_isf.where(zGL_cavity_isf < front_bot_dep_max_isf, front_bot_dep_max_isf)
    S_GL = salinity_isf.interp({'depth': depth_GL_to_interp}).drop('depth')
    Tf_GL = freezing_temperature(S_GL, -1*zGL_cavity_isf)

    ###### LOCAL STUFF
    depth_of_int = ice_draft_points.where(ice_draft_points < front_bot_dep_max_isf, front_bot_dep_max_isf)
    # Local temperature and salinity at ice draft depth
    T0_loc = theta_isf.interp({'depth': depth_of_int}).drop('depth')
    S0_loc = salinity_isf.interp({'depth': depth_of_int}).drop('depth')
    Tf_loc = freezing_temperature(S0_loc, -1*ice_draft_points)
    c_rho_1_loc, c_rho_2_loc, c_tau_loc = compute_c_rho_tau(gamma, S0_loc)
    alpha_loc = alpha_isf.sel(option='local')
    
    ##### CAVITY MEAN => Mhat
    # Mean temperature and salinity over whole ice-shelf base
    T0_mean_cav = uf.weighted_mean(T0_loc, ['mask_coord'], conc_isf)
    S0_mean_cav = uf.weighted_mean(S0_loc, ['mask_coord'], conc_isf)
    # Mean coefficients over whole ice-shelf base
    c_rho_1_mean_cav, c_rho_2_mean_cav, c_tau_mean_cav = compute_c_rho_tau(gamma, S0_mean_cav)
    # Length scale and M_hat over whole ice-shelf base
    x_mean_cav = compute_X_hat(-1*ice_draft_points,-1*zGL_cavity_isf,T0_mean_cav,Tf_GL,E0,c_tau_mean_cav,alpha_isf.sel(option='cavity'),gamma)
    M_hat = compute_M_hat(x_mean_cav)
    
    ##### UPSTREAM STUFF
    ### if we want to take the mean of the profile in front of the ice shelf
    # interpolate T and S profiles on regular depth axis for the mean
    depth_axis_loc = xr.DataArray(data=np.arange(1,zGL_cavity_isf+1),dims=['depth']).chunk({'depth':50})
    # make sure to not use T and S lower than the maximum front depth
    depth_axis_loc_cut = depth_axis_loc.where(depth_axis_loc<front_bot_dep_max_isf,front_bot_dep_max_isf)
    depth_axis_loc_cut = depth_axis_loc_cut.assign_coords({'depth':depth_axis_loc})
    # do the interpolation
    theta_1m = theta_isf.interp({'depth': depth_axis_loc_cut}).chunk({'depth':50})
    salinity_1m = salinity_isf.interp({'depth': depth_axis_loc_cut}).chunk({'depth':50})
    
    # depth between grounding line and local ice draft depth
    depth_range_mask = (theta_1m.depth <= zGL_lazero_points+1) & (theta_1m.depth >= ice_draft_points-1)
    
    # average upstream properties
    theta_ups = theta_1m.where(depth_range_mask).mean('depth')
    salinity_ups = salinity_1m.where(depth_range_mask).mean('depth')
    
    Tf_ups = freezing_temperature(salinity_ups,-1*zGL_lazero_points)
    thermal_forcing_ups = theta_ups - Tf_ups
    
    # angle between local point and grounding line
    alpha_ups = np.arctan((zGL_lazero_points - ice_draft_points)/dGL_isf)
    alpha_ups = alpha_ups.where(dGL_isf>0, alpha_loc)

    c_rho_1_ups, c_rho_2_ups, c_tau_ups = compute_c_rho_tau(gamma, salinity_ups)
    #c_rho_1_ups0, c_rho_2_ups0, c_tau_ups0 = compute_c_rho_tau(gamma, uf.weighted_mean(S0_loc.where(both_masks2), 'mask_coord', conc_isf.where(both_masks2))) # same as above
    
    Mterm = (np.sqrt((beta_coeff_lazero*S0_loc*g) / (l_3*(L_i/c_po)**3)) 
    * np.sqrt((1 - c_rho_1_loc * gamma) / (C_d_lazero + E0*np.sin(alpha_loc))) 
    * np.sqrt((gamma * E0 * np.sin(alpha_loc)) / (gamma + c_tau_loc + E0*np.sin(alpha_loc))) * (T0_loc - Tf_loc)
    * ((gamma * E0 * np.sin(alpha_ups)) / (gamma + c_tau_ups + E0*np.sin(alpha_ups))) * thermal_forcing_ups) 
    
    ##### MELT RATE
    melt_rate = M_hat * Mterm * rho_sw/rho_i
    
    return melt_rate.squeeze().load()#.drop('option')  


def T_correction_PICO(isf_name, T_in):
    
    """
    Apply regional temperature corrections to input temperatures for box model.
    
    This functions produces locally "corrected" temperature profiles for input to box model, fitting the "new" Reese parameters. 
    
    Parameters
    ----------
    isf_name : str
        Name of the ice shelf of interest.
    T_in : array
        Temperature profile in degrees C.

    Returns
    -------
    T_corrected : array
        Corrected temperature profile in degrees C.
    """
    
    #print('T_correction!')
    if isf_name in ['Filchner','Ronne','Filchner-Ronne']: #1
        T_correction= 0.17475163929428827 # Best
        # T_correction = 0.0743606316460328 # Max
        # T_correction = 0.5244083302228169 # Min
    elif isf_name in ['Stancomb Brunt','Riiser-Larsen']: #2
        T_correction= -0.0255081029413049 # Best
        # T_correction = -0.12548257518694172 # Max
        # T_correction = 0.1501077396062951 # Min
    elif isf_name in ['Fimbul','Ekstrom']: #3
        T_correction= -0.2395108534458497 # Best
        # T_correction = -0.3148726679812903 # Max
        # T_correction = -0.08954377969106031 # Min
    elif isf_name in ['Roi Baudouin','Nivl','Lazarev','Borchgrevink','Jelbart','Prince Harald']: #4
        T_correction= -0.2395108534458497 # Best
        # T_correction = -0.2302679118230881 # Max
        # T_correction = 0.019731915700313518 # Min
    elif isf_name in ['Amery']: #6
        T_correction= -0.1499489774269802 # Best
        # T_correction = -0.2253108845620384 # Max
        # T_correction = 0.02450222960220194 # Min
    elif isf_name in ['West','Shackleton','Tracy Tremenchus']: #7
        T_correction= 0.07451035394701355 # Best 
        # T_correction = -0.025540487708914258 # Max
        # T_correction = 0.27447478156728833 # Min
    elif isf_name in ['Totten','Moscow Univ.']: #8
        T_correction= -1.1254891863236058 # Best 
        # T_correction = -1.2739265689483057 # Max
        # T_correction = -0.9255087879987864 # Min
    elif isf_name in ['Cook']: #9
        T_correction= -0.3921345255042934 # Best
        # T_correction = -0.4564870583860179 # Max
        # T_correction = -0.32351184947581224 # Min
    elif isf_name in ['Ross']: #12
        T_correction= 0.3496226730326237 # Best
        # T_correction = 0.17430680792175826 # Max
        # T_correction = 0.7497004029068532 # Min
    elif isf_name in ['Getz','Nickerson','Sulzberger','Dotson']: #13
        T_correction= -1.1000717723391031 # Best
        # T_correction = -1.2500222564940329 # Max
        # T_correction = -0.8500613833763373 # Min
    elif isf_name in ['Pine Island','Thwaites','Crosson']: #14
        T_correction= -1.3499828424718645 # Best
        # T_correction = -1.67548468708992 # Max
        # T_correction = -0.92548468708992 # Min
    elif isf_name in ['Cosgrove','Abbot','Venable']: #15
        T_correction= -1.9999923008093563 # Best
        # T_correction = -1.9999923008093563 # Max
        # T_correction = -1.9999923008093563 # Min
    elif isf_name in ['Wilkins','Stange','Bach','George VI']: #16
        T_correction= -1.9999930749714996 # Best
        # T_correction = -1.9999930749714996 # Max
        # T_correction = -1.6999787982359889 # Min
    elif isf_name in ['Larsen C']: #18
        T_correction= -0.3819311957529061 # Best
        # T_correction = -0.4819107321859564 # Max
        # T_correction = -0.15736237917388785 # Min
    elif isf_name in ['Larsen D']: #19
        T_correction= -0.14732530277455758 # Best
        # T_correction = -0.1616054051436846 # Max
        # T_correction = -0.09932079835170393 # Min
    
    T_corrected = T_in + T_correction
    return T_corrected

def PICO_and_PICOP_param(T_in,S_in,box_location,box_depth_below_surface,box_area_whole,nD,spatial_coord,isf_cell_area,
                         gamma, E0, C,
                         picop='no',pism_version='no',zGL=None,alpha=None,ice_draft_neg=None):
    
    """
    Compute melt rate using PICO or PICOP. THIS FUNCTION IS NOT USED ANYMORE => NOW TWO SEPARATE FOR PICO AND PICOP
    
    This function computes the basal melt based on PICO or PICOP (see Reese et al., 2018 and Pelle et al., 2019).
    
    Parameters
    ----------
    T_in : xr.DataArray
        Temperature entering the cavity in degrees C.
    S_in : xr.DataArray
        Salinity entering the cavity in psu.
    box_location : xr.DataArray
        Spatial location of the boxes (dims=['box_nb_tot']), 
    box_depth_below_surface : xr.DataArray
        Mean ice draft depth of the box in m (dims=['box_nb_tot','box_nb']). Negative downwards!
    box_area_whole : xr.DataArray
        Area of the different boxes (dims=['box_nb_tot','box_nb']).
    nD: scalar
        Number of boxes to use (i.e. box_nb_tot).
    spatial_coord : list of str
        Coordinate(s) to use for spatial means.
    isf_cell_area: float
        Area covered by ice shelf in each cell
    gamma : float
        Gamma to be tuned in m/s.
    E0 : float
        Entrainment coefficient.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    picop: str
        Can be ``yes`` or ``no``, depending on if you want to use PICOP or the original PICO. 
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    zGL: scalar or array
        Depth of the grounding line where the source of the plume is in m (depth is negative!).
    alpha: scalar or array
        Slope angle in rad (must be positive).
    ice_draft_neg : scalar or array
        Depth of the ice draft in m (depth is negative!).#

    Returns
    -------
    T_all_boxes : xr.DataArray
        Temperature field in degrees C.
    S_all_boxes : xr.DataArray
        Salinity field in psu.
    m_all_boxes : xr.DataArray
        Melt rate field in m ice per second. 
    """
        
    box_loc = box_location.sel(box_nb_tot=nD)
    #box_loc = box_loc_orig.where(box_loc_orig < 11)

    box_depth_all = box_depth_below_surface.sel(box_nb_tot=nD)
    box_area_all = box_area_whole.sel(box_nb_tot=nD)

    if pism_version == 'yes':
        box_depth = ice_draft_neg.where(box_loc == box_area_all.box_nb)
    else: 
        box_depth = box_depth_all
    box_area = box_area_all

    for bbox in range(1,nD+1):

        if bbox == 1:
            
            T_prev_box = T_in
            S_prev_box = S_in

            q_orig, T_cur_box, S_cur_box = compute_T_S_one_box_PICO(T_prev_box,S_prev_box,
                                                                    box_depth.sel(box_nb=bbox),
                                                                    box_area.sel(box_nb=bbox),
                                                                    bbox,
                                                                    C,gamma,None)
            T_all_boxes = T_cur_box.where(box_loc == bbox)
            S_all_boxes = S_cur_box.where(box_loc == bbox)
            q = uf.weighted_mean(q_orig.where(box_loc == bbox),spatial_coord,isf_cell_area.where(box_loc == bbox))

        else:

            if pism_version=='yes':
                T_prev_box = uf.weighted_mean(T_cur_box.where(box_loc == bbox-1),spatial_coord,isf_cell_area.where(box_loc == bbox-1))
                S_prev_box = uf.weighted_mean(S_cur_box.where(box_loc == bbox-1),spatial_coord,isf_cell_area.where(box_loc == bbox-1))
            else:
                T_prev_box = T_cur_box
                S_prev_box = S_cur_box

            q, T_cur_box, S_cur_box = compute_T_S_one_box_PICO(T_prev_box,S_prev_box,
                                                                box_depth.sel(box_nb=bbox),
                                                                box_area.sel(box_nb=bbox),
                                                                bbox,
                                                                C,gamma,q)

            T_all_boxes = T_all_boxes.combine_first(T_cur_box.where(box_loc == bbox))
            S_all_boxes = S_all_boxes.combine_first(S_cur_box.where(box_loc == bbox))     

        if picop == 'no':

            thermal_forcing_box = -(freezing_temperature(S_cur_box, box_depth.sel(box_nb=bbox)) - T_cur_box)
            m_cur_box = linear_local_param(gamma, melt_factor, thermal_forcing_box)

            if bbox == 1:
                m_all_boxes = m_cur_box.where(box_loc == bbox)
            else:
                m_all_boxes = m_all_boxes.combine_first(m_cur_box.where(box_loc == bbox))

        elif picop == 'yes':

            ice_draft_depth_box = ice_draft_neg.where(box_loc == bbox) #zz1

            zGL_box = zGL.where(box_loc == bbox)
            alpha_box = alpha.where(box_loc == bbox)
            m_cur_box_pts = plume_param(T_cur_box, S_cur_box, ice_draft_depth_box, zGL_box, alpha_box, gamma, E0, picop=True)

            if bbox == 1:
                m_all_boxes = m_cur_box_pts.copy()
            else:
                m_all_boxes = m_all_boxes.combine_first(m_cur_box_pts)

    return T_all_boxes.drop('box_nb').drop('box_nb_tot'), S_all_boxes.drop('box_nb').drop('box_nb_tot'), m_all_boxes.drop('box_nb').drop('box_nb_tot')


def PICO_param(T_in,S_in,box_location,box_depth_below_surface,box_area_whole,nD,spatial_coord,isf_cell_area,gamma, C, ice_draft_neg, pism_version='no'):
    
    """
    Compute melt rate using PICO.
    
    This function computes the basal melt based on PICO  (see Reese et al., 2018).
    
    Parameters
    ----------
    T_in : xr.DataArray
        Temperature entering the cavity in degrees C.
    S_in : xr.DataArray
        Salinity entering the cavity in psu.
    box_location : xr.DataArray
        Spatial location of the boxes (dims=['box_nb_tot']), 
    box_depth_below_surface : xr.DataArray
        Mean ice draft depth of the box in m (dims=['box_nb_tot','box_nb']). Negative downwards!
    box_area_whole : xr.DataArray
        Area of the different boxes (dims=['box_nb_tot','box_nb']).
    nD: scalar
        Number of boxes to use (i.e. box_nb_tot).
    spatial_coord : list of str
        Coordinate(s) to use for spatial means.
    isf_cell_area: float
        Area covered by ice shelf in each cell
    gamma : float
        Gamma to be tuned in m/s.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    ice_draft_neg : scalar or array
        Depth of the ice draft in m (depth is negative!).
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 

    Returns
    -------
    T_all_boxes : xr.DataArray
        Temperature field in degrees C.
    S_all_boxes : xr.DataArray
        Salinity field in psu.
    m_all_boxes : xr.DataArray
        Melt rate field in m ice per second. 
    """
        
    box_loc = box_location.sel(box_nb_tot=nD)
    #box_loc = box_loc_orig.where(box_loc_orig < 11)

    box_depth_all = box_depth_below_surface.sel(box_nb_tot=nD)
    box_area_all = box_area_whole.sel(box_nb_tot=nD)

    if pism_version == 'yes':
        box_depth = ice_draft_neg.where(box_loc == box_area_all.box_nb)
    else: 
        box_depth = box_depth_all
    box_area = box_area_all

    for bbox in range(1,nD+1):

        if bbox == 1:
            
            T_prev_box = T_in
            S_prev_box = S_in

            q_orig, T_cur_box, S_cur_box = compute_T_S_one_box_PICO(T_prev_box,S_prev_box,
                                                                    box_depth.sel(box_nb=bbox),
                                                                    box_area.sel(box_nb=bbox),
                                                                    bbox,
                                                                    C,gamma,None)
            T_all_boxes = T_cur_box.where(box_loc == bbox)
            S_all_boxes = S_cur_box.where(box_loc == bbox)
            q = uf.weighted_mean(q_orig.where(box_loc == bbox),spatial_coord,isf_cell_area.where(box_loc == bbox))

        else:

            if pism_version=='yes':
                T_prev_box = uf.weighted_mean(T_cur_box.where(box_loc == bbox-1),spatial_coord,isf_cell_area.where(box_loc == bbox-1))
                S_prev_box = uf.weighted_mean(S_cur_box.where(box_loc == bbox-1),spatial_coord,isf_cell_area.where(box_loc == bbox-1))
            else:
                T_prev_box = T_cur_box
                S_prev_box = S_cur_box

            q, T_cur_box, S_cur_box = compute_T_S_one_box_PICO(T_prev_box,S_prev_box,
                                                                box_depth.sel(box_nb=bbox),
                                                                box_area.sel(box_nb=bbox),
                                                                bbox,
                                                                C,gamma,q)

            T_all_boxes = T_all_boxes.combine_first(T_cur_box.where(box_loc == bbox))
            S_all_boxes = S_all_boxes.combine_first(S_cur_box.where(box_loc == bbox))     

            
        thermal_forcing_box = -(freezing_temperature(S_cur_box, box_depth.sel(box_nb=bbox)) - T_cur_box)
        m_cur_box = linear_local_param(gamma, melt_factor, thermal_forcing_box)

        if bbox == 1:
            m_all_boxes = m_cur_box.where(box_loc == bbox)
        else:
            m_all_boxes = m_all_boxes.combine_first(m_cur_box.where(box_loc == bbox))


    return T_all_boxes.drop('box_nb').drop('box_nb_tot'), S_all_boxes.drop('box_nb').drop('box_nb_tot'), m_all_boxes.drop('box_nb').drop('box_nb_tot')



def PICOP_param(T_in,S_in,box_location,box_depth_below_surface,box_area_whole,nD,spatial_coord,isf_cell_area,
                         gamma_pico, gamma_plume, E0, C,zGL,alpha,ice_draft_neg,pism_version='no',picop_opt='2019'):
    
    """
    Compute melt rate using PICOP.
    
    This function computes the basal melt based on PICOP (see Pelle et al., 2019). Using the empirical equations from Lazeroms et al. 2018.
    
    Parameters
    ----------
    T_in : xr.DataArray
        Temperature entering the cavity in degrees C.
    S_in : xr.DataArray
        Salinity entering the cavity in psu.
    box_location : xr.DataArray
        Spatial location of the boxes (dims=['box_nb_tot']), 
    box_depth_below_surface : xr.DataArray
        Mean ice draft depth of the box in m (dims=['box_nb_tot','box_nb']). Negative downwards!
    box_area_whole : xr.DataArray
        Area of the different boxes (dims=['box_nb_tot','box_nb']).
    nD: scalar
        Number of boxes to use (i.e. box_nb_tot).
    spatial_coord : list of str
        Coordinate(s) to use for spatial means.
    isf_cell_area: float
        Area covered by ice shelf in each cell
    gamma_pico : float
        Gamma to be used in PICO part.
    gamma_plume : float
        Gamma to be used in plume part.
    E0 : float
        Entrainment coefficient.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    zGL: scalar or array
        Depth of the grounding line where the source of the plume is in m (depth is negative!).
    alpha: scalar or array
        Slope angle in rad (must be positive).
    ice_draft_neg : scalar or array
        Depth of the ice draft in m (depth is negative!).
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    picop_opt: str
        Can be ``2019``, ``2018`` , depending on if you want to use PICOP with analytical plume param or with empirical plume param.

    Returns
    -------
    T_all_boxes : xr.DataArray
        Temperature field in degrees C.
    S_all_boxes : xr.DataArray
        Salinity field in psu.
    m_all_boxes : xr.DataArray
        Melt rate field in m ice per second. 
    """
        
    box_loc = box_location.sel(box_nb_tot=nD)
    #box_loc = box_loc_orig.where(box_loc_orig < 11)

    box_depth_all = box_depth_below_surface.sel(box_nb_tot=nD)
    box_area_all = box_area_whole.sel(box_nb_tot=nD)

    if pism_version == 'yes':
        box_depth = ice_draft_neg.where(box_loc == box_area_all.box_nb)
    else: 
        box_depth = box_depth_all
    box_area = box_area_all

    for bbox in range(1,nD+1):

        if bbox == 1:
            
            T_prev_box = T_in
            S_prev_box = S_in

            q_orig, T_cur_box, S_cur_box = compute_T_S_one_box_PICO(T_prev_box,S_prev_box,
                                                                    box_depth.sel(box_nb=bbox),
                                                                    box_area.sel(box_nb=bbox),
                                                                    bbox,
                                                                    C,gamma_pico,None)
            T_all_boxes = T_cur_box.where(box_loc == bbox)
            S_all_boxes = S_cur_box.where(box_loc == bbox)
            q = uf.weighted_mean(q_orig.where(box_loc == bbox),spatial_coord,isf_cell_area.where(box_loc == bbox))

        else:

            if pism_version=='yes':
                T_prev_box = uf.weighted_mean(T_cur_box.where(box_loc == bbox-1),spatial_coord,isf_cell_area.where(box_loc == bbox-1))
                S_prev_box = uf.weighted_mean(S_cur_box.where(box_loc == bbox-1),spatial_coord,isf_cell_area.where(box_loc == bbox-1))
            else:
                T_prev_box = T_cur_box
                S_prev_box = S_cur_box

            q, T_cur_box, S_cur_box = compute_T_S_one_box_PICO(T_prev_box,S_prev_box,
                                                                box_depth.sel(box_nb=bbox),
                                                                box_area.sel(box_nb=bbox),
                                                                bbox,
                                                                C,gamma_pico,q)

            T_all_boxes = T_all_boxes.combine_first(T_cur_box.where(box_loc == bbox))
            S_all_boxes = S_all_boxes.combine_first(S_cur_box.where(box_loc == bbox))     


        ice_draft_depth_box = ice_draft_neg.where(box_loc == bbox) #zz1

        zGL_box = zGL.where(box_loc == bbox)
        alpha_box = alpha.where(box_loc == bbox)
        if picop_opt == '2018':
            m_cur_box_pts = plume_param(T_cur_box, S_cur_box, ice_draft_depth_box, zGL_box, alpha_box, gamma_plume, E0, picop=True)
        elif picop_opt == '2019':
            m_cur_box_pts = plume_param(T_cur_box, S_cur_box, ice_draft_depth_box, zGL_box, alpha_box, gamma_plume, E0, picop=False)
        if bbox == 1:
            m_all_boxes = m_cur_box_pts.copy()
        else:
            m_all_boxes = m_all_boxes.combine_first(m_cur_box_pts)

    return T_all_boxes.drop('box_nb').drop('box_nb_tot'), S_all_boxes.drop('box_nb').drop('box_nb_tot'), m_all_boxes.drop('box_nb').drop('box_nb_tot')


def merge_over_dim(da_in, da_out, dim, dim_index):
    
    """
    Utility function to merge different melt rate results into one DataArray.
    
    This function merges different melt rate results into one DataArray (e.g. different param choices or different ice shelves).
    
    Parameters
    ----------
    da_in : xr.DataArray
        DataArray to be merged to the rest.
    da_out : xr.DataArray
        DataArray merged until now.
    dim : str
        Name of the dimension
    dim_index: str or int
        Index of dim corresponding to ``da_in``.

    Returns
    -------
    da_out : xr.DataArray
        Merged DataArray.
    """    

    da_in = da_in.expand_dims(dim)
    da_in[dim] = np.array((dim_index, ))
    if da_out is None:
        da_out = da_in.copy()
    else:
        da_out = xr.concat([da_out, da_in], dim=dim)
    return da_out



def calculate_melt_rate_2D_simple_1isf(kisf, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, U_param=True, HUB=False):
    
    """
    Function to compute melt from simple parameterisations for one ice shelf.
        
    Parameters
    ----------
    kisf : int
        Ice shelf ID for the ice shelf of interest. 
    T_S_profile : xarray.Dataset
        Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    geometry_info_1D : xarray.Dataset
        Dataset containing relevant 1D geometrical information.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    mparam : str
        Parameterisation to be applied.
    gamma : float
        Gamma to be tuned.
    U_param : Boolean
        If ``True`` we use the complex parameterisation of U, if ``False``, this is "only" equal to (rho_sw * c_po) / (rho_i * L_i).

    Returns
    -------
    melt_rate : xr.DataArray
        Horizontal distribution of the melt rate in m/s for the ice shelf of interest.
    """     

    geometry_isf_2D = uf.choose_isf(geometry_info_2D,isf_stack_mask, kisf)
    geometry_isf_1D = geometry_info_1D.sel(Nisf=kisf)
    
    # decide on the depth to which to extrapolate the entry temperature
    if mparam[-6::]=='bottom':
        mparam0 = mparam[:-7:]
        # deepest entrance point
        #depth_of_int = geometry_isf_1D['front_bot_depth_max'] # deepest point of the entrance
        depth_of_int = geometry_isf_1D['front_bot_depth_avg']
        depth_of_int.where(geometry_isf_2D)
    elif HUB:
        mparam0 = mparam
        depth_of_int = geometry_isf_2D['ice_draft_pos'].where(geometry_isf_2D['ice_draft_pos']<geometry_isf_2D['HUB_depth'], geometry_isf_2D['HUB_depth'])
    else:
        mparam0 = mparam
        # either the depth of the draft or the deepest entrance point
        depth_of_int = geometry_isf_2D['ice_draft_pos'].where(geometry_isf_2D['ice_draft_pos']<geometry_isf_1D['front_bot_depth_max'], geometry_isf_1D['front_bot_depth_max']) # ice draft depth or deepest entrance depth
    
    #print('here1')
    # extrapolate temperature and salinity at the given depth
    T0 = T_S_profile['theta_ocean'].sel(Nisf=kisf).squeeze().interp(depth=depth_of_int)
    S0 = T_S_profile['salinity_ocean'].sel(Nisf=kisf).squeeze().interp(depth=depth_of_int)

    #print('here2')
    # compute the freezing temperature at the ice draft depth
    Tf = freezing_temperature(S0, -geometry_isf_2D['ice_draft_pos'])
    
    #print('here3')
    # compute thermal forcing
    thermal_forcing = T0 - Tf
    thermal_forcing_avg = uf.weighted_mean(thermal_forcing, ['mask_coord'], geometry_isf_2D['isfdraft_conc']) #weighted mean 
    S_avg = uf.weighted_mean(S0, ['mask_coord'], geometry_isf_2D['isfdraft_conc'])
    
    if U_param and 'mixed' in mparam0:
        U_factor = (c_po / L_i) * beta_coeff_lazero * (g/(2*abs(f_coriolis))) * S_avg
    elif U_param and 'mixed' not in mparam0:
        U_factor = (c_po / L_i) * beta_coeff_lazero * (g/(2*abs(f_coriolis))) * S0
    else:
        U_factor = melt_factor

    #print('here4')
    # Melt in m/s (meters of ice per s), positive if ice ablation
    if mparam0 == 'linear_local':
        melt_rate = linear_local_param(gamma, melt_factor, thermal_forcing) 
    elif mparam0 == 'quadratic_local':
        melt_rate = quadratic_local_param(gamma, melt_factor, thermal_forcing, U_factor)
    elif mparam0 == 'quadratic_local_locslope':
        local_angle = geometry_isf_2D['alpha'].sel(option='local')
        melt_rate = quadratic_mixed_slope(gamma, melt_factor, thermal_forcing, thermal_forcing, U_factor, local_angle)
    elif mparam0 == 'quadratic_local_cavslope':
        local_angle = geometry_isf_2D['alpha'].sel(option='cavity')
        melt_rate = quadratic_mixed_slope(gamma, melt_factor, thermal_forcing, thermal_forcing, U_factor, local_angle)
    elif mparam0 == 'quadratic_mixed_mean':  
        melt_rate = quadratic_mixed_mean(gamma, melt_factor, thermal_forcing, thermal_forcing_avg, U_factor)
    elif mparam0 == 'quadratic_mixed_locslope':
        local_angle = geometry_isf_2D['alpha'].sel(option='local')
        melt_rate = quadratic_mixed_slope(gamma, melt_factor, thermal_forcing, thermal_forcing_avg, U_factor, local_angle)
    elif mparam0 == 'quadratic_mixed_cavslope':
        local_angle = geometry_isf_2D['alpha'].sel(option='cavity')
        melt_rate = quadratic_mixed_slope(gamma, melt_factor, thermal_forcing, thermal_forcing_avg, U_factor, local_angle)
        
    return melt_rate




def calculate_melt_rate_2D_plumes_1isf(kisf, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, E0):

    """
    Function to compute melt from plume parameterisations for one ice shelf.
    'lazero19': uses average of hydrographic properties extrapolated to local ice draft depth as ambient temperature and salinity
    'lazero19_modif': Modification presented in Burgard et al. 2022
        
    Parameters
    ----------
    kisf : int
        Ice shelf ID for the ice shelf of interest. 
    T_S_profile : xarray.Dataset
        Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    geometry_info_1D : xarray.Dataset
        Dataset containing relevant 1D geometrical information.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    mparam : str
        Parameterisation to be applied.
    gamma : float
        Gamma to be tuned.
    E0 : float
        Entrainment coefficient.
        
    Returns
    -------
    melt_rate : xr.DataArray
        Horizontal distribution of the melt rate in m/s for the ice shelf of interest.
    """  
        
    geometry_isf_2D = uf.choose_isf(geometry_info_2D,isf_stack_mask, kisf)
    zGL_isf = geometry_isf_2D['zGL']
    alpha_isf = geometry_isf_2D['alpha']
    ice_draft_pos_isf = geometry_isf_2D['ice_draft_pos']
    dGL_isf = geometry_isf_2D['dGL']
    conc_isf = geometry_isf_2D['isfdraft_conc']
    isf_mask = geometry_isf_2D['ISF_mask']
    isf_cell_area = geometry_isf_2D['grid_cell_area_weighted']
    
    front_bot_dep_max_isf = geometry_info_1D['front_bot_depth_max'].sel(Nisf=kisf)
    theta_isf = T_S_profile['theta_ocean'].sel(Nisf=kisf)
    salinity_isf = T_S_profile['salinity_ocean'].sel(Nisf=kisf)
    
    zGL_cavity_pos = -1*zGL_isf.sel(option='cavity') 

    if mparam == 'lazero19':
        
        moption = 'lazero'
        zGL = zGL_isf.sel(option=moption)
        zGL_pos = -1*zGL
        alpha = alpha_isf.sel(option=moption)
        
        #### if we want to take the mean of the profile in front of the ice shelf
        ## interpolate T and S profiles on regular depth axis for the mean
        #depth_axis_loc = xr.DataArray(data=np.arange(zGL_pos.max()+1),dims=['depth'])
        ## make sure to not use T and S lower than the maximum front depth
        #depth_axis_loc = depth_axis_loc.where(depth_axis_loc<front_bot_dep_max_isf,front_bot_dep_max_isf)
        ## do the interpolation
        #theta_1m = theta_isf.interp({'depth': depth_axis_loc})
        #salinity_1m = salinity_isf.interp({'depth': depth_axis_loc})
        ## make a mean over the whole profile
        #theta_mean = theta_1m.mean('depth')
        #salinity_mean = salinity_1m.mean('depth')

        # depth_of_int to which to interpolate the T and S at ice draft depth
        depth_of_int = ice_draft_pos_isf.where(ice_draft_pos_isf < front_bot_dep_max_isf, front_bot_dep_max_isf)
        # Local temperature and salinity at ice draft depth
        T0_loc = theta_isf.interp({'depth': depth_of_int}).drop('depth')
        S0_loc = salinity_isf.interp({'depth': depth_of_int}).drop('depth')
        
        # Mean temperature and salinity over whole ice-shelf base (cavity mean)
        T0_mean_cav = uf.weighted_mean(T0_loc, ['mask_coord'], conc_isf)
        S0_mean_cav = uf.weighted_mean(S0_loc, ['mask_coord'], conc_isf)
        #print(kisf)
        #print(T0_mean_cav)
        #print(S0_mean_cav)
        
        melt_rate = plume_param(T0_mean_cav, S0_mean_cav, -1*ice_draft_pos_isf, zGL.squeeze().drop('option'), alpha.squeeze().drop('option'), 
                              gamma, E0)
    
    elif mparam == 'lazero19_modif':
        melt_rate = plume_param_modif(theta_isf,salinity_isf,
                                       ice_draft_pos_isf,front_bot_dep_max_isf,zGL_isf,alpha_isf,conc_isf,dGL_isf,
                                       gamma,E0)
    
    return melt_rate




def calculate_melt_rate_2D_boxes_1isf(kisf, T_S_profile, geometry_info_2D, geometry_info_1D, box_charac_all_2D, box_charac_all_1D, isf_stack_mask, mparam, box_tot_nb, gamma, C,
                                      angle_option, pism_version, T_corrections):
    
    """
    Function to compute melt from box parameterisations for one ice shelf.
        
    Parameters
    ----------
    kisf : int
        Ice shelf ID for the ice shelf of interest. 
    T_S_profile : xarray.Dataset
        Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    geometry_info_1D : xarray.Dataset
        Dataset containing relevant 1D geometrical information.
    box_charac_all_2D : xarray.Dataset
        Dataset containing relevant 2D box characteristics for all ice shelves.
    box_charac_all_1D : xarray.Dataset
        Dataset containing relevant 1D box characteristics for all ice shelves.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    mparam : str
        Parameterisation to be applied.
    box_tot_nb : int
        Total number of boxes
    gamma : float
        Gamma to be tuned.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    T_corrections : Boolean
        If ``True``, use regional corrections (preferentially only when using "new" Reese parameters).

    Returns
    -------
    m_out : xr.DataArray
        Horizontal distribution of the melt rate in m ice/s for the ice shelf of interest.
    """  
    
    # select indices corresponding to the ice shelf points
    #print(angle_option)
    geometry_isf_2D = uf.choose_isf(geometry_info_2D.sel(option=angle_option),isf_stack_mask, kisf)
    conc_isf = geometry_isf_2D['isfdraft_conc']
    isf_cell_area = geometry_isf_2D['grid_cell_area_weighted']
    ice_draft_neg_isf = -1*geometry_isf_2D['ice_draft_pos']
    deepest_GL = -1*uf.choose_isf(geometry_info_2D['zGL'].sel(option='cavity'),isf_stack_mask, kisf).max()
    front_bot_dep_max_isf = geometry_info_1D['front_bot_depth_max'].sel(Nisf=kisf)
    front_bot_dep_avg_isf = geometry_info_1D['front_bot_depth_avg'].sel(Nisf=kisf)
    
    # box characteristics
    box_charac_1D_isf = box_charac_all_1D.sel(Nisf=kisf)
    box_charac_2D_isf = uf.choose_isf(box_charac_all_2D, isf_stack_mask, kisf)

    # Entering temperature and salinity profiles
    #depth_of_int = geometry_info_1D['front_bot_depth_max'].sel(Nisf=kisf)
    depth_of_int = geometry_info_1D['front_bot_depth_avg'].sel(Nisf=kisf)
    #depth_of_int = deepest_GL.where(deepest_GL < front_bot_dep_avg_isf, front_bot_dep_avg_isf)
    #print(depth_of_int.values)
    T_isf = T_S_profile['theta_ocean'].sel(Nisf=kisf).interp({'depth': depth_of_int}).drop('depth')
    S_isf = T_S_profile['salinity_ocean'].sel(Nisf=kisf).interp({'depth': depth_of_int}).drop('depth')
    
    if T_corrections:
        T_isf = T_correction_PICO(geometry_info_1D['isf_name'].sel(Nisf=kisf), T_isf)

        
    # compute the melt rate
    T_out, S_out, m_out = PICO_param(T_isf,S_isf,
                                   box_charac_2D_isf['box_location'],
                                   box_charac_1D_isf['box_depth_below_surface'],
                                   box_charac_1D_isf['box_area'],
                                   box_tot_nb, ['mask_coord'],
                                   isf_cell_area,
                                   gamma, C,
                                   ice_draft_neg_isf,
                                   pism_version)

    return m_out

def calculate_melt_rate_2D_picop_1isf(kisf, T_S_profile, geometry_info_2D, geometry_info_1D, box_charac_all_2D, box_charac_all_1D, isf_stack_mask, mparam, box_tot_nb, gamma_pico, gamma_plume, C, E0, 
                                      angle_option, pism_version, picop_opt):
    
    """
    Function to compute melt from box parameterisations for one ice shelf.
        
    Parameters
    ----------
    kisf : int
        Ice shelf ID for the ice shelf of interest. 
    T_S_profile : xarray.Dataset
        Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    geometry_info_1D : xarray.Dataset
        Dataset containing relevant 1D geometrical information.
    box_charac_all_2D : xarray.Dataset
        Dataset containing relevant 2D box characteristics for all ice shelves.
    box_charac_all_1D : xarray.Dataset
        Dataset containing relevant 1D box characteristics for all ice shelves.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    mparam : str
        Parameterisation to be applied.
    box_tot_nb : int
        Total number of boxes
    gamma_pico : float
        Gamma to be used in PICO part.
    gamma_plume : float
        Gamma to be used in plume part.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    E0 : float
        Entrainment coefficient.
    angle_option : str
        Slope to be used, choice between "cavity" (cavity slope), "lazero" (lazeroms18), "local" (local slope)
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    picop_opt: str
        Can be ``2019``, ``2018`` , depending on if you want to use PICOP with analytical plume param or with empirical plume param.
        
    Returns
    -------
    m_out : xr.DataArray
        Horizontal distribution of the melt rate in m ice/s for the ice shelf of interest.
    """  
    
    # select indices corresponding to the ice shelf points
    #print(angle_option)
    geometry_isf_2D = uf.choose_isf(geometry_info_2D.sel(option=angle_option),isf_stack_mask, kisf)
    zGL_isf = geometry_isf_2D['zGL']
    alpha_isf = geometry_isf_2D['alpha']
    ice_draft_neg_isf = -1*geometry_isf_2D['ice_draft_pos']
    conc_isf = geometry_isf_2D['isfdraft_conc']
    isf_cell_area = geometry_isf_2D['grid_cell_area_weighted']

    # box characteristics
    box_charac_1D_isf = box_charac_all_1D.sel(Nisf=kisf)
    box_charac_2D_isf = uf.choose_isf(box_charac_all_2D, isf_stack_mask, kisf)

    # Entering temperature and salinity profiles
    #depth_of_int = geometry_info_1D['front_bot_depth_max'].sel(Nisf=kisf)
    depth_of_int = geometry_info_1D['front_bot_depth_avg'].sel(Nisf=kisf)
    T_isf = T_S_profile['theta_ocean'].sel(Nisf=kisf).interp({'depth': depth_of_int}).drop('depth')
    S_isf = T_S_profile['salinity_ocean'].sel(Nisf=kisf).interp({'depth': depth_of_int}).drop('depth')

    # compute the melt rate
    T_out, S_out, m_out = PICOP_param(T_isf,S_isf,
                                       box_charac_2D_isf['box_location'],
                                       box_charac_1D_isf['box_depth_below_surface'],
                                       box_charac_1D_isf['box_area'],
                                       box_tot_nb, ['mask_coord'],
                                       isf_cell_area,
                                       gamma_pico, gamma_plume, E0, C, 
                                       zGL_isf,alpha_isf,ice_draft_neg_isf,
                                       pism_version, picop_opt)

    return m_out


def calculate_melt_rate_2D_1isf(kisf, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, 
                                U_param=True, C=None, E0=None, angle_option='lazero',
                                box_charac_2D=None, box_charac_1D=None, box_tot=None, box_tot_option='box_nb_tot', pism_version='no', picop_opt='no', gamma_plume=None, 
                                T_corrections=False, HUB=False):

        """
        Wrap function to point to the right melt parameterisation for one ice shelf.

        Parameters
        ----------
        kisf : int
            Ice shelf ID for the ice shelf of interest. 
        T_S_profile : xarray.Dataset
            Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
        geometry_info_2D : xarray.Dataset
            Dataset containing relevant 2D geometrical information.
        geometry_info_1D : xarray.Dataset
            Dataset containing relevant 1D geometrical information.
        isf_stack_mask : xarray.DataArray
            DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
        mparam : str
            Parameterisation to be applied.
        gamma : float
            Gamma to be tuned. Will be used as gamma_pico in PICOP.
        U_param : Boolean
            If ``True`` we use the complex parameterisation of U, if ``False``, this is "only" equal to (rho_sw * c_po) / (rho_i * L_i). Relevant for simple parameterisations only.
        C : float
            Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
        E0 : float
            Entrainment coefficient.
        angle_option : str
            Slope to be used, choice between "cavity" (cavity slope), "lazero" (lazeroms18), "local" (local slope)
        box_charac_2D : xarray.Dataset
            Dataset containing relevant 2D box characteristics for all ice shelves.
        box_charac_1D : xarray.Dataset
            Dataset containing relevant 1D box characteristics for all ice shelves.
        box_tot : int
            Either the total number of boxes being used if box_tot_option='box_nb_tot' or the configuration to use if box_tot_option='nD_config'.
        box_tot_option : str
            Defines how ``box_tot``should be interpreted. Can be either 'box_nb_tot', then ``box_tot`` is the total number of boxes or 'nD_config', then ``box_tot`` is the configuration to use.
        pism_version: str
            Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
        picop_opt : str
            Can be ``2019``, ``2018`` or ``no``, depending on if you want to use PICOP with analytical plume param, with empirical plume param or the original PICO without plume. 
        gamma_plume : float
            Gamma to be used in the plume part of PICOP.
        T_corrections : Boolean
            If ``True``, use regional corrections (preferentially only when using "new" Reese parameters).

        Returns
        -------
        melt_rate_2D_isf : xr.DataArray
            Horizontal distribution of the melt rate in m ice/s for the ice shelf of interest.

        """  
        
        filled_TS = T_S_profile.ffill(dim='depth')
        if mparam in ['linear_local', 'quadratic_local', 'quadratic_local_locslope','quadratic_local_cavslope',
                      'quadratic_mixed_mean', 'quadratic_mixed_locslope','quadratic_mixed_cavslope',
                     'linear_local_bottom', 'quadratic_local_bottom', 'quadratic_local_locslope_bottom','quadratic_local_cavslope_bottom',
                      'quadratic_mixed_mean_bottom', 'quadratic_mixed_locslope_bottom','quadratic_mixed_cavslope_bottom']:
            #print('Computing simple '+mparam+' melt rates and writing to file')
            melt_rate_2D_isf = calculate_melt_rate_2D_simple_1isf(kisf, filled_TS, geometry_info_2D, geometry_info_1D, 
                                                                  isf_stack_mask, mparam, gamma, U_param, HUB)
        elif mparam in ['lazero19', 'lazero19_modif']:
            #print('Computing plume '+mparam+' melt rates and writing to file')
            if E0 is None:
                print('Careful! I did not receive a E0, I am using the default value!')
                E0 = E0_lazero
            melt_rate_2D_isf = calculate_melt_rate_2D_plumes_1isf(kisf, filled_TS, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, E0)
        elif picop_opt in ['2018','2019']:
            if C is None:
                print('Careful! I did not receive a C, I am using the default value!')
                C = C_pico
            if E0 is None:
                print('Careful! I did not receive a E0, I am using the default value!')
                E0 = E0_lazero
            if gamma_plume is None:
                print('Careful! I did not receive a gamma_plume, I am using the default value!')
                gamma_plume = gamma_eff_T_lazero
            if box_tot_option == 'box_nb_tot':
                box_tot_nb = box_tot
            elif box_tot_option == 'nD_config':
                box_tot_nb = box_charac_1D['nD_config'].sel(Nisf=kisf).sel(config=box_tot).values
            melt_rate_2D_isf = calculate_melt_rate_2D_picop_1isf(kisf, filled_TS, geometry_info_2D, geometry_info_1D, box_charac_2D, box_charac_1D, isf_stack_mask, 
                                                                 mparam, box_tot_nb, gamma, gamma_plume, C, E0, 
                                                                 angle_option, pism_version, picop_opt)
            
        elif picop_opt=='no':
            #print('Computing box '+mparam+' melt rates and writing to file')
            if C is None:
                print('Careful! I did not receive a C, I am using the default value!')
                C = C_pico
            if box_tot_option == 'box_nb_tot':
                box_tot_nb = box_tot
            elif box_tot_option == 'nD_config':
                box_tot_nb = box_charac_1D['nD_config'].sel(Nisf=kisf).sel(config=box_tot).values
            else:
                print('Careful! I received the following as box_tot_nb: ',box_tot)
            melt_rate_2D_isf = calculate_melt_rate_2D_boxes_1isf(kisf, filled_TS, geometry_info_2D, geometry_info_1D, box_charac_2D, box_charac_1D, isf_stack_mask, 
                                                                 mparam, box_tot_nb, gamma, C, angle_option, pism_version, T_corrections)
            
        return melt_rate_2D_isf
    
    
    
    
def calculate_melt_rate_2D_all_isf(nisf_list, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, 
                                   U_param=True, C=None, E0=None, angle_option='lazero',
                                   box_charac_2D=None, box_charac_1D=None, box_tot=None, box_tot_option='box_nb_tot', pism_version='no', picop_opt='no',
                                   gamma_plume=None, T_corrections=False,
                                   options_2D=['melt_m_ice_per_y','melt_m_we_per_y'],
                                   HUB=False,
                                   verbose=True):
    
    
    """
    Wrap function to loop over all ice shelves and combine result to a map.

    Parameters
    ----------
    nisf_list : array of int
        List containing the ice shelf IDs for all ice shelves of interest. 
    T_S_profile : xarray.Dataset
        Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    geometry_info_1D : xarray.Dataset
        Dataset containing relevant 1D geometrical information.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    mparam : str
        Parameterisation to be applied.
    gamma : float
        Gamma to be tuned.
    U_param : Boolean
        If ``True`` we use the complex parameterisation of U, if ``False``, this is "only" equal to (rho_sw * c_po) / (rho_i * L_i). Relevant for simple parameterisations only.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    E0 : float
        Entrainment coefficient.
    angle_option : str
        Slope to be used, choice between "cavity" (cavity), "lazero" (lazeroms18), "local" (local)
    box_charac_2D : xarray.Dataset
        Dataset containing relevant 2D box characteristics for all ice shelves.
    box_charac_1D : xarray.Dataset
        Dataset containing relevant 1D box characteristics for all ice shelves.
    box_tot : int
        Either the total number of boxes being used if box_tot_option='box_nb_tot' or the configuration to use if box_tot_option='nD_config'.
    box_tot_option : str
        Defines how ``box_tot``should be interpreted. Can be either 'box_nb_tot', then ``box_tot`` is the total number of boxes or 'nD_config', then ``box_tot`` is the configuration to use.
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    picop_opt : str
        Can be ``yes`` or ``no``, depending on if you want to use PICOP or the original PICO. 
    gamma_plume : float
        Gamma to be used in the plume part of PICOP.
    T_corrections : Boolean
        If ``True``, use regional corrections (preferentially only when using "new" Reese parameters).
    options_2D : list of str
        2D variables to be written out. Possible options: 'melt_m_ice_per_y','melt_m_we_per_y'. 'melt_m_ice_per_s' is always written out!
    verbose : Boolean
        ``True`` if you want the program to keep you posted on where it is in the calculation.

    Returns
    -------
    ds_melt_rate_2D_unstacked : xarray.Dataset
        Horizontal distribution of the melt rate for all ice shelves in one map.

    """  
    
    if verbose:
        print('LET US START WITH THE 2D VALUES')

    n = 0
    
    if verbose:
        list_loop = tqdm(nisf_list)
    else:
        list_loop = nisf_list
        
    for kisf in list_loop:
        #print(kisf, n)
        
        melt_rate_2D_isf = calculate_melt_rate_2D_1isf(kisf, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, 
                                                       U_param, C, E0, angle_option, 
                                                       box_charac_2D, box_charac_1D, box_tot, box_tot_option, pism_version, picop_opt, gamma_plume, T_corrections, HUB)
        
        if n == 0:
            ds_melt_rate_2D_all = melt_rate_2D_isf.squeeze().drop('Nisf')
        else:
            ds_melt_rate_2D_all = ds_melt_rate_2D_all.combine_first(melt_rate_2D_isf).squeeze().drop('Nisf')
        n = n+1
        
        del melt_rate_2D_isf
        
    ds_melt_rate_2D_unstacked = uf.bring_back_to_2D(ds_melt_rate_2D_all)
    ds_melt_rate_2D_unstacked = ds_melt_rate_2D_unstacked.chunk({'x': 1000, 'y': 1000}).reindex_like(geometry_info_2D['ISF_mask'].chunk({'x': 1000, 'y': 1000})).to_dataset(name='melt_m_ice_per_s')

    if 'melt_m_ice_per_y' in options_2D:
        ds_melt_rate_2D_unstacked['melt_m_ice_per_y'] = ds_melt_rate_2D_unstacked['melt_m_ice_per_s'] * yearinsec
    if 'melt_m_we_per_y' in options_2D:
        ds_melt_rate_2D_unstacked['melt_m_we_per_y'] = convert_m_ice_to_m_water(ds_melt_rate_2D_unstacked['melt_m_ice_per_s'], geometry_info_2D['grid_cell_area_weighted']) * yearinsec

    
    return ds_melt_rate_2D_unstacked



def calculate_melt_rate_1D_all_isf(nisf_list, ds_melt_rate_2D_all, geometry_info_2D, isf_stack_mask, 
                                    options_1D=['melt_m_ice_per_y_avg', 'melt_m_ice_per_y_min', 'melt_m_ice_per_y_max', 'melt_we_per_y_tot',
                                                'melt_we_per_y_avg','melt_Gt_per_y_tot'],
                                    verbose=True):
    
    
    """
    Function to transform 2D melt information into 1D values.

    Parameters
    ----------
    nisf_list : array of int
        List containing the ice shelf IDs for all ice shelves of interest. 
    ds_melt_rate_2D_all : xarray.Dataset
        Dataset containing 2D melt information.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    options_1D : list of str
        1D variables to be written out. Possible options: 'melt_m_ice_per_y_avg', 'melt_m_ice_per_y_min', 'melt_m_ice_per_y_max', 'melt_we_per_y_tot'. 'melt_m_ice_per_y_tot' is always written out!
    verbose : Boolean
        ``True`` if you want the program to keep you posted on where it is in the calculation.
    
    Returns
    -------
    ds_melt_rates_1D_tot : xarray.Dataset
        1D metrics about the melt rate for all ice shelves.

    """  
    
    if verbose:
        print("NOW LET US SUMMARIZE WITH THE 1D VALUES")
        
    ds_melt_rates_1D_all = [ ]
    
    if verbose:
        list_loop = tqdm(nisf_list)
    else:
        list_loop = nisf_list
        
    for kisf in list_loop:
        ds_2D_isf = uf.choose_isf(ds_melt_rate_2D_all, isf_stack_mask, kisf)
        geometry_isf_2D = uf.choose_isf(geometry_info_2D, isf_stack_mask, kisf)

        #print('here7')
        ds_melt_rates_1D_isf = ds_2D_isf['melt_m_ice_per_y'].sum(dim=['mask_coord']).to_dataset(name='melt_m_ice_per_y_tot')
        if 'melt_m_ice_per_y_avg' in options_1D:
            ds_melt_rates_1D_isf['melt_m_ice_per_y_avg'] = uf.weighted_mean(ds_2D_isf['melt_m_ice_per_y'],['mask_coord'], geometry_isf_2D['isfdraft_conc'])
        if 'melt_m_ice_per_y_min' in options_1D:
            ds_melt_rates_1D_isf['melt_m_ice_per_y_min'] = ds_2D_isf['melt_m_ice_per_y'].min(dim=['mask_coord'])
        if 'melt_m_ice_per_y_max' in options_1D:
            ds_melt_rates_1D_isf['melt_m_ice_per_y_max'] = ds_2D_isf['melt_m_ice_per_y'].max(dim=['mask_coord'])
        if 'melt_we_per_y_tot' in options_1D:
            ds_melt_rates_1D_isf['melt_we_per_y_tot'] = ds_2D_isf['melt_m_we_per_y'].sum(dim=['mask_coord'])
        if 'melt_we_per_y_avg' in options_1D:
            ds_melt_rates_1D_isf['melt_we_per_y_avg'] = uf.weighted_mean(ds_2D_isf['melt_m_we_per_y'], ['mask_coord'], geometry_isf_2D['isfdraft_conc'])
        if 'melt_Gt_per_y_tot' in options_1D:
            ds_melt_rates_1D_isf['melt_Gt_per_y_tot'] = (ds_2D_isf['melt_m_ice_per_y']*geometry_isf_2D['grid_cell_area_weighted']).sum(dim=['mask_coord']) * rho_i / 10**12
        ds_melt_rates_1D_all.append(ds_melt_rates_1D_isf)
        del ds_2D_isf
        del geometry_isf_2D
        del ds_melt_rates_1D_isf
    
    ds_melt_rates_1D_tot = xr.concat(ds_melt_rates_1D_all, dim='Nisf')
    
    return ds_melt_rates_1D_tot


def calculate_melt_rate_1D_and_2D_all_isf(nisf_list, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, 
                                          U_param=True, C=None, E0=None, angle_option='lazero',
                                          box_charac_2D=None, box_charac_1D=None, box_tot=None, box_tot_option='box_nb_tot', pism_version='no', picop_opt='no',
                                          gamma_plume=None, T_corrections=False,
                                          options_2D=['melt_m_ice_per_y','melt_m_we_per_y'],
                                          options_1D=['melt_m_ice_per_y_avg', 'melt_m_ice_per_y_min', 'melt_m_ice_per_y_max', 'melt_we_per_y_tot',
                                                     'melt_we_per_y_avg','melt_Gt_per_y_tot'],
                                          HUB=False,
                                          verbose=True):
    
    """
    Function to process input information and call the 2D and 1D functions to compute melt rate variables.

    Parameters
    ----------
    nisf_list : array of int
        List containing the ice shelf IDs for all ice shelves of interest. 
    T_S_profile : xarray.Dataset
        Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    geometry_info_1D : xarray.Dataset
        Dataset containing relevant 1D geometrical information.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    mparam : str
        Parameterisation to be applied.
    gamma : float
        Gamma to be tuned.
    U_param : Boolean
        If ``True`` we use the complex parameterisation of U, if ``False``, this is "only" equal to (rho_sw * c_po) / (rho_i * L_i). Relevant for simple parameterisations only.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    E0 : float
        Entrainment coefficient.
    angle_option : str
        Slope to be used, choice between "cavity" (cavity), "lazero" (lazeroms18), "local" (local)
    box_charac_2D : xarray.Dataset
        Dataset containing relevant 2D box characteristics for all ice shelves.
    box_charac_1D : xarray.Dataset
        Dataset containing relevant 1D box characteristics for all ice shelves.
    box_tot : int
        Either the total number of boxes being used if box_tot_option='box_nb_tot' or the configuration to use if box_tot_option='nD_config'.
    box_tot_option : str
        Defines how ``box_tot``should be interpreted. Can be either 'box_nb_tot', then ``box_tot`` is the total number of boxes or 'nD_config', then ``box_tot`` is the configuration to use.
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    picop_opt : str
        Can be ``yes`` or ``no``, depending on if you want to use PICOP or the original PICO. 
    gamma_plume : float
        Gamma to be used in the plume part of PICOP.
    T_corrections : Boolean
        If ``True``, use regional corrections (preferentially only when using "new" Reese parameters).
    options_2D : list of str
        2D variables to be written out. Possible options: 'melt_m_ice_per_y','melt_m_we_per_y'. 'melt_m_ice_per_s' is always written out!
    options_1D : list of str
        1D variables to be written out. Possible options: 'melt_m_ice_per_y_avg', 'melt_m_ice_per_y_min', 'melt_m_ice_per_y_max', 'melt_we_per_y_tot'. 'melt_m_ice_per_y_tot' is always written out!
    verbose : Boolean
        ``True`` if you want the program to keep you posted on where it is in the calculation.

    Returns
    -------
    ds_2D : xarray.Dataset
        Horizontal distribution of the melt rate for all ice shelves in one map.
    ds_1D : xarray.Dataset
        1D metrics about the melt rate for all ice shelves.
    """  
    
    if verbose:
        time_start = time.time()
        print('WELCOME! AS YOU WISH, I WILL COMPUTE MELT RATES FOR THE PARAMETERISATION "'+mparam+'" FOR '+str(len(nisf_list))+' ICE SHELVES')
    
    ds_2D = calculate_melt_rate_2D_all_isf(nisf_list, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, U_param, C, E0, angle_option,
                                           box_charac_2D, box_charac_1D, box_tot, box_tot_option, pism_version, picop_opt, gamma_plume, T_corrections,
                                           options_2D, HUB, verbose)
    
    ds_1D = calculate_melt_rate_1D_all_isf(nisf_list, ds_2D, geometry_info_2D, isf_stack_mask, options_1D, verbose)
    
    if verbose:
        timelength = time.time() - time_start
        print("I AM DONE! IT TOOK: "+str(round(timelength,2))+" seconds.")
        
    return ds_2D, ds_1D


def calculate_melt_rate_Gt_and_box1_all_isf(nisf_list, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam, gamma, 
                                          U_param=True, C=None, E0=None, angle_option='lazero',
                                          box_charac_2D=None, box_charac_1D=None, box_tot=None, box_tot_option='box_nb_tot', pism_version='no', picop_opt='no',
                                          gamma_plume=None, T_corrections=False,
                                          tuning_mode=False, 
                                          HUB=False,
                                          verbose=True):
    
    """
    Function to process input information and call the 2D and 1D functions to compute melt rate variables.

    Parameters
    ----------
    nisf_list : array of int
        List containing the ice shelf IDs for all ice shelves of interest. 
    T_S_profile : xarray.Dataset
        Dataset containing temperature (in degrees C) and salinity (in psu) input profiles.
    geometry_info_2D : xarray.Dataset
        Dataset containing relevant 2D geometrical information.
    geometry_info_1D : xarray.Dataset
        Dataset containing relevant 1D geometrical information.
    isf_stack_mask : xarray.DataArray
        DataArray containing the stacked coordinates of the ice shelves (to make computing faster).
    mparam : str
        Parameterisation to be applied.
    gamma : float
        Gamma to be tuned.
    U_param : Boolean
        If ``True`` we use the complex parameterisation of U, if ``False``, this is "only" equal to (rho_sw * c_po) / (rho_i * L_i). Relevant for simple parameterisations only.
    C : float
        Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6.
    E0 : float
        Entrainment coefficient.
    angle_option : str
        Slope to be used, choice between "cavity" (cavity), "lazero" (lazeroms18), "local" (local)
    box_charac_2D : xarray.Dataset
        Dataset containing relevant 2D box characteristics for all ice shelves.
    box_charac_1D : xarray.Dataset
        Dataset containing relevant 1D box characteristics for all ice shelves.
    box_tot : int
        Either the total number of boxes being used if box_tot_option='box_nb_tot' or the configuration to use if box_tot_option='nD_config'.
    box_tot_option : str
        Defines how ``box_tot``should be interpreted. Can be either 'box_nb_tot', then ``box_tot`` is the total number of boxes or 'nD_config', then ``box_tot`` is the configuration to use.
    pism_version: str
        Can be ``yes`` or ``no``, depending on if you want to use the PICO version as implemented in PISM or the original PICO (uniform box melt). See Sec. 2.4 by Reese et al. 2018 for more info. 
    picop_opt : str
        Can be ``yes`` or ``no``, depending on if you want to use PICOP or the original PICO. 
    gamma_plume : float
        Gamma to be used in the plume part of PICOP.
    T_corrections : Boolean
        If ``True``, use regional corrections (preferentially only when using "new" Reese parameters).
    tuning_mode : Boolean
        If ``True``, only compute integrated melt.
    verbose : Boolean
        ``True`` if you want the program to keep you posted on where it is in the calculation.

    Returns
    -------
    out_1D : xarray.Dataset
        Containing the melt in Gt/yr for each ice shelf and the mean melt rate in m/yr i
    """  
    
    if verbose:
        time_start = time.time()
        print('WELCOME! AS YOU WISH, I WILL COMPUTE THE EVALUATION METRICS FOR THE PARAMETERISATION "'+mparam+'" FOR '+str(len(nisf_list))+' ICE SHELVES')
    
    
    if verbose:
        list_loop = tqdm(nisf_list)
    else:
        list_loop = nisf_list

    if box_charac_2D and box_charac_1D:
        box_loc_config2 = box_charac_2D['box_location'].sel(box_nb_tot=box_charac_1D['nD_config'].sel(config=2))
        box1 = box_loc_config2.where(box_loc_config2==1).isel(Nisf=1).drop('Nisf')
    elif not box_charac_2D:
        return print('You have not given me the 2D box characteristics! :( ')
    elif not box_charac_1D:
        return print('You have not given me the 1D box characteristics! :( ')

    melt1D_Gt_per_yr_list = []
    if not tuning_mode:
        melt1D_myr_box1_list = []

    for kisf in list_loop:
        #print(kisf, n)

        geometry_isf_2D = uf.choose_isf(geometry_info_2D,isf_stack_mask, kisf)

        melt_rate_2D_isf = calculate_melt_rate_2D_1isf(kisf, T_S_profile, geometry_info_2D, geometry_info_1D, isf_stack_mask, mparam,
                                                       gamma, 
                                                       U_param, C, E0, angle_option, 
                                                       box_charac_2D, box_charac_1D, box_tot, box_tot_option, 
                                                       pism_version, picop_opt, gamma_plume, T_corrections, HUB)

        melt_rate_2D_isf_m_per_y = melt_rate_2D_isf * yearinsec
        melt_rate_1D_isf_Gt_per_y = (melt_rate_2D_isf_m_per_y * geometry_isf_2D['grid_cell_area_weighted']).sum(dim=['mask_coord']) * rho_i / 10**12
        if 'option' in melt_rate_1D_isf_Gt_per_y.coords:
            melt_rate_1D_isf_Gt_per_y = melt_rate_1D_isf_Gt_per_y.drop('option')
        melt1D_Gt_per_yr_list.append(melt_rate_1D_isf_Gt_per_y)
        
        if not tuning_mode:
            box_loc_config_stacked = uf.choose_isf(box1, isf_stack_mask, kisf)
            param_melt_2D_box1_isf = melt_rate_2D_isf_m_per_y.where(np.isfinite(box_loc_config_stacked))
            melt_rate_1D_isf_myr_box1_mean = uf.weighted_mean(param_melt_2D_box1_isf,['mask_coord'], geometry_isf_2D['isfdraft_conc'])     
            if 'option' in melt_rate_1D_isf_myr_box1_mean.coords:
                melt_rate_1D_isf_myr_box1_mean = melt_rate_1D_isf_myr_box1_mean.drop('option')

            melt1D_myr_box1_list.append(melt_rate_1D_isf_myr_box1_mean)

    melt1D_Gt_per_yr = xr.concat(melt1D_Gt_per_yr_list, dim='Nisf')
    if not tuning_mode:
        melt1D_myr_box1 = xr.concat(melt1D_myr_box1_list, dim='Nisf')

    melt1D_Gt_per_yr_ds = melt1D_Gt_per_yr.to_dataset(name='melt_1D_Gt_per_y')
    if not tuning_mode:
        melt1D_myr_box1_ds = melt1D_myr_box1.to_dataset(name='melt_1D_mean_myr_box1')
        out_1D = xr.merge([melt1D_Gt_per_yr_ds, melt1D_myr_box1_ds])
    else:
        out_1D = melt1D_Gt_per_yr_ds
        
    if verbose:
        timelength = time.time() - time_start
        print("I AM DONE! IT TOOK: "+str(round(timelength,2))+" seconds.")
    
    return out_1D