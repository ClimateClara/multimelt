
##### GENERAL

yearinsec = 86400.0 * 365.2422 
g = 9.81 # m s-2 Gravitational acceleration
f_coriolis = 1.4*10**-4 #s−1

##### THERMODYNAMIC CONSTANTS

## coefficients for freezing temperature
l_1 = -0.0575 #°C per psu; Reese et al: -0.0572
l_2 = 0.0832 #°C; Reese et al: 0.0788
l_3 = 7.59*10**-4 #°C per m; Reese et al: 7.77 * 10**-8 (per Pa)

## densities and heat capacities
rho_sw = 1028. # kg m-3
c_po = 3974. # J kg-1 K-1
rho_i = 917. # kg m-3
L_i = 3.34 * 10**5# J kg-1
rho_fw = 1000.

melt_factor = (rho_sw * c_po) / (rho_i * L_i) # K-1

###### PICO constants

# from Reese et al. 2018
C_pico = 1.0*10**6                 # Circulation parameter C (Sv m3 kg-1 = m6 kg-1 s-1) in [0.1;9]*1.e6
gT_star_pico = 2.0*10**-5          # Effective turbulent temperature exchange velocity PICO in m s-1
alpha_coeff_pico  = 7.5*10**-5    # Thermal expansion coeff PICO in EOS °C**-1
beta_coeff_pico    = 7.7*10**-4    # Salinity contraction coeff PICO in EOS PSU**-1
rho_star_pico = 1033.         # EOS ref Density PICO kg m-3

###### Plume constants

# from Lazeroms et al. 2019
C_eps_lazero = 0.6 # Slope correction parameter
E0_lazero = 3.6*10**-2 # Entrainment coefficient
gamma_eff_T_lazero = 5.9*10**-4 # Effective thermal Stanton number / Heat exchange parameter (6.0 * 10**-4 in Pelle et al 2019)
alpha_coeff_lazero = 3.87*10**-5 # degC-1 Thermal expansion coefficient
beta_coeff_lazero = 7.86*10**-4 # psu-1 Haline contraction coefficient
C_d_lazero = 2.5*10**-3 # Drag coefficient

####### PICOP constants
C_d_gamma_T_S0 = 6.0 * 10**-4 # Heat exchange parameter
#C_d_gamma_T = 1.1 * 10**-3 # Turbulent heat exchange coefficient
C_d_gamma_T = 7 * 10**-5 # Turbulent heat exchange coefficient
g_1 = 0.545 # Heat exchange parameter
g_2 = 3.5 * 10**-5 # Heat exchange parameter m-1
M0 = 10./yearinsec # Melt rate parameter ms−1◦C−2
x0 = 0.56 # Dimensionless scaling factor



