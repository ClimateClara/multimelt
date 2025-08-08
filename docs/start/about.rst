About
=====

The motivation behind multimelt
-------------------------------

Ocean-induced melt at the base of Antarctic ice shelves (also called basal melt) is a crucial component for simulations of the Antarctic contribution to future sea-level evolution. Ice shelves have been thinning all around Antarctica in past decades. Thinning reduces the ice shelves' buttressing potential, which means that the restraining force that they exert on the ice outflow at the grounding line is lower and more ice is discharged into the ocean. In some bedrock configurations, increased melt can trigger instabilities. This is why it is currently the main uncertainty source in such projections.

To represent basal melt in numerical simulations, especially in ice-sheet models, parameterisations have been developed in past decades. They link a profile of temperature and salinity in front of an ice shelf to the melt rates below the ice shelf. The multimelt package provides code for the most commonly used of these parameterisations for rapid use and possible assessment or evaluation. 

It takes input temperature and salinity profiles and returns several 2D and 1D metrics about the basal melt. It is designed to be applicable on circum-Antarctic scale.


Authors
-------

| The multimelt code is based on existing physical formulations:

* the linear fomulation as described by :cite:`beckmann03`
* the quadratic formulation as described by :cite:`holland08`, :cite:`deconto16`, :cite:`favier19` and :cite:`jourdain20`
* the plume parameterisation as described by :cite:`lazeroms18` and :cite:`lazeroms19`
* the box parameterisation (or PICO) as described by :cite:`reese18`
* the PICOP parameterisation as described py :cite:`pelle19`
* the DeepMelt model (neural network) as described by :cite:`burgard23`
    
| An existing Fortran code by N. Jourdain has been translated and adapted to Python and further multimelt code has been developed by Clara Burgard - `ClimateClara <http://www.github.com/ClimateClara>`_.

How to cite
-----------
The detailed description of the application of the functions in multimelt is found in :cite:`burgard22` and should therefore, when used, be cited as follows:

Burgard, C., Jourdain, N. C., Reese, R., Jenkins, A., and Mathiot, P. (2022): An assessment of basal melt parameterisations for Antarctic ice shelves, The Cryosphere, https://doi.org/10.5194/tc-16-4931-2022.

For DeepMelt, we point to:

Burgard, C., Jourdain, N. C., Mathiot, P., Smith, R. S., Schäfer, R., Caillet, J., et al. (2023): Emulating present and future simulations of melt rates at the base of Antarctic ice shelves with neural networks. Journal of Advances in Modeling Earth Systems, 15, e2023MS003829. https://doi.org/10.1029/2023MS003829

License
-------

This project is licensed under the GPL3 License - see the
`license <https://www.gnu.org/licenses/gpl-3.0.en.html>`_ for details.
