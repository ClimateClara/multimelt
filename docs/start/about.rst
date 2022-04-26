About
=====

The motivation behind multimelt
-------------------------------

Ocean-induced melt at the base of Antarctic ice shelves (also called basal melt) is a crucial component for simulations of the Antarctic contribution to future sea-level evolution. Ice shelves have been thinning all around Antarctica in past decades. Thinning reduces the ice shelves' buttressing potential, which means that the restraining force that they exert on the ice outflow at the grounding line is lower and more ice is discharged into the ocean. In some bedrock configurations, increased melt can trigger instabilities. This is why it is currently the main uncertainty source in such projections.

To represent basal melt in numerical simulations, especially in ice-sheet models, parameterisations have been developed in past decades. They link a profile of temperature and salinity in front of an ice shelf to the melt rates below the ice shelf. The multimelt package provides code for the most commonly used of these parameterisations for rapid use and possible assessment or evaluation. 

It takes input temperature and salinity profiles and returns several 2D and 1D metrics about the basal melt. It is designed to be applicable on circum-Antarctic scale.


Authors
-------

| The multimelt code is based on physical formulations by :cite:`beckmann03`, :cite:`holland08`, :cite:`deconto16`,  :cite:`reese18`, :cite:`lazeroms18`, :cite:`lazeroms19`, :cite:`favier19`, :cite:`jourdain20`} and Fortran code by N. Jourdain.
| This Fortran code has been translated and adapted to Python and further multimelt code has been developed by Clara Burgard - `ClimateClara <http://www.github.com/ClimateClara>`_.

License
-------

This project is licensed under the GPL3 License - see the
`license <https://www.gnu.org/licenses/gpl-3.0.en.html>`_ for details.
