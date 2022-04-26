.. _installation:

How to install
==============

you can install it via pip

.. code-block:: bash

  pip install arc3o

or via conda


.. code-block:: bash

  conda install -c conda-forge arc3o

This package is programmed in python 3.6 and should be working with all `python
versions > 3.6`. Additional requirements are numpy, xarray, pandas, tqdm and pathos.

We recommend to install the dependencies via 

.. code-block:: bash
  
  conda install -c conda-forge pandas tqdm pathos 

as they might not work well using ``pip``.

If you want to work on the code, please fork this github repository: https://github.com/ClimateClara/arc3o/
.. _installation:

How to install
==============

Currently, the easiest is to clone the `multimelt git hub repository <github.com/ClimateClara/multimelt>`_,
go to the repository folder and type:

.. code-block:: bash

  pip install -e . 
  
If you want to modify it locally, you can fork the `multimelt git hub repository <github.com/ClimateClara/multimelt>`_, go to the repository folder and type:
  
.. code-block:: bash

  pip install -e .
  
This package is programmed in python 3.8 and should be working with all `python
versions > 3.8`. Additional requirements are numpy, xarray, pandas, tqdm and dask. (to be refined)

