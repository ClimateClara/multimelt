"""
Created on 22.08.2021
THIS DOCUMENT IS WORK IN PROGRESS
Based on: https://github.com/pypa/sampleproject
@author: Clara Burgard, clara.burgard@gmail.com
    Copyright (C) {2020}  {Clara Burgard}
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import setuptools
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = open(os.path.join(here, 'README.rst'), encoding='utf-8').read()

setuptools.setup(
	
    #The project's name
    name='multimelt',
    
    #The project's version 
    version='0.2',
    
    #The project's metadata
    author='Clara Burgard',
    author_email='clara.burgard@univ-grenoble-alpes.fr',
    description='Regroupment of the main existing ice shelf basal melt parameterisations',
    long_description=long_description,
    
    #The project's main homepage.
    url='https://github.com/ClimateClara/multimelt',
    
    #The project's license
    license='GPL-3.0',
    
    packages=setuptools.find_packages(exclude=['docs', 'tests*', 'examples']),
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
    ],
      
    project_urls={
        'Source': 'https://github.com/ClimateClara/multimelt',
        'Tracker': 'https://github.com/ClimateClara/multimelt/issues',
    #    'Documentation': 'https://multimelt.readthedocs.io',
      },
    
    keywords='earth-sciences climate-modeling ice-sheet antarctica oceanography',
    
    python_requires='>=3.5',
	
    install_requires=[
          'numpy',
          'xarray',
          'pandas',
          'tqdm',
          'scipy',
          'cc3d',
          'gsw',
          'pyproj',
          'dask',
      ],

)
