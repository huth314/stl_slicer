=======================================
Simple STL Slicer to be used with FEBID
=======================================

:Author: Michael Huth
:Maintainer: Michael Huth
:Contact: michael.huth@physik.uni-frankfurt.de

What is FEBID?
--------------
FEBID stands for focused electron beam induced deposition, a direct-write nanofabrication
technology that uses the targeted dissociation of chemical precursors adsorbed on the
surface of a substrate to generate a localized permanent deposit. The deposit's material
composition and shape will define its functionality.

Why does FEBID need a slicer for STL files?
-------------------------------------------
FEBID is uniquely powerful in generating 3D nanostructures very much like 3D printing on
the nano- to micrometer scale. With FEBID various materials can be printed, e.g.,
ferromagnetic or superconducting. In order to conveniently defining 3D models, the established
workflow in standard 3D printing is to use STL files exported from a CAD program. These
files than have to be sliced into layers perpendicular to the growth direction (typically,
z-direction). In an STL file the 3D structure's surface is represented as a set of oriented
triangles.

Installation
------------
The STL Slicer requires Python 3.x and the Python packages numpy, scipy and pandas.

Usage
-----
.. code-block:: python
   :caption: Simple example for usage
   :lineos:
   import stl_slice as stl
    stl_object = stl.STLObject(stl_file='test.stl', scale=1.0) # e.g.
    # set various parameters
    pitch = 1.0 # distance between raster points for any given slice (in nm)
    dzp = 0.05 # height increase in growth direction for basis dwell time (1 ms) (in nm)
    dz_min = 1.0 # mininum distance between neighboring slices (in nm)
    dz_max = 3.0 # maximum distance between neighboring slices (in nm)
    sigma = 5.0 # FWHM of electron beam diameter for Gaussian beam (in nm)
    HFW = 8500.0 # Horizontal field width of SEM image (in nm); Thermo Scientific (FEI) specific
    stl_object.generate_streamfile('test.str', HFW, pitch, dzp, dz_min, dz_max, sigma) # generates streamfile for ThermoFisher (FEI) SEMs with 16-bit pattern generator

Look into the class file for more details. The code is well documented. Use Doxygen for documentation generation.
