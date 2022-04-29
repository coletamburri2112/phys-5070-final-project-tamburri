
Cole Tamburri
ASTR 5070 Final Project
University of Colorado Boulder
Prof. Ethan Neil
Final Version: 29 April 2022

 In addition to the below, there are several miscellaneous plots produced by the flare_example_process.py script.
 Feel free to peruse these for some more details, but the most important and relevant of these
 (in addition to some extras) are included in the presentation Jupyter Notebook, which uses the same code
 and explains some of the background in a way friendly to someone not directly in the field.
 
 For details about algorithms, the source code in fl_funcs_proj.py is useful.  For large-scale relevant
 physical and computational discussion, the comments in the Jupyter Notebook should be sufficient.
 
 Some pre-existing code is necessary; new steps taken are identified both in the source code and
 the presentation Jupyter Notebook.
 
A Continuous Integration setup with GitHub is unexplored - this is the only part of the project proposal
not carried out to completion.  The error checking and integration tests developed in the Jupyter Notebook
are comprehensive for presentation and error checking purposes, and should be sufficient for the purposes
here.
 
(1) In fl_funcs_proj.py:

Major working functions for analysis of SDO/EVE 304 Angstrom light curves,
SDO/AIA 1600 Angstrom images and ribbon masks, and SDO/HMI magnetograms.
Includes algorithm for determination of separation and elongation of both
ribbons relative to the polarity inversion line. Flares able to be analyzed
are contained in the RibbonDB database (Kazachenko et al. 2017), though only one is
chosen here (13 October 2013 00:12 UT).

The polarity inversion line is determined by convolving the HMI masks
associated for each flare with a Gaussian of predetermined length, then
finding the major region of intersection between these and using the
resulting heatmap to fit a fourth-order polynomial.  Computational details of separation
and elongation methods relative to the PIL can be seen in the code.

Reconnection rates and ribbon areas for both positive and negative ribbons
are determined, and the values corresponding to the rise phase of the flare
(with some flare-specific variation) are fit to an exponential model, in
order to prepare for modeling efforts of flare heating particularly in the
rise phase.

Separation and elongation values (perpendicular and parallel PIL-relative
motion, respectively) are used to find separation and elongation rates,
through which significant periods of these two phases of ribbon motion can
be identified.

Code for identification of magnetic shear is included, using the footpoints of magnetic loops
as a guide for the orientation of the magnetic field overlying the PIL.

Least-squares fitting and error determination relative to an exponential model are included
for the ribbon area functions.

(2) In flare_example_process.py:

Processing of fl_funcs_proj.py functions.  Reproduced in the presentation Jupyter Notebook for better
commenting and viewing.

(3) In fermi_processing_proj.py:

Creation of fermi HXR processing script - processes data 
downloaded from the OSPEX gui for the flare, which will give insight into the timing of 
HXR emission and therefore particle acceleration for the flare.
 
 (4) In fin_proj_testing.ipynb
 
 Unit testing, presentation and explanation of results, as well as integrated testing. 
  This notebook does some pre-processing of flare data, and then delves deeply into the shear 
  processing, ribbon area model fitting, and other new components of the project.  Science 
  background and comments as the process is carried out are included for clarity.
