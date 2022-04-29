# phys-5070-final-project-tamburri

# Description of existing code, transferred to github and commented:

(1) In fl_funcs_proj.py:
Major working functions for analysis of SDO/EVE 304 Angstrom light curves,
SDO/AIA 1600 Angstrom images and ribbon masks, and SDO/HMI magnetograms.
Includes algorithm for determination of separation and elongation of both
ribbons relative to the polarity inversion line. Flares able to be analyzed
are contained in the RibbonDB database (Kazachenko et al. 2017).  Prior to
running this script, the user should obtain flare light curves and times
corresponding to the modeled flares in this database, for which the
impulsiveness index has been determined previously.

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

(3) Creation of fermi HXR processing script ("fermi_processing_proj.py") - processes data 
downloaded from the OSPEX gui for the flare, which will give insight into the timing of 
HXR emission and therefore particle acceleration for the flare.
 
 (4) Testing: fin_proj_testing.ipynb includes unit testing, presentation and explanation of results, 
 as well as integrated testing.  This notebook does some pre-processing of flare data, and then
 delves deeply into the shear processing, ribbon area model fitting, and other new components
 of the project.  Science background and comments as the process is carried out are included 
 for clarity.