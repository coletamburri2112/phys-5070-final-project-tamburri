# phys-5070-final-project-tamburri

# Description of existing code, transferred to github and commented:

(1) In fl_funcs:
Major working functions for analysis of SDO/EVE 304 Angstrom light curves,
SDO/AIA 1600 Angstrom images and ribbon masks, and SDO/HMI magnetograms.
Includes algorithm for determination of separation and elongation of both
ribbons relative to the polarity inversion line. Flares able to beanalyzed
are contained in the RibbonDB database (Kazachenko et al. 2017).  Prior to
running this script, the user should obtain flare light curves and times
corresponding to the modeled flares in this database, for which the
impulsiveness index has been determined previously.

The polarity inversion line is determined by convolving the HMI masks
associated for each flare with a Gaussian of predetermined length, then
finding the major region of intersection between these and using the
resulting heatmap to fit a fourth-order polynomial.  Details of separation
and elongation methods relative to the PIL are included below.

Reconnection rates and ribbon areas for both positive and negative ribbons
are determined, and the values corresponding to the rise phase of the flare
(with some flare-specific variation) are fit to an exponential model, in
order to prepare for modeling efforts of flare heating particularly in the
rise phase.

Separation and elongation values (perpendicular and parallel PIL-relative
motion, respectively) are used to find separation and elongation rates,
through which significant periods of these two phases of ribbon motion can
be identified.

Plotting and data presentation routines are also below, which includes an
animation showing the timing of separation, elongation, and chromospheric
line light curves.

(2) In high1_process.py:
Processing of fl_funcs.py functions, up to the ribbon area plots

EXPANSION AS PART OF COMPUTATIONAL PHYSICS PROJECT

(3) Creation of fermi HXR processing script ("fermi_processing.py")
 - Fourier transform of existing HXR light curve
 - Commenting
 - Cleanup 
 
 (4) Creation of new algorithm to identify shear in flare ribbons (functions included at the end of "fl_funcs.py")
 - Algorithm to determine shear of flare ribbons, a proxy for the PIL-perpendicular component of the magnetic field
 - Commenting and cleanup
 
 (5) Testing: least-squares fitting for ribbon area, etc 
 
 (6) pytest testing routines (testing_routines.py)
 
 (7) Integrated error tests