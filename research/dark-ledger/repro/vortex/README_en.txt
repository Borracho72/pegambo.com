PHASE VORTICES IN THE BERRY RANDOM-WAVE MODEL
This package independently reproduces known statistics of an isotropic 2D
monochromatic complex Gaussian wave. Comparisons with the first edition
of the note are in the correction table in the note itself.

RUN
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python reproduce.py --smoke
  python reproduce.py

On Windows PowerShell, use .\.venv\Scripts\Activate.ps1 to activate the
environment. The smoke run exercises the full pipeline on small windows.
Its output in smoke_output/ is not a statistical reproduction. The full
run saves new results, logs and regenerated point sets in reproduced/.
Use --output-dir DIR to change the destination, or --reuse-fields to
reanalyze fields already generated there. The shipped results and logs
are not overwritten. Raw point sets are not included in this package.

METHOD
The reference run uses 64 fields, 640 planar modes and 463,113 detected
zeros in analysis windows of side 48 wavelengths, surrounded by a guard
zone of 3 wavelengths. Plaquette phase winding identifies vortices.
The grid steps are 0.10, 0.05 and 0.025 wavelengths; 32 of the same fields
are checked at 0.0125. Newton refinement uses the original wave sum.
The overlay experiment pools zero coordinates, not complex amplitudes.
Uncertainties below are one bootstrap standard error, over realizations
or, for the overlay analysis, eight independent blocks of realizations.

REFERENCE RESULTS
  Same-sign asymptote: g_same(r) ~ 1.80451063 (r/lambda)^3 as r -> 0.
  At r = 0.1 lambda: g_same = 0.00192375 (theory).
  For 0.09 <= r/lambda < 0.11: 0.00241 +/- 0.00079 (verified roots).
  Opposite-sign nearest neighbour: 78.712%, SE 0.071 percentage points
  (one directed link per zero).
  Nearest-neighbour mode, irrespective of sign: 0.4574 +/- 0.0025 lambda.
  Mode for links whose actual nearest neighbour has opposite sign:
  0.4189 +/- 0.0042 lambda.
  Overlays (m = 1, 2, 4, 8), KS distance to the matching Poisson
  nearest-neighbour CDF: 0.3402 +/- 0.0006, 0.1105 +/- 0.0011,
  0.0490 +/- 0.0012, 0.0237 +/- 0.0006. At fixed r/lambda,
  the population pair-correlation excess g_m(r)-1 is (g_1(r)-1)/m.

LIMITS
The simulation is an idealized null model, not apparatus data. Small
same-sign distances have few counts and are sensitive to grid spacing,
binning and zero verification. Modes depend on sign selection and
smoothing. Finite overlays are not exactly Poisson, and measured
distances to Poisson depend on the metric and normalization.

REFERENCES
Berry & Dennis (2000), pair correlations, density and ring spectrum:
https://doi.org/10.1098/rspa.2000.0602
Shvartsman & Freund (1994), nearest-neighbour sign anticorrelation:
https://doi.org/10.1103/PhysRevLett.72.1008
Saichev, Berggren & Sadreev (2001), nearest-neighbour distances:
https://arxiv.org/abs/nlin/0012019
Dennis (2001), partial pair correlations and their short-distance limit:
https://eprints.soton.ac.uk/29376/1/SPIE4403_13.pdf
