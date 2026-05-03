# Ramjet Sizing Tool

Quasi-one-dimensional sizing tool used to estimate the stagnation conditions and air mass flow rates that must be replicated by a direct-connect ramjet test facility for a given combustor size, flight Mach number and altitude.

The physics and notation are based primarily on Gary W. Johnson's *A Practical Guide to Ramjet Propulsion*, particularly the quasi-one-dimensional ramjet sizing, inlet performance and direct-connect testing methods discussed in the text.

## Purpose

This code was developed as part of a preliminary design study for a pebble-bed heated direct-connect ramjet test facility. It is intended to support early-stage engineering estimates, including:

- required inlet stagnation temperature and pressure,
- air mass flow rate for selected Mach-altitude conditions,
- approximate direct-connect facility operating requirements,
- parameter sweeps used to explore feasible test envelopes,
- generation of selected plots used in the accompanying report.

## Repository Contents

- `gasdyn.py` - compressible-flow and gas-dynamic utility functions.
- `engine_types.py` - shared data containers used across the solver.
- `engine_core.py` - component-level flow and thrust calculation functions
- `engine_sizing.py` - main ramjet sizing functions.
- `run_engine_sizing.py` - example script for running a single sizing case.
- `requirements.txt` - Python package requirements.


## Requirements

The code was developed in Python 3. Required packages are listed in `requirements.txt`.

Typical dependencies include:

```text
numpy
scipy
matplotlib
```

## Usage

After installing the required packages, example sizing cases can be run using:

```bash
python run_engine_sizing.py
```

Some file paths, plot output locations or input values may need to be adjusted depending on the local machine and folder structure.

## Limitations

This code is a preliminary engineering sizing tool, not a validated design package. The calculations rely on quasi-one-dimensional assumptions, empirical inlet data and simplified models. Results should be interpreted alongside the assumptions and limitations discussed in the accompanying report.

The tool is intended to estimate the operating conditions required from a ground-test facility, rather than to provide a complete ramjet performance prediction or final mechanical design.

## Author

Developed by Max Crawford-Collins as part of an undergraduate aerospace engineering project on ramjet direct-connect testing and pebble-bed heated air supply.
