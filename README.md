# LTEsparse
Using Julia to optimize LTE fitting

[![Build Status](https://github.com/pukpr/LTEsparse/actions/workflows/ci.yml/badge.svg)](https://github.com/pukpr/LTEsparse/actions)

## Description

**LTEsparse** is a Julia tool for fitting Laplace Tidal Equations (LTE) to geophysical time series data. It models a target signal—such as the ENSO Niño 3.4 index—as a sparse superposition of tidal constituents (e.g., Mf, Draconic lunisolar) passed through a nonlinear folding function.

The discovery engine (`ls.jl`) works in two stages:
1. **Backpropagation** via [Zygote.jl](https://github.com/FluxML/Zygote.jl) to differentiably optimize tidal amplitudes, phases, wavenumber gains, and modulation parameters.
2. **SINDy-style sparsity pruning** that zeroes out low-power tidal components to find the minimal set of constituents needed to explain the data.

Training uses an 80/20 train/cross-validation split with early stopping, and the best model is saved as a versioned JSON config alongside a time-series plot.

## Installation

Julia 1.6 or later is required. Clone the repository and install the dependencies via the Julia package manager:

```bash
git clone https://github.com/pukpr/LTEsparse.git
cd LTEsparse
```

```julia
using Pkg
Pkg.add(["Zygote", "LinearAlgebra", "Statistics", "JSON3", "Dates", "Plots", "CSV", "DataFrames"])
```

## Usage

Run the discovery engine from the command line, providing a JSON config file and a data file:

```bash
julia main.jl simple.json mf.dat
```

Both arguments are optional and default to `simple.json` and `mf.dat` respectively.

**Data format:** A two-column file (space- or comma-separated) with columns `year_fraction` and `nino34_normalized`, with or without a header row.

**Config format (`simple.json`):** Defines the tidal constituents (name, frequency alias, initial amplitude and phase), nonlinear folding parameters (`beta_weights`, `wavenumber_gains`, `mod_phases`), and an optional `bias` offset.

After training, the script writes:
- An updated (versioned) JSON config with the optimized parameters and validation MSE.
- A PNG plot comparing the target data against the best prediction, with the cross-validation interval shaded.

## Project Structure

```
LTEsparse/
├── main.jl       # Entry point: loads data, invokes the discovery engine
├── ls.jl         # Core discovery engine (forward model, training loop, save logic)
├── simple.json   # Example tidal constituent config (Mf + Draconic lunisolar)
└── mf.dat        # Example ENSO Niño 3.4 time series (year fraction vs normalised index)
```

## Contributing

Pull requests and suggestions are welcome!  
To contribute:

1. Fork this repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request with a description of what you changed.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

The LTE framework is inspired by research into tidal forcing of ENSO and other low-frequency climate modes. The sparsity approach draws on ideas from SINDy (Sparse Identification of Nonlinear Dynamics).
