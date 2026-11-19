# Setuav fork of NeuralFoil

This fork makes NeuralFoil run on NumPy alone, for use in
[PyThrust](https://github.com/Setuav/PyThrust) without AeroSandbox's dependencies. It is published on
PyPI as `setuav-neuralfoil` and imported as `setuav_neuralfoil`, so it never overwrites an upstream
`neuralfoil` installed alongside it (AeroSandbox depends on one).

- AeroSandbox is optional (`pip install setuav-neuralfoil[asb]`); results are bit-identical with or
  without it, and CasADi inputs still work when it is installed.
- `neuralfoil.geometry`: reading .dat files (Selig and Lednicer), NACA 4-digit and 5-digit airfoils,
  normalization and Kulfan (CST) fitting, in NumPy.
- New entry point `get_aero_from_naca()`.

## Branches

- `master`: equal to upstream; sync with `git fetch upstream && git merge upstream/master`.
- `feat/numpy-core`: the changes above, kept clean for a pull request to upstream.
- `setuav` (default): `feat/numpy-core` plus packaging as `setuav-neuralfoil` and this file.

## Releases

Version is upstream's plus a fourth number for this fork: `0.3.3.1` is NeuralFoil 0.3.3's models with
this fork's first release. Publishing a GitHub release from `setuav` runs `.github/workflows/publish.yml`,
which tests, builds, checks the wheel next to upstream NeuralFoil, and uploads to PyPI with Trusted
Publishing.

For development, uv does not install the project (the wheel's rename is not possible in an editable
install); `uv run --extra test pytest tests` imports `neuralfoil` from the source tree.

## Upstream

Not yet proposed to [peterdsharpe/NeuralFoil](https://github.com/peterdsharpe/NeuralFoil). The upstream
README's FAQ names a small stand-alone module with simple dependencies as a goal, so `feat/numpy-core`
may be welcome there. To do: open an issue first, then a pull request without `FORK.md`.
