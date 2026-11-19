# Setuav fork of NeuralFoil

This fork makes NeuralFoil run on NumPy alone, for use in
[PyThrust](https://github.com/Setuav/PyThrust) without AeroSandbox's dependencies.

Branch `feat/numpy-core`:

- AeroSandbox is optional (`pip install neuralfoil[asb]`); results are bit-identical with or without it,
  and CasADi inputs still work when it is installed.
- `neuralfoil.geometry`: reading .dat files (Selig and Lednicer), NACA 4-digit and 5-digit airfoils,
  normalization and Kulfan (CST) fitting, in NumPy.
- New entry point `get_aero_from_naca()`.

## Upstream

Not yet proposed to [peterdsharpe/NeuralFoil](https://github.com/peterdsharpe/NeuralFoil). The upstream
README's FAQ names a small stand-alone module with simple dependencies as a goal, so this change may be
welcome there. To do: open an issue first, then a pull request without this file.

Keep `master` equal to upstream; sync with `git fetch upstream && git merge upstream/master`.
