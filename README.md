# MIT-16.940 Final Project: Uncertainty-Aware Physics-Informed Neural Networks
This repository implements a uncertainty-aware physics-informed neural network (UA-PINNs) to learn parametrizations of ocean equations. Specifically, this repository is a pytorch-based reimplementation of the approach in [Zhang et al., 2019](https://doi.org/10.1016/j.jcp.2019.07.048 "Quantifying total uncertainty in physics-informed neural networks"), which combines Polynomial Chaos Expansion (PCE) and physics-informed neural networks (PINNs). The implemented use-case is heavily based on [Lütjens et al., 2020](http://lutjens.scripts.mit.edu/me/docs/lutjens_ua_pinns_oceans_compsust_20.pdf "Uncertainty-Aware Physics-Informed Neural Networks"), which aims to learn parametrizations of ocean models. In this repository a very simple equation is used to demonstrate UA-PINNs, specifically a 1D stochastic elliptic equation. The motivation, approach, and results are detailed in the [write-up](https://github.com/blutjens/pce-pinns/blob/main/doc/main.pdf "Uncertainty Quantification in Physics-Informed Neural Networks").

# Installation
## Environment 
```
conda env create -f environment.yml # tested on Ubuntu 18.04
```

## Run
```
conda activate pce-pinns
# Train NN to learn polynomial coefficients of deg. 3 
python main.py --rand_flux_bc --pce --nn_pce --pce_dim 3 --n_samples 1000 --path_load_simdata pce_1k.pickle --est_param_nn k
python main.py --rand_flux_bc --pce --nn_pce --pce_dim 3 --n_samples 8 --est_param_nn k_true        
```

# References 
``
@article{Lutjens_2020,
	title = {Uncertainty-Aware Physics-Informed Neural Networks for Parametrizations in Ocean Modeling},
	authors = {Bj{\"o}rn L{\"u}tjens and Mark Veillette and Dava Newman},
	journal = {AGU Fall Meeting, Session on AI in Weather and Climate Modeling},
	year = 2020,
	url = {http://lutjens.scripts.mit.edu/me/docs/lutjens_ua_pinns_oceans_compsust_20.pdf}
}
@article{Zhang_2019,
  title={Quantifying total uncertainty in physics-informed neural networks for solving forward and inverse stochastic problems},
  author={D. Zhang and L. Lu and L. Guo and G. Karniadakis},
  journal={J. Comput. Phys.},
  year={2019},
  volume={397}
}
``