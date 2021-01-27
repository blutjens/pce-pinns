

# MIT-16.940 Final Project: Uncertainty-Aware Physics-Informed Neural Networks
This repository implements a physics-informed neural network (PINN) to learn the diffusion parameter in a 1D stochastic elliptic equation. The use-case is heavily based on Lütjens et al., 2020 and methodology is heavily based on Zhang et al., 2019. 

# Installation
## Environment
- tested on Ubuntu 18.04
```
conda create --name pce-pinns python==3.8.5 pytorch==1.7.1 cudatoolkit==11.0 pandas==1.2.1 scikit-learn==0.23.2 matplotlib==3.3.2 tqdm 4.55.1
```

## Run
```
conda activate eie-lu-seg
# Train NN to learn polynomial coefficients of deg. 3 
python main.py --rand_flux_bc --pce --nn_pce --poly_deg 3 --n_samples 1000 --path_load_simdata pce_1k.pickle --est_param_nn k
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