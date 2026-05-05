def compute_params_CG(scale_factor):
    MASS_Ni = 58.69
    EPS_Ni = 2.5
    SIGMA_Ni = 2.28

    sigma = SIGMA_Ni * scale_factor
    mass = MASS_Ni * (scale_factor**3)
    epsilon = EPS_Ni * (scale_factor**3)
    lattice = 3.52 * scale_factor
    return sigma, lattice, epsilon, mass