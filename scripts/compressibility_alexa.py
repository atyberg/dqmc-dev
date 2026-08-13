#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compressibility_alexa.py 
atyberg@stanford.edu

Computes the compressibility vs. mu for square and honeycomb geometries.

Usage example:
    ADD IN OWN
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import util
import argparse

Nx, Ny, Norb, U, beta, nsweep = 0, 0, 0, 0, 0, 0
geometry = ""
t = 1

# TODO: add in thing to use only completed bins from jqjq.py file get_component
def load_densities(base_dir):
    """
    Loads the data for each mu value in the directory and computes the density <n> at each temperature, saving it in a dict.
    Each entry in the dict for a specific mu value is of shape (Nbins, Norb) if trans sym is on (has not been averaged yet).
    """
    global Nx, Ny, Norb, U, beta, geometry, nsweep
    densities_dict = {} # dictionary to store the density arrays at each mu. each entry is of shape (Nbins,).

    for folder in os.listdir(base_dir): # each folder represents a mu value
        if not folder.startswith("mu_"):
            continue
        mu = float(folder[3:])
        # print(f"mu = {mu}") # for testing
        folder_path = os.path.join(base_dir, folder) + "/"

        Nx, Ny, Norb, U, beta, geometry, trans_sym, nsweep = util.load_firstfile(folder_path, 
                                                            "metadata/Nx", 
                                                            "metadata/Ny", 
                                                            "metadata/Norb",
                                                            "metadata/U",
                                                            "metadata/beta",
                                                            "metadata/geometry", 
                                                            "metadata/trans_sym",
                                                            "params/n_sweep_meas")
        
        ns, density, sign = util.load(folder_path, "meas_eqlt/n_sample", "meas_eqlt/density", "meas_eqlt/sign")
                
        # only use completed bins
        mask = ns == ns.max()
        density = density[mask]
        sign = sign[mask]

        geometry = geometry[()].decode('utf-8') # now it's a string, not an array with a bytes object

        # divide by sign
        # everything is of shape (Nbins, -1)
        sign = np.reshape(sign, (-1,1))
        density = density / sign # don't divide by 2 here because we want to set half filling at <n>=1

        # find the per site density, geometry-specific
        if geometry == "square":
            if trans_sym:
                densities_dict[mu] = density[:,0]
            else:
                raise NotImplementedError
            
        elif geometry == "honeycomb":
            if trans_sym:
                # 2 orbitals per cell, density array has shape (Nbins, 2)
                densities_dict[mu] = 1/Norb * (density[:,0] + density[:,1])
            else:
                raise NotImplementedError
                 
    return densities_dict

'''
def compute_energy(g00_u, g00_d, double_occ, density_u, density_d, sign, geometry, trans_sym):
    """
    Computes the energy of the system at a specific temperature given g00_u, g00_d, double occupancy, and density. Geometry specific.
    g00 arrays are of shape (Nbins, num_ij).

    Returns
    -------
        energy : ndarray
            Energy per unit cell. Array of shape (Nbins,).

    """
    # t = 1
    # Nx, Ny, Norb, mu, U, geometry, trans_sym = util.load_firstfile(path, 
    #                                                             "metadata/Nx", 
    #                                                             "metadata/Ny", 
    #                                                             "metadata/Norb",
    #                                                             "metadata/mu",
    #                                                             "metadata/U",
    #                                                             "metadata/geometry", 
    #                                                             "metadata/trans_sym")
    # g00_u, g00_d, double_occ, density_u, density_d, sign = util.load(path,
    #                                                         "meas_eqlt/g00_u",
    #                                                         "meas_eqlt/g00_d",
    #                                                         "meas_eqlt/double_occ",
    #                                                         "meas_eqlt/density_u",
    #                                                         "meas_eqlt/density_d",
    #                                                         "meas_eqlt/sign")
    
    # Divide by sign
    # everything is of shape (Nbins, -1)
    sign = np.reshape(sign, (-1,1))
    g00_u = g00_u / sign
    g00_d = g00_d / sign
    double_occ = double_occ / sign
    density_u = density_u / sign
    density_d = density_d / sign

    N = Nx * Ny * Norb
    # geometry = geometry[()].decode('utf-8') # now it's a string, not an array with a bytes object

    if geometry == "square":
        if trans_sym:
            # kinetic term
            E_hop_up = t * N * (g00_u[:,1] + g00_u[:,Nx-1] + g00_u[:,Nx] + g00_u[:,(Ny-1)*Nx])
            E_hop_down = t * N * (g00_d[:,1] + g00_d[:,Nx-1] + g00_d[:,Nx] + g00_d[:,(Ny-1)*Nx])

            # chemical term
            E_chem_up = -mu * N * density_u[:,0]
            E_chem_down = -mu * N * density_d[:,0]

            # potential term
            E_int = U * N * double_occ[:,0]

            # print(f"g00_u value = {g00_u[0,0]}") # for testing

            energy = (E_hop_up + E_hop_down + E_chem_up + E_chem_down + E_int) / Nx/Ny # per unit cell (also per site for square)

        else:
            raise NotImplementedError("square trans_sym=0 case not implemented.")

    elif geometry == "honeycomb":
        if trans_sym:
            # # kinetic term (for whole lattice, not per site)
            # # indexing of g00 is displacement vector k = kx + Nx*ky + Nx*Ny*jo + N*io with jo and io orbital indices
            # E_hop_up = t * Nx * Ny * (g00_u[:,N] +
            #                           g00_u[:,Nx-1+N] + 
            #                           g00_u[:,Nx*(Ny-1)+N] +
            #                           g00_u[:,Nx*Ny] +
            #                           g00_u[:,1+Nx*Ny] +
            #                           g00_u[:,Nx+Nx*Ny])
            # E_hop_down = t * Nx * Ny * (g00_d[:,N] + 
            #                           g00_d[:,Nx-1+N] + 
            #                           g00_d[:,Nx*(Ny-1)+N] +
            #                           g00_d[:,Nx*Ny] +
            #                           g00_d[:,1+Nx*Ny] +
            #                           g00_d[:,Nx+Nx*Ny])
            
            # kinetic term treating the bonds from the i atom (something weird with fortran ordering?)
            E_hop_up = t * Nx * Ny * (g00_u[:,Nx*Ny] +
                                      g00_u[:,Nx-1+Nx*Ny] + 
                                      g00_u[:,Nx*(Ny-1)+Nx*Ny] +
                                      g00_u[:,N] +
                                      g00_u[:,1+N] +
                                      g00_u[:,Nx+N])
            E_hop_down = t * Nx * Ny * (g00_d[:,Nx*Ny] +
                                      g00_d[:,Nx-1+Nx*Ny] + 
                                      g00_d[:,Nx*(Ny-1)+Nx*Ny] +
                                      g00_d[:,N] +
                                      g00_d[:,1+N] +
                                      g00_d[:,Nx+N])
            
            # chemical term
            # density array is of shape (Nbins, Norb). it has a measurement for each site in the unit cell (A and B for honeycomb)
            # TODO: check if this is right. maybe use green's functions
            E_chem_up = -mu * Nx * Ny * (density_u[:,0] + density_u[:,1])
            E_chem_down = -mu * Nx * Ny * (density_d[:,0] + density_d[:,1])

            # potential term
            # TODO: check that this is the right representation of interaction term (just adding A and B sites together)
            # maybe change this to using green's functions
            E_int = U * Nx * Ny * (double_occ[:,0] + double_occ[:,1])

            # energy = (E_hop_up + E_hop_down + E_chem_up + E_chem_down + E_int) / N # per site
            energy = (E_hop_up + E_hop_down + E_chem_up + E_chem_down + E_int)/Nx/Ny # per unit cell
            # energy = (E_hop_up + E_hop_down + E_chem_up + E_chem_down + E_int)/2 # whole lattice/2


        else:
            raise NotImplementedError("honeycomb trans_sym=0 case not implemented.")

    else:
        raise NotImplementedError(f"{geometry} geometry not yet implemented.")
    
    return energy
'''

def plot_density_jackknife(base_dir, output_dir, TB_file=None):
    densities_dict = load_densities(base_dir)

    # make sure all the entries have the same number of bins
    min_size = min(n.shape[0] for n in densities_dict.values())
    densities_dict = {mu: n[:min_size] for mu, n in densities_dict.items()}

    mu_vals = []
    n_vals = []

    for mu in densities_dict:
        n = densities_dict[mu]
        mu_vals.append(mu)
        n_vals.append(n)

    # Sort by mu for clean plotting
    sorted_indices = np.argsort(mu_vals)
    mu_vals = np.array(mu_vals)[sorted_indices]
    n_vals = np.array(n_vals)[sorted_indices]

    # use jackknife resampling to get mean and error estimates
    n_means = []
    n_errs = []
    for ind, n in enumerate(n_vals):
        # print(f"mu {mu_vals[ind]}, n shape: {np.shape(n)}")
        mean, err = util.jackknife(np.ones_like(n), n)
        # print(f"mean={mean}, err={err}")
        n_means.append(mean)
        n_errs.append(err)

    n_means = np.array(n_means)
    n_errs = np.array(n_errs)

        # Create plot
    plt.figure(figsize=(8, 6))
    # plt.plot(T_vals, avg_spheat_vals, 'o', linewidth=2, markersize=6, label="finite difference")

    temp = 1.0 / beta
    if TB_file != None:
        # Read in the tightbinding data
        mu_vals_TB = np.loadtxt(TB_file, delimiter=",", skiprows=1, usecols=0)
        n_vals_TB = np.loadtxt(TB_file, delimiter=",", skiprows=1, usecols=1)
        mu_max = np.max(mu_vals_TB)
        mu_min = np.min(mu_vals_TB)
        mu_mask = (mu_vals <= mu_max) & (mu_vals >= mu_min)

        plt.errorbar(mu_vals[mu_mask], n_means[mu_mask], n_errs[mu_mask], fmt='o', linewidth=2, markersize=6, label="DQMC")
        plt.plot(mu_vals_TB, n_vals_TB, 'o', linewidth=2, markersize=6, label="TB")

        save_file = f"density_{geometry}_DQMC+TB_{Nx}x{Ny}_U{U}_T{temp:.2f}"

    else:
        plt.errorbar(mu_vals, n_means, n_errs, fmt='o', linewidth=2, markersize=6, label="DQMC")
        # plt.xscale('log')

        save_file = f"density_{geometry}_{Nx}x{Ny}_U{U}_T{temp:.2f}_{nsweep}sweeps"
        # np.savetxt(f"/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/{save_file}.csv", np.c_[mu_vals, n_means, n_errs], delimiter="\t", header="mu\t<n>\t<n> error")
        np.savetxt(os.path.join(output_dir, f"{save_file}.csv"), np.c_[mu_vals, n_means, n_errs], delimiter="\t", header="mu\t<n>\t<n> error")


    plt.xlabel('$\\mu$', fontsize=12)
    plt.ylabel('$\\langle n \\rangle$', fontsize=12)
    plt.title(f'Density for {Nx}x{Ny} {geometry} lattice, U={U}, T={temp:.2f}', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    # plt.savefig(f"/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/{save_file}.png", dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{save_file}.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_compressibility_jackknife(base_dir, output_dir, TB_file=None, save_data=False, npoint_stencil=3):
    densities_dict = load_densities(base_dir)

    # make sure all the entries have the same number of bins
    min_size = min(n.shape[0] for n in densities_dict.values())
    densities_dict = {mu: n[:min_size] for mu, n in densities_dict.items()}

    mu_vals = []
    n_vals = []

    for mu in densities_dict:
        n = densities_dict[mu]
        mu_vals.append(mu)
        n_vals.append(n)
        # print(n.shape)     

    # Sort by mu for clean plotting
    sorted_indices = np.argsort(mu_vals)
    mu_vals = np.array(mu_vals)[sorted_indices]
    n_vals = np.array(n_vals)[sorted_indices]

    # print(n_vals.shape)

    # calculate compressibility using finite difference method
    N = Nx * Ny * Norb
    kappa_vals = []
    if npoint_stencil == 3:
        for i in range(1, len(n_vals)-1):
            kappa = (n_vals[i+1,:] - n_vals[i-1,:]) / (mu_vals[i+1] - mu_vals[i-1])
            kappa_vals.append(kappa)
        mu_vals = mu_vals[1:-1]
    elif npoint_stencil == 5:
        # check that mu spacing is uniform
        diffs = np.diff(mu_vals)
        if not np.allclose(diffs, diffs[0]):
            raise ValueError("mu array spacing is not uniform")
        
        dmu = diffs[0]
        for i in range(2, len(n_vals)-2):
            kappa = (-n_vals[i+2,:] + 8*n_vals[i+1,:] - 8*n_vals[i-1,:] + n_vals[i-2,:]) / (12*dmu)
            kappa_vals.append(kappa)
        mu_vals = mu_vals[2:-2]
    else:
        raise ValueError("Only 3 and 5 point stencils allowed.")

    # use jackknife resampling to get mean and error estimates
    kappa_means = []
    kappa_errs = []
    for ind, kappa in enumerate(kappa_vals):
        mean, err = util.jackknife(np.ones_like(kappa), kappa)
        # print(f"mean={mean}, err={err}")
        kappa_means.append(mean)
        kappa_errs.append(err)

    kappa_means = np.array(kappa_means)
    kappa_errs = np.array(kappa_errs)

    # Create plot
    plt.figure(figsize=(8, 6))
    # plt.plot(T_vals, avg_spheat_vals, 'o', linewidth=2, markersize=6, label="finite difference")

    # TODO: modify plotting, it's currently density
    print(beta)
    temp = 1.0 / beta
    if TB_file != None:
        # Read in the tightbinding data
        mu_vals_TB = np.loadtxt(TB_file, delimiter=",", skiprows=1, usecols=0)
        kappa_vals_TB = np.loadtxt(TB_file, delimiter=",", skiprows=1, usecols=1)
        mu_max = np.max(mu_vals_TB)
        mu_min = np.min(mu_vals_TB)
        mu_mask = (mu_vals <= mu_max) & (mu_vals >= mu_min)

        plt.errorbar(mu_vals[mu_mask], kappa_means[mu_mask], kappa_errs[mu_mask], fmt='o', linewidth=2, markersize=6, label="DQMC")
        plt.plot(mu_vals_TB, kappa_vals_TB, 'o', linewidth=2, markersize=6, label="TB")

        if save_data:
            save_file = f"compressibility_{geometry}_DQMC+TB_{Nx}x{Ny}_U{U}_T{temp:.2f}"

    else:
        plt.errorbar(mu_vals, kappa_means, kappa_errs, fmt='o', linewidth=2, markersize=6, label="DQMC")
        # plt.xscale('log')
        
        if save_data:
            save_file = f"compressibility_{geometry}_{Nx}x{Ny}_U{U}_T{temp:.2f}_{npoint_stencil}stencil_{nsweep}sweeps"
            # np.savetxt(f"/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/{save_file}.csv", np.c_[mu_vals, n_means, n_errs], delimiter="\t", header="mu\t<n>\t<n> error")
            np.savetxt(os.path.join(output_dir, f"{save_file}.csv"), np.c_[mu_vals, kappa_means, kappa_errs], delimiter="\t", header="mu\tkappa\tkappa error")


    plt.xlabel('$\\mu$', fontsize=12)
    plt.ylabel('$\\kappa$', fontsize=12)
    plt.title(f'Compressibility for {Nx}x{Ny} {geometry} lattice, U={U}, T={temp:.2f}', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_data:
        # plt.savefig(f"/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/{save_file}.png", dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{save_file}.png"), dpi=300, bbox_inches='tight')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot density and/or compressibility from DQMC data')
    parser.add_argument('--input_dir', required=True, help='Path to input data directory')
    parser.add_argument('--output_dir', required=True, help='Path to output directory for plots and .txt files.')
    parser.add_argument('--plot', required=True, choices=['density', 'compressibility', 'both'], 
                        help='What to plot: density, compressibility, or both')
    parser.add_argument('--tb_file', default=None, help='Optional path to tight-binding data file')
    parser.add_argument('--npoint_stencil', default=3, type=int, help='Number of points to use in central difference for compressibility.')
    
    args = parser.parse_args()

    # Check for incompatible options
    if args.plot == 'both' and args.tb_file is not None:
        parser.error("Cannot use --plot both with --tb_file.")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.plot in ['density', 'both']:
        plot_density_jackknife(args.input_dir, args.output_dir, args.tb_file)
    
    if args.plot in ['compressibility', 'both']:
        plot_compressibility_jackknife(args.input_dir, args.output_dir, args.tb_file, save_data=True, npoint_stencil=args.npoint_stencil)


def main_vs():
    # TODO: modify this so it can take path in as a command line argument

    # SQUARE FILEPATHS
    path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/mu_sweep_square_6x6_U0_T1.0_10000sweeps"

    # HONEYCOMB FILEPATHS
    # TB_file = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/density_square_TB_T1.0.csv"
    # TB_file = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/compressibility_square_TB_T1.0.csv"

    out_path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/"

    # plot_density_jackknife(path, out_path)
    plot_compressibility_jackknife(path, out_path, TB_file=None, save_data=False, npoint_stencil=5)


if __name__ == "__main__":
    # If running using command line arguments
    main()
    # main_vs()
    
    
