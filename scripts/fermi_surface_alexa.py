#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fermi_surface_alexa.py 
atyberg@stanford.edu

Plots the fermi surface in reciprocal space as a function of kx and ky. Uses the unequal time Green's functions at tau = beta/2
as a proxy for the spectral function A(k, w=0).

Usage example:
    ADD IN OWN
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import util
import argparse

Nx, Ny, Norb, mu, U = 0, 0, 0, 0, 0
geometry = ""
t = 1

# def load_gt0(base_dir, T):
#     """Loads the unequal time Green's functions for each temperature in a directory, given the path to the parent directory."""

#     global Nx, Ny, Norb, mu, U, geometry
#     gt0_dict = {}

def get_sign(path: str) -> np.ndarray:
    ns, s = util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/sign")
    # use only completed bins
    mask = ns == ns.max()

    return s[mask]

def get_gt0(path):
    """Takes in path to a single temperature folder and loads gt0 data. Returns gt0 of shape (Nbins, L, num_ij), already divided by sign."""

    global Nx, Ny, Norb, mu, U, geometry
    Nx, Ny, Norb, L, geometry, trans_sym, U, mu = util.load_firstfile(path, "metadata/Nx", "metadata/Ny", "metadata/Norb", "params/L", "metadata/geometry", "metadata/trans_sym", "metadata/U", "metadata/mu")
    geometry = geometry[()].decode('utf-8') # now it's a string, not an array with a bytes object

    ns, gt0 = util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/gt0")

    # Reshape gt0 array
    if trans_sym:
        if geometry == "square":
            num_ij = Nx * Ny
        elif geometry == "honeycomb":
            num_ij = Norb * Norb * Ny * Nx
    else:
        N = Nx * Ny * Norb
        num_ij = N * N

    gt0.shape = -1, L, num_ij # note that now you have to keep track of indexing to access a specified r vector

    # use only completed bins
    mask = ns == ns.max()
    gt0 = gt0[mask]

    # Divide by mean of sign
    s = get_sign(path)
    gt0 /= np.mean(s)

    # print(gt0.shape)

    return gt0

def fold_to_first_bz_honeycomb(k_mesh, b1, b2):
    """
    Fold k-points into the first Brillouin zone (hexagon) of honeycomb lattice.
    Note: written by Claude.
    
    Parameters:
    -----------
    k_mesh : array, shape (N, 2)
        k-points to fold
    b1, b2 : array, shape (2,)
        Reciprocal lattice vectors
    
    Returns:
    --------
    k_folded : array, shape (N, 2)
        k-points folded into first BZ
    """
    k_folded = k_mesh.copy()
    
    # Reciprocal lattice vectors for translations
    G_vectors = []
    max_shell = 3  # usually sufficient
    for n1 in range(-max_shell, max_shell+1):
        for n2 in range(-max_shell, max_shell+1):
            G_vectors.append(n1 * b1 + n2 * b2)
    G_vectors = np.array(G_vectors)
    
    # For each k-point, find the G that minimizes |k - G|
    for i in range(len(k_mesh)):
        k = k_mesh[i]
        # Calculate distances to all translated versions
        k_translated = k[None, :] - G_vectors
        distances = np.linalg.norm(k_translated, axis=1)
        # Pick the one closest to Gamma
        min_idx = np.argmin(distances)
        k_folded[i] = k_translated[min_idx]
    
    return k_folded

def fourier_transform_gt0_bins(path, gt0_proxy, fold_k):
    """Performs the Fourier transform from G(r) to G(k) for a 2d k mesh using the allowed k points set by the lattice size.
    gt0_proxy is of shape (Nbins, num_ij). Assumes data is binned (so before jackknife resampling).
    
    Returns
    -------
        k_mesh : ndarray
            Mesh of k points of shape (Npoints, 2) where the [:,0] entries give the kx coordinate and the [:,1] entries give ky coordinate

        gt0_k : ndarray
            Fourier transformed green's functions of k, shape (Nbins, Npoints). Npoints is the number of k points. Can fetch the kx and ky
            coordinates of a given point from the k_mesh array.
    """

    Nx, Ny, Norb, L, geometry, trans_sym = util.load_firstfile(path, "metadata/Nx", "metadata/Ny", "metadata/Norb", "params/L", "metadata/geometry", "metadata/trans_sym")
    geometry = geometry[()].decode('utf-8') # now it's a string, not an array with a bytes object


    # First transform the r vector from an index to actual coordinates
    num_ij = gt0_proxy.shape[1]
    # print(num_ij)

    # The r indexing depends on geometry and trans_sym
    if geometry == "square":
        # define the k mesh over the whole brillouin zone
        # for square, BZ bounds are +- pi/a
        # rx and ry are translations along lattice vectors. define lattice vectors.
        a = 1
        a1 = np.array([a, 0])
        a2 = np.array([0, a])

        kx_vals = 2 * np.pi / Nx / a * np.arange(-Nx//2, Nx//2+1)
        ky_vals = 2 * np.pi / Nx / a * np.arange(-Ny//2, Ny//2+1)

        # kx_vals = np.arange(-np.pi/a, np.pi/a + k_spacing, k_spacing)
        # ky_vals = np.arange(-np.pi/a, np.pi/a + k_spacing, k_spacing)
        # kx_vals = np.arange(0, np.pi/a + k_spacing, k_spacing)
        # ky_vals = np.arange(0, np.pi/a + k_spacing, k_spacing)
        # ky_vals = np.arange(-np.pi/a, 0 + k_spacing, k_spacing)


        kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)

        # print(kx_grid.shape)

        # print(Kx)
        # print(Ky)

        k_mesh = np.stack([kx_grid.flatten(), ky_grid.flatten()], axis=1) # mesh with [:, 0] corresponding to the kx coordinates and [:, 1] the ky coordinates
        # print(k_mesh.shape) 

        # print(k_mesh[:5,:])

        fourier_sum = np.zeros((gt0_proxy.shape[0],k_mesh.shape[0]), dtype=np.complex128)
        # print(fourier_sum.shape)

        if trans_sym:
            for r in range(num_ij): # change back to num_ij
                # indexing is r = rx + Nx * ry
                rx = r % Nx
                ry = r // Nx
                print(f"(rx, ry) = ({rx}, {ry})")

                rx_neg = Nx - rx

                rvec = rx * a1 + ry * a2

                # compute the dot product of rvec and kvec for every k point in the mesh
                k_dot_r = k_mesh[:,0] * rvec[0] + k_mesh[:,1] * rvec[1]
                # k_dot_r_bins = np.tile(k_dot_r[None,:], (100, 1))

                # if r == 10:
                #     print(k_dot_r[:4])
                #     print(k_dot_r_bins.shape)
                #     print(k_dot_r_bins[0,:4])
                #     print(k_dot_r_bins[1,:4])

                fourier_sum += gt0_proxy[:,r,None] * np.exp(-1j * k_dot_r[None,:])

            # print(f"G[k] shape: {fourier_sum.shape}")
            gt0_k = fourier_sum
                
        else:
            raise NotImplementedError
        
    elif geometry == "honeycomb":
        # define real and reciprocal space lattice vectors
        a = 1
        a1 = np.array([3*a/2, np.sqrt(3)*a/2])
        a2 = np.array([3*a/2, -np.sqrt(3)*a/2])
        b1 = np.array([2*np.pi/3/a, 2*np.pi/np.sqrt(3)/a])
        b2 = np.array([2*np.pi/3/a, -2*np.pi/np.sqrt(3)/a])

        # define spacing between nearest neighbor points (which is also the spacing between the two sites in the basis)
        dvec = np.array([a, 0]) # vector from A to B site within unit cell

        # define allowed k points using reciprocal lattice vectors b1 and b2
        mx_arr = np.arange(-Nx//2, Nx//2+1) # NOTE: potentially change this because we may have too many k points if we include the endpoint
        my_arr = np.arange(-Ny//2, Ny//2+1)
        mx_grid, my_grid = np.meshgrid(mx_arr, my_arr)

        mxy_mesh = np.stack([mx_grid.flatten(), my_grid.flatten()], axis=1) # mesh with [:, 0] corresponding to the mx values and [:, 1] the my coordinates
        k_mesh = mxy_mesh[:,0,None] / Nx * b1[None,:] + mxy_mesh[:,1,None] / Ny * b2[None,:]
        # print(k_mesh.shape)
        # print(k_mesh)

        fourier_sum = np.zeros((gt0_proxy.shape[0],k_mesh.shape[0]), dtype=np.complex128)

        if trans_sym:
            for r in range(num_ij):
                # have to convert indexing for honeycomb to real space translation vectors
                # recall that r = dx + Nx * dy + Nx*Ny * jo + NxNyNorb * io
                # where dx, dy are the integer number of a1 and a2 vectors it has been translated along, respectively
                dx = r % Nx
                temp1 = r // Nx
                dy = temp1 % Ny
                temp2 = temp1 // Ny
                jo = temp2 % Norb
                io = temp2 // Norb

                # create displacement vector between sites
                if io == jo:
                    rvec = dx * a1 + dy * a2
                elif (io==0 and jo==1):
                    rvec = dx * a1 + dy * a2 + dvec
                elif (io==1 and jo==0):
                    rvec = dx * a1 + dy * a2 - dvec

                # dot product of rvec and kvec for every k point in the mesh
                k_dot_r = k_mesh[:,0] * rvec[0] + k_mesh[:,1] * rvec[1]

                fourier_sum += gt0_proxy[:,r,None] * np.exp(-1j * k_dot_r[None,:])

            gt0_k = fourier_sum
            if fold_k:
                k_mesh = fold_to_first_bz_honeycomb(k_mesh, b1, b2)

        else:
            raise NotImplementedError

    return k_mesh, gt0_k # maybe change this to either just k_mesh or just the grids

def plot_spectral_proxy(path, output_dir, gt0, show_BZ=False, fold_k=True):
    """
    Computes a proxy for A(k, w=0) using the unequal-time green's function.

    Args
    ----
        gt0 : ndarray
            Unequal time Green's function, of shape (Nbins, L, num_ij)
    """

    # Retrieve the tau=beta/2 slice (the only one we care about for the proxy) 
    L, beta = util.load_firstfile(path, "params/L", "metadata/beta")
    # L = L[0]
    # dt, L = util.load_firstfile(folder_path, "params/dt", "params/L")
    # tau_vals = np.arange(L) * dt
    gt0_proxy = gt0[:, L//2, :]
    # print("gt0_proxy shape:", gt0_proxy.shape)

    k_mesh, gt0_k = fourier_transform_gt0_bins(path, gt0_proxy, fold_k)
    # print(k_mesh.shape)
    print(gt0_k.shape)

    # Now do jackknife resampling to get mean and error for each of the k points
    # will take gt0_k of shape (Nbins, Npoints) and spit out an averaged array of shape (Npoints,)
    gt0_k_mean, gt0_k_err = util.jackknife(np.ones_like(gt0_k), gt0_k)
    # print(gt0_k_mean.shape)
    # print(gt0_k_mean[100])

    spectral_proxy = beta * gt0_k_mean

    # # FOR TESTING TO FIND K POINT
    # # index of the K point at (0, 4pi/3a)
    # a = 1
    # K_target = np.array([0, 4*np.pi/(3*np.sqrt(3)*a)])
    # distances = np.linalg.norm(k_mesh - K_target, axis=1)
    # K_idx = np.argmin(distances)

    plt.figure(figsize=(8, 6))
    # plt.scatter(k_mesh[:, 0], k_mesh[:, 1], c=spectral_proxy, vmin=0, vmax=10, marker='o', edgecolors='none', cmap='viridis', s=500)
    plt.scatter(k_mesh[:, 0], k_mesh[:, 1], c=spectral_proxy, marker='o', edgecolors='none', cmap='viridis', s=500)
    # plt.scatter(k_mesh[K_idx, 0], k_mesh[K_idx, 1], c=spectral_proxy[K_idx], marker='o', edgecolors='none', cmap='viridis', s=500) # for testing to find index of peak
    if show_BZ:
        if geometry=="honeycomb":
            # define the first BZ corners
            a=1
            bz_corners_x = np.array([2*np.pi/3/a, 2*np.pi/3/a, 0, -2*np.pi/3/a, -2*np.pi/3/a, 0, 2*np.pi/3/a])
            bz_corners_y = np.array([2*np.pi/3/a/np.sqrt(3), -2*np.pi/3/a/np.sqrt(3), -4*np.pi/3/a/np.sqrt(3), -2*np.pi/3/a/np.sqrt(3), 2*np.pi/3/a/np.sqrt(3), 4*np.pi/3/a/np.sqrt(3), 2*np.pi/3/a/np.sqrt(3)])
            plt.plot(bz_corners_x, bz_corners_y, linewidth=2, markersize=100, label='first BZ boundary')
            plt.legend()
        elif geometry=="square":
            a=1
            bz_corners_x = np.array([np.pi/a, np.pi/a, -np.pi/a, -np.pi/a, np.pi/a])
            bz_corners_y = np.array([np.pi/a, -np.pi/a, -np.pi/a, np.pi/a, np.pi/a])
            plt.plot(bz_corners_x, bz_corners_y, linewidth=2, label='first BZ boundary')
            plt.legend(loc='upper right')


    # plt.pcolormesh(k_mesh[:,0], k_mesh[:,1], spectral_proxy.reshape(k_mesh[:,0].shape), cmap='viridis', shading='auto')
    plt.colorbar(label='$ \\beta G(k, \\tau=\\beta/2)$')
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.title(f'Spectral function proxy for {Nx}x{Ny} {geometry} lattice, U={U}, mu={mu:.2f}, T={1/beta:.1f}', fontsize=10)
    plt.axis('equal')
    plt.tight_layout()

    save_file = f"spectral_func_{geometry}_{Nx}x{Ny}_U{U}_mu{mu}_T{1/beta:.1f}"
    plt.savefig(os.path.join(output_dir, f"{save_file}.png"), dpi=300, bbox_inches='tight') 
    # NOTE: bring this saving back after you stop playing with it

    plt.show()

def plot_peak_intensity(base_dir, output_dir):
    """Plots the peak intensity at the upper K point at k-coordinates 1/3(b1 + b2) wrt temperature. Base_dir is the
    parent folder with all the temperature subfolders."""

    temps = []
    intensities = []
    errors = []
    for folder in os.listdir(base_dir): # each folder represents a temperature
        if not folder.startswith("T_"):
            continue
        T = float(folder[2:])
        folder_path = os.path.join(base_dir, folder) + "/"

        gt0 = get_gt0(folder_path)
        L, beta, nsweep_meas = util.load_firstfile(folder_path, "params/L", "metadata/beta", "params/n_sweep_meas")
        gt0_proxy = gt0[:, L//2, :]

        k_mesh, gt0_k = fourier_transform_gt0_bins(folder_path, gt0_proxy, fold_k=False)
        # print(gt0_k.shape)

        gt0_k_mean, gt0_k_err = util.jackknife(np.ones_like(gt0_k), gt0_k)
        spectral_proxy = beta * gt0_k_mean
        spectral_proxy_err = beta * gt0_k_err

        # find K point using a search over the coordinate mesh and minimizing distance to K=(0, 4π/3a)
        # for 6x6 the value is 12
        a = 1
        K_target = np.array([0, 4*np.pi/(3*np.sqrt(3)*a)])
        distances = np.linalg.norm(k_mesh - K_target, axis=1)
        K_idx = np.argmin(distances)

        temps.append(T)
        intensities.append(spectral_proxy[K_idx])
        errors.append(spectral_proxy_err[K_idx])

    sorted_indices = np.argsort(temps)
    temps = np.array(temps)[sorted_indices]
    intensities = np.array(intensities)[sorted_indices]
    errors = np.array(errors)[sorted_indices]

    # # If you want to only plot less than a certain temperature
    # mask = temps<1
    # temps = temps[mask]
    # intensities = intensities[mask]
    # errors = errors[mask]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(temps, intensities, errors, fmt='o-', linewidth=2, markersize=6, label=f"U={U}")

    # plt.xscale('log')
    plt.xlabel('$T$', fontsize=12)
    plt.ylabel('$\\beta G(K, \\tau=\\beta/2)$', fontsize=12)
    plt.title(f'$A(k, \\omega)$ proxy peak intensity at K point $k=(0, 4\\pi/3a)$\n{Nx}x{Ny} {geometry} lattice, U={U}, mu={mu:.2f}', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    save_file = f"Kpoint_intensity_{geometry}_{Nx}x{Ny}_U{U}_mu{mu}_{nsweep_meas}sweeps"
    plt.savefig(os.path.join(output_dir, f"{save_file}.png"), dpi=300, bbox_inches='tight') 
    np.savetxt(os.path.join(output_dir, f"{save_file}.csv"), np.c_[temps, intensities, errors], delimiter="\t", header="T\tpeak intensity\terror")

    plt.show()

def main():
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/square_6x6_U6_mu0_bondcorr_400000sweeps/T_0.5/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/beta_sweep_log_square_6x6_U0_mu0_20000sweeps_nbeta50/T_0.1/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/beta_sweep_log_square_6x6_U0_mu0_20000sweeps_nbeta50/T_0.542868/"
    
    # HONEYCOMB PATH
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U0_mu0_10sweeps/T_0.1/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U0.5_mu0_20000sweeps/T_0.1/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U2_mu0_10000sweeps/T_0.1/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U4_mu0_20000sweeps/T_0.1/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U6_mu0_20000sweeps/T_0.1/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U10_mu0_20000sweeps/T_0.1/"
    # path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_3x3_U3_mu0_20000sweeps_betarange2-15_nbeta30/T_0.101148/"

    out_path = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/plots/"

    # # NOTE: uncomment to plot the spectral function proxy for a given temperature to visualize the fermi surface
    # gt0 = get_gt0(path)
    # plot_spectral_proxy(path, out_path, gt0, show_BZ=True, fold_k=False)


    # base_dir = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U0.5_mu0_20000sweeps/"
    # base_dir = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U5_mu0_20000sweeps_betamin2/"
    # base_dir = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_6x6_U3_mu0_20000sweeps_betarange2-15/"
    base_dir = "/Users/alexatyberg/Documents/Stanford/Devereaux_Research/DQMC_code/honeycomb_12x12_U3_mu0_20000sweeps_betarange2-15_nbeta30/"

    plot_peak_intensity(base_dir, out_path)
    
def main_cmd_line():
    parser = argparse.ArgumentParser(description='Plot density and/or compressibility from DQMC data')
    parser.add_argument('--input_path', required=True, help='Path to input data directory. If plotting fermi surface should be temperature specific. If plotting peak intensity vs. temperature should be parent directory. Make sure to use trailing slash.')
    parser.add_argument('--output_dir', required=True, help='Path to output directory for plots.')
    parser.add_argument('--plot', required=True, choices=['fermi_surface', 'K_peak'], 
                        help='What to plot: the whole fermi surface (temperature specific) or the peak intensity at the K point vs. temperature.')
    parser.add_argument('--show_bz', choices=[0,1], default=1, help='Turn off to not show 1st BZ outline.')
    parser.add_argument('--fold_k', choices=[0,1], default=0, help='Turn off to not fold k points into the 1st BZ.')    
    args = parser.parse_args()

    # TODO: ADD CHECK FOR INCOMPATIBLE PLOT AND FILEPATH.

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.plot == 'fermi_surface':
        gt0 = get_gt0(args.input_path)
        plot_spectral_proxy(args.input_path, args.output_dir, gt0, show_BZ=args.show_bz, fold_k=args.fold_k)
    
    if args.plot == 'K_peak':
        plot_peak_intensity(args.input_path, args.output_dir)
    
    
if __name__ == "__main__":
    main()
    # main_cmd_line()
    
    
