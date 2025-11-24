import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from time import time
import os
import logging
from typing import Any, Union, Tuple

from goto_swap.backends.jax.emd import emd_1d_sorted_jax_differentiable
from cryojax.io import read_atoms_from_pdb
import cryojax.simulator as cxs
from cryojax.ndimage.operators import FourierGaussian

from joint_gradient_descent import sw_sorted_jax_differentiable
from joint_gradient_descent import (
    SpecimenConfig,
    SetupConfig,
    apply_all_cv,
)
from tqdm import tqdm


@dataclass
class LandscapeConfig:
    landscape_variables: dict[str, dict[str, Any]] 
    n_trials: int
    n_trials_chunks: int

    def __post_init__(self):
        if len(self.landscape_variables) != 2:
            raise ValueError("Exactly 2 landscape variables are supported.")
        for key in self.landscape_variables.keys():
            if key not in ['rotation', 'defocus', 'offset_x_in_angstroms', 'offset_y_in_angstroms']:
                raise ValueError(f"Landscape variable {key} not recognized.")


@dataclass
class SlicedWassersteinCdfConfig:
    n_projections: int
    crop_to_box_size: int
    method: str = 'cdf'
    def __post_init__(self):
        self.method = 'cdf'


@dataclass
class SlicedWassersteinICdfConfig:
    n_projections: int
    crop_to_box_size: int
    n_interpolation_points: int
    method: str = 'icdf'
    def __post_init__(self):
        self.method = 'icdf'


@dataclass
class SlicedWassersteinDifferentiableSortConfig:
    n_projections: int
    crop_to_box_size: int
    method: str = 'dsort'
    def __post_init__(self):
        self.method = 'dsort'

@dataclass
class SlicedWassersteinRandomPixelsConfig:
    crop_to_box_size: int
    n_projections: int
    n_pixels: int
    random_seed: int
    n_projections_chunk_size: int
    normalize_safely_epsilon: float = 1e-12

    # post init for n_projections divisible by chunk size
    def __post_init__(self):
        if self.n_projections % self.n_projections_chunk_size == 0:
            raise ValueError("n_projections must be divisible by n_projections_chunk_size")


@dataclass
class EmdConfig:
    crop_to_box_size: int
    kwargs: dict[str, Any]

@dataclass
class ObjectiveConfig:
    sliced_wasserstein_cdf: SlicedWassersteinCdfConfig
    sliced_wasserstein_icdf: SlicedWassersteinICdfConfig
    sliced_wasserstein_random_pixels: SlicedWassersteinRandomPixelsConfig
    sliced_wasserstein_dsort: SlicedWassersteinDifferentiableSortConfig
    use_this_objective: str 
    emd: EmdConfig

    # post init to check objective type
    def __post_init__(self):
        if self.use_this_objective not in ['sw_random_pixels', 'emd', 'sw_cdf', 'sw_icdf', 'sw_dsort']:
            raise ValueError("use_this_objective must be either 'sw_random_pixels' or 'emd' or 'sliced_wasserstein_cdf'.")


@dataclass
class FullConfig:
    setup: SetupConfig
    landscape: LandscapeConfig
    objective: ObjectiveConfig


@hydra.main(version_base=None, config_path="./", config_name="config_balancing_tradeoff_loss_landscape")
def main(cfg: FullConfig):

    logger = logging.getLogger(__name__)
    config = OmegaConf.merge(OmegaConf.structured(FullConfig), cfg)
    
    run_dir = HydraConfig.get().run.dir 
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    logger.info(f"Saving traces to {run_dir}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    def setup_specimen(specimen: SpecimenConfig):
        atom_positions, _ = read_atoms_from_pdb(
            specimen.pdb_file, 
            center=specimen.center, 
            selection_string=specimen.selection_string,
            loads_properties=False
        )
        return atom_positions
    atom_positions = setup_specimen(config.setup.specimen)

    n_pix = config.setup.image.pad_size
    image_config = cxs.BasicImageConfig(
        (config.setup.image.grid_size, config.setup.image.grid_size), 
        config.setup.image.pixel_size, 
        voltage_in_kilovolts=config.setup.ctf.voltage, 
        pad_options=dict(shape=(n_pix, n_pix))
    )
    volume = cxs.GaussianMixtureVolume(
                atom_positions,
                amplitudes=config.setup.integrator.atom_amplitude,
                variances=config.setup.integrator.atom_width,
            )
    setup_pose_wxyz = cxs.EulerAnglePose(
        phi_angle=config.setup.pose.euler_angles_deg.phi_angle,
        theta_angle=config.setup.pose.euler_angles_deg.theta_angle,
        psi_angle=config.setup.pose.euler_angles_deg.psi_angle,
        ).rotation.wxyz
    pose = cxs.QuaternionPose(wxyz=setup_pose_wxyz, 
        offset_x_in_angstroms=config.setup.pose.translation.offset_x_in_angstroms, 
        offset_y_in_angstroms=config.setup.pose.translation.offset_y_in_angstroms
        )

    image_model = cxs.make_image_model(
        volume,
        image_config,
        pose=pose,
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF(defocus_in_angstroms=config.setup.ctf.defocus_in_angstroms),
                                                   envelope = FourierGaussian(amplitude=config.setup.envelope.amplitude, b_factor=config.setup.envelope.b_factor),
                                                   ),
        simulates_quantity=True,
        quantity_mode='intensity'
    )
    
    logger.info(f"image_model: {image_model}")
    jnp.save(os.path.join(run_dir, "data/initial_image_model.npy"), image_model )

    image_model_obs = apply_all_cv(
        image_model=image_model,
        wxyz=pose.wxyz,
        offset_x_in_angstroms=config.setup.pose.translation.offset_x_in_angstroms,
        offset_y_in_angstroms=config.setup.pose.translation.offset_y_in_angstroms,
        cv_scalar=config.setup.cv.value,
        defocus_scalar=config.setup.ctf.defocus_in_angstroms,
    )

    proj_obs_clean = image_model_obs.simulate()
    proj_obs_noisy = jax.random.poisson(jax.random.PRNGKey(0),  config.setup.noise.dose*(proj_obs_clean + config.setup.noise.background))

    plt.imshow(proj_obs_clean, cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(run_dir, "plots/proj_obs_clean.png"))
    plt.clf()

    plt.imshow(proj_obs_noisy, cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(run_dir, "plots/proj_obs_noisy.png"))
    plt.clf()

    jnp.save(os.path.join(run_dir, "data/proj_obs_clean.npy"), proj_obs_clean)
    jnp.save(os.path.join(run_dir, "data/proj_obs_noisy.npy"), proj_obs_noisy)

    

    cv_dict = {
            'rotation': pose.wxyz,
            'defocus': jnp.array(config.setup.ctf.defocus_in_angstroms),
            'offset_x_in_angstroms': jnp.array(config.setup.pose.translation.offset_x_in_angstroms),
            'offset_y_in_angstroms': jnp.array(config.setup.pose.translation.offset_y_in_angstroms),
            'cv': jnp.array(config.setup.cv.value),
        }

    for var_name, var_config in config.landscape.landscape_variables.items():
        cv_dict[var_name] = jnp.broadcast_to(cv_dict[var_name], (var_config.n_points,) + cv_dict[var_name].shape)


        if var_name == 'rotation':

            def axis_angle_to_quaterion(axis, angle):
                half_angle = angle / 2.0
                w = jnp.cos(half_angle)
                x = axis[0] * jnp.sin(half_angle)
                y = axis[1] * jnp.sin(half_angle)
                z = axis[2] * jnp.sin(half_angle)
                return jnp.array([w, x, y, z])


            wxyz_list = []
            for angle in jnp.linspace(var_config['min_geodesic_angle_in_degrees'],
                                        var_config['max_geodesic_angle_in_degrees'],
                                        var_config['n_points']):
                wxyz = axis_angle_to_quaterion(
                    axis=jnp.array(config.landscape.landscape_variables['rotation']['axis']),
                    angle=angle * (jnp.pi / 180.0)
                )
                wxyz_list.append(wxyz)
            cv_dict['rotation'] = jnp.array(wxyz_list)
        elif var_name == 'defocus':
            cv_dict['defocus'] = jnp.linspace(
                var_config['min_value_in_angstroms'],
                var_config['max_value_in_angstroms'],
                var_config['n_points'],
            )
        elif var_name == 'offset_x_in_angstroms':
            cv_dict['offset_x_in_angstroms'] = jnp.linspace(
                var_config['min_value_in_angstroms'],
                var_config['max_value_in_angstroms'],
                var_config['n_points'],
            )
        elif var_name == 'offset_y_in_angstroms':
            cv_dict['offset_y_in_angstroms'] = jnp.linspace(
                var_config['min_value_in_angstroms'],
                var_config['max_value_in_angstroms'],
                var_config['n_points'],
            )
        elif var_name == 'cv':
            cv_dict['cv'] = jnp.linspace(
                var_config['min_value'],
                var_config['max_value'],
                var_config['n_points'],
            )
    logger.info(f"cv_dict: {cv_dict}")

    key1, key2 = config.landscape.landscape_variables.keys()
    cv_grid = cv_dict[key1]
    other_grid = cv_dict[key2]

    if config.objective.use_this_objective == 'emd':
        import ot
        H = W = config.objective.emd.crop_to_box_size
        n_pix = proj_obs_noisy.shape[0]
        crop_start_idx, crop_end_idx = (n_pix - H)//2, (n_pix + H)//2
        img1 = proj_obs_noisy[crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]
        assert img1.shape == (H, W)


        def setup_cost_matrix(H, W):
            xs = jnp.arange(H)
            ys = jnp.arange(W)
            X, Y = jnp.meshgrid(xs, ys, indexing="ij")            # shape (H, W)
            coords = jnp.stack([X.ravel(), Y.ravel()], axis=1)    # (n_pixels, 2)
            diff = coords[:, None, :] - coords[None, :, :]        # (n, n, 2)
            C = jnp.sum(diff**2, axis=-1)                         # (n, n) squared Euclidean distances
            return C
        C = setup_cost_matrix(H, W)

        def full_wasserstein_distance(img1, img2, C, **kwargs):
            """Compute the full Wasserstein distance between two histograms a and b with cost matrix C."""
            a = img1.ravel()
            b = img2.ravel()
            a = a / a.max()
            b = b / b.max()
            a /= a.sum()
            b /= b.sum()
            w_dist = ot.emd2(a, b, C, **kwargs)  # optimal transport plan
            return w_dist


        emd_landscape = jnp.zeros((len(cv_grid), len(other_grid)))
        for i, cv in enumerate(tqdm(cv_grid, desc="Computing EMD landscape")):
            for j, other in enumerate(other_grid):

                wxyz = pose.wxyz
                defocus = config.setup.ctf.defocus_in_angstroms
                offset_x = config.setup.pose.translation.offset_x_in_angstroms
                offset_y = config.setup.pose.translation.offset_y_in_angstroms
                cv_scalar = config.setup.cv.value
                if 'rotation' == key1:
                    wxyz = cv
                elif 'defocus' == key1:
                    defocus = cv
                elif 'offset_x_in_angstroms' == key1:
                    offset_x = cv
                elif 'offset_y_in_angstroms' == key1:
                    offset_y = cv
                elif 'cv' == key1:
                    cv = cv
                if 'rotation' == key2:
                    wxyz = other
                elif 'defocus' == key2:
                    defocus = other
                elif 'offset_x_in_angstroms' == key2:
                    offset_x = other
                elif 'offset_y_in_angstroms' == key2:
                    offset_y = other
                elif 'cv' == key2:
                    cv = other
                new_image_model = apply_all_cv(
                    image_model,
                    wxyz=wxyz,
                    cv_scalar=cv_scalar,
                    defocus_scalar=defocus,
                    offset_x_in_angstroms=offset_x,
                    offset_y_in_angstroms=offset_y,
                )
                proj_simulated = config.setup.noise.dose * new_image_model.simulate() + config.setup.noise.background
                img2 = proj_simulated[crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]
                assert img2.shape == (H, W)
                emd_landscape = emd_landscape.at[i,j].set(full_wasserstein_distance(img1.astype(jnp.float32), img2.astype(jnp.float32), C, **config.objective.emd.kwargs))  # scalar transport cost with squared-euclidean cost
        print("emd2 (squared-Wasserstein cost):", emd_landscape) 
        jnp.save(os.path.join(run_dir, "data/emd_landscape.npy"), emd_landscape)
        landscape_for_plotting = emd_landscape


    elif config.objective.use_this_objective in ['sw_cdf', 'sw_icdf', 'sw_dsort']:
        if config.objective.use_this_objective == 'sw_cdf':
            sliced_wasserstein = config.objective.sliced_wasserstein_cdf
            wasserstein_distance_loss_images_kwargs = dict()

        elif config.objective.use_this_objective == 'sw_icdf':
            sliced_wasserstein = config.objective.sliced_wasserstein_icdf
            wasserstein_distance_loss_images_kwargs = dict(n_interpolation_points=sliced_wasserstein.n_interpolation_points)

        elif config.objective.use_this_objective == 'sw_dsort':
            sliced_wasserstein = config.objective.sliced_wasserstein_dsort
            wasserstein_distance_loss_images_kwargs = dict()

        H = W = sliced_wasserstein.crop_to_box_size
        n_pix = proj_obs_noisy.shape[0]
        crop_start_idx, crop_end_idx = (n_pix - H)//2, (n_pix + H)//2
        img1 = proj_obs_noisy[crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]

        from goto_swap import get_backend
        jax_backend = get_backend("jax")
        sw_cdf_landscape = jnp.zeros((1, len(cv_grid), len(other_grid), sliced_wasserstein.n_projections))
        for i, cv in enumerate(tqdm(cv_grid, desc="Computing SW CDF landscape")):
            for j, other in enumerate(other_grid):

                wxyz = pose.wxyz
                defocus = config.setup.ctf.defocus_in_angstroms
                offset_x = config.setup.pose.translation.offset_x_in_angstroms
                offset_y = config.setup.pose.translation.offset_y_in_angstroms
                cv_scalar = config.setup.cv.value
                if 'rotation' == key1:
                    wxyz = cv
                elif 'defocus' == key1:
                    defocus = cv
                elif 'offset_x_in_angstroms' == key1:
                    offset_x = cv
                elif 'offset_y_in_angstroms' == key1:
                    offset_y = cv
                elif 'cv' == key1:
                    cv = cv
                if 'rotation' == key2:
                    wxyz = other
                elif 'defocus' == key2:
                    defocus = other
                elif 'offset_x_in_angstroms' == key2:
                    offset_x = other
                elif 'offset_y_in_angstroms' == key2:
                    offset_y = other
                elif 'cv' == key2:
                    cv = other
                new_image_model = apply_all_cv(
                    image_model,
                    wxyz=wxyz,
                    cv_scalar=cv_scalar,
                    defocus_scalar=defocus,
                    offset_x_in_angstroms=offset_x,
                    offset_y_in_angstroms=offset_y,
                )
                proj_simulated = config.setup.noise.dose * new_image_model.simulate() + config.setup.noise.background
                img2 = proj_simulated[crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]
                assert img2.shape == (H, W)
                sw_1d_cdf_values = jax_backend.wasserstein_distance_loss_images(img1.astype(jnp.float32).reshape(1, H, W), 
                                                                                img2.astype(jnp.float32).reshape(1, H, W), 
                                                                                num_rotations=sliced_wasserstein.n_projections,
                                                                                method=sliced_wasserstein.method,
                                                                                **wasserstein_distance_loss_images_kwargs)
                sw_cdf_landscape = sw_cdf_landscape.at[:,i,j].set(sw_1d_cdf_values)  
        jnp.save(os.path.join(run_dir, "data/sw_landscape.npy"), sw_cdf_landscape)
        landscape_for_plotting = jnp.sqrt(sw_cdf_landscape.mean(axis=(0, -1)))

    elif config.objective.use_this_objective =='sw_random_pixels':
        sliced_wasserstein = config.objective.sliced_wasserstein_random_pixels
        logger.info(f"Computing SW landscape for sw objective: {sliced_wasserstein}...")
        n_projections, n_pixels = sliced_wasserstein.n_projections, sliced_wasserstein.n_pixels
        H = W = sliced_wasserstein.crop_to_box_size
        n_pix = proj_obs_noisy.shape[0]
        crop_start_idx, crop_end_idx = (n_pix - H)//2, (n_pix + H)//2
        img1 = proj_obs_noisy[crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]

        def sw_sorted_jax_differentiable_theta_chunk(image_1, image_2, grid, n_pixels, normalize_safely_epsilon, key, theta):
        
            v_weights = image_1.flatten()
            u_weights = image_2.flatten()

            def loss_per_theta(theta, key):
                project = jnp.tensordot(grid, jnp.array([jnp.cos(theta), jnp.sin(theta)]), axes=1).flatten()
                perm = jax.random.permutation(key, len(project))[:n_pixels]
                x_ab = project[perm]
                sort_idx = jnp.argsort(x_ab)
                x_ab_sorted = x_ab[sort_idx]
                a_sorted = u_weights[perm][sort_idx] + normalize_safely_epsilon
                b_sorted = v_weights[perm][sort_idx] + normalize_safely_epsilon
                return emd_1d_sorted_jax_differentiable(x_ab_sorted, x_ab_sorted, a_sorted / a_sorted.sum(), b_sorted / b_sorted.sum())

            return loss_per_theta(theta, key)

        def _sw_sorted_jax_differentiable(image_1, image_2, grid, n_pixels, normalize_safely_epsilon, theta, key):
            return sw_sorted_jax_differentiable_theta_chunk(image_1, image_2, grid, n_pixels, normalize_safely_epsilon, key, theta)
    
        xm, ym = jnp.meshgrid(
            jnp.linspace(-H // 2, H // 2 - 1, H),
            jnp.linspace(-H // 2, H // 2 - 1, H),
            indexing='ij',
        )
        grid = jnp.stack([xm, ym], axis=-1)

        def eval_sw_single_theta(cv, other, theta, key):
            wxyz = pose.wxyz
            defocus = config.setup.ctf.defocus_in_angstroms
            offset_x = config.setup.pose.translation.offset_x_in_angstroms
            offset_y = config.setup.pose.translation.offset_y_in_angstroms
            cv_scalar = config.setup.cv.value
            if 'rotation' == key1:
                wxyz = cv
            elif 'defocus' == key1:
                defocus = cv
            elif 'offset_x_in_angstroms' == key1:
                offset_x = cv
            elif 'offset_y_in_angstroms' == key1:
                offset_y = cv
            elif 'cv' == key1:
                cv = cv
            if 'rotation' == key2:
                wxyz = other
            elif 'defocus' == key2:
                defocus = other
            elif 'offset_x_in_angstroms' == key2:
                offset_x = other
            elif 'offset_y_in_angstroms' == key2:
                offset_y = other
            elif 'cv' == key2:
                cv = other
            new_image_model = apply_all_cv(
                image_model,
                wxyz=wxyz,
                cv_scalar=cv_scalar,
                defocus_scalar=defocus,
                offset_x_in_angstroms=offset_x,
                offset_y_in_angstroms=offset_y,
            )
            proj_simulated = config.setup.noise.dose * new_image_model.simulate() + config.setup.noise.background
            img2 = proj_simulated[crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]
            return _sw_sorted_jax_differentiable(
                img1,
                img2,
                grid,
                n_pixels,
                sliced_wasserstein.normalize_safely_epsilon,
                theta,
                key,
            )
        thetas = jnp.linspace(0, jnp.pi, n_projections, endpoint=False) 

        theta_vmap = jax.vmap(
            eval_sw_single_theta,
            in_axes=(None, None, 0, 0)   # cv, other, theta_i, key_i
        )

        inner_vmap = jax.vmap(
            theta_vmap,
            in_axes=(None, 0, None, 0)
        )

        full_vmap = jax.vmap(
            inner_vmap,
            in_axes=(0, None, None, 0)
        )

        thetas = jnp.linspace(0, jnp.pi, n_projections, endpoint=False)

        # Chunk over theta to limit peak device memory.
        n_trials = config.landscape.n_trials
        n_chunks = config.landscape.n_trials_chunks
        trial_chunk_size = (n_trials + n_chunks - 1) // n_chunks

        key_main = jax.random.PRNGKey(sliced_wasserstein.random_seed)
        rep_keys = jax.random.split(key_main, n_trials)

        theta_chunk_size = sliced_wasserstein.n_projections_chunk_size
        theta_results_parts = []  # will collect chunks along theta axis

        # iterate over theta chunks
        for t_start in tqdm(range(0, len(thetas), theta_chunk_size), desc="Theta chunks", unit="theta-chunk"):
            t_end = min(t_start + theta_chunk_size, len(thetas))
            thetas_chunk = thetas[t_start:t_end]
            n_theta_chunk = len(thetas_chunk)

            # build a per-trial runner that creates per-(cv,other,theta) keys for this theta chunk
            def run_trial_theta_chunk(k):
                vmap_shape_chunk = (len(cv_grid), len(other_grid), n_theta_chunk)
                ks = jax.random.split(k, vmap_shape_chunk[0] * vmap_shape_chunk[1] * vmap_shape_chunk[2])
                ks = ks.reshape(vmap_shape_chunk + (2,))
                return full_vmap(cv_grid, other_grid, thetas_chunk, ks)

            # vmapped + jitted over trials for this theta chunk (compiled for the chunk shape)
            vmapped_run_trial_for_chunk = jax.jit(jax.vmap(run_trial_theta_chunk))

            # run over trial-chunks, transfer each chunk to host to free device memory between iterations
            trial_chunks = []
            total_trial_chunks = (n_trials + trial_chunk_size - 1) // trial_chunk_size
            for i in tqdm(range(0, n_trials, trial_chunk_size), total=total_trial_chunks,
                        desc=f"Theta {t_start}:{t_end}", unit="trial-chunk"):
                keys_chunk = rep_keys[i : i + trial_chunk_size]
                res_chunk = vmapped_run_trial_for_chunk(keys_chunk)   # shape: (chunk_size, n_cv, n_other, n_theta_chunk)
                res_chunk.block_until_ready()
                res_chunk_host = jax.device_get(res_chunk).astype(np.float32)
                trial_chunks.append(res_chunk_host)

            # concatenate trial chunks -> (n_trials, n_cv, n_other, n_theta_chunk)
            chunked_by_trials = np.concatenate(trial_chunks, axis=0)
            theta_results_parts.append(chunked_by_trials)

        # concatenate along theta axis -> (n_trials, n_cv, n_other, n_projections)
        sw_landscapes_host = np.concatenate(theta_results_parts, axis=-1)
        logger.info(f"sw_landscapes_host shape: {sw_landscapes_host.shape}")

        jnp.save(os.path.join(run_dir, "data/sw_landscape.npy"), sw_landscapes_host)
        landscape_for_plotting = sw_landscapes_host.mean(axis=(0, -1))

    if key1 == 'rotation':
        cv_grid_min = config.landscape.landscape_variables['rotation']['min_geodesic_angle_in_degrees']
        cv_grid_max = config.landscape.landscape_variables['rotation']['max_geodesic_angle_in_degrees']
    else:
        cv_grid_min = cv_grid.min()
        cv_grid_max = cv_grid.max()
    if key2 == 'rotation':
        other_grid_min = config.landscape.landscape_variables['rotation']['min_geodesic_angle_in_degrees']
        other_grid_max = config.landscape.landscape_variables['rotation']['max_geodesic_angle_in_degrees']
    else:
        other_grid_min = other_grid.min()
        other_grid_max = other_grid.max()
    plt.imshow(landscape_for_plotting, cmap='jet',
               extent=(other_grid_min, other_grid_max, cv_grid_min, cv_grid_max),
               origin='lower', aspect='auto')
    plt.colorbar(label=f"{config.objective.use_this_objective.capitalize()}")
    plt.xlabel(f"{key2} ")
    plt.ylabel(f"{key1} ")
    plt.title(f"{config.objective.use_this_objective.capitalize()} Loss Landscape")
    plt.savefig(os.path.join(run_dir, f"plots/{config.objective.use_this_objective}_landscape.png"))
    plt.clf()

if __name__ == "__main__":
    main()