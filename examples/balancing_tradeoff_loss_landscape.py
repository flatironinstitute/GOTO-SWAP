import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from time import time
import os
import logging
from typing import Any, Union

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


@dataclass
class LandscapeConfig:
    n_trials_per_point: int 
    n_landscape_values: int
    landscape_variables: dict[str, Any] #dict[str, dict[str, Union[float, list[float]]]]
    n_trials: int

    def __post_init__(self):
        if len(self.landscape_variables) != 2:
            raise ValueError("Exactly 2 landscape variables are supported.")
        for key in self.landscape_variables.keys():
            if key not in ['rotation', 'defocus', 'offset_x_in_angstroms', 'offset_y_in_angstroms']:
                raise ValueError(f"Landscape variable {key} not recognized.")

@dataclass
class SlicedWassersteinConfig:
    n_projections: int
    n_pixels: int
    normalize_safely_epsilon: float = 1e-12

@dataclass
class ObjectiveConfig:
    sliced_wasserstein_list: list[SlicedWassersteinConfig]


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

    proj_obs = image_model_obs.simulate()
    proj_obs_noisy = jax.random.poisson(jax.random.PRNGKey(0),  config.setup.noise.dose*(proj_obs + config.setup.noise.background))

    plt.imshow(proj_obs, cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(run_dir, "plots/proj_obs.png"))
    plt.clf()

    plt.imshow(proj_obs_noisy, cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(run_dir, "plots/proj_obs_noisy.png"))
    plt.clf()

    jnp.save(os.path.join(run_dir, "data/proj_obs.npy"), proj_obs)
    jnp.save(os.path.join(run_dir, "data/proj_obs_noisy.npy"), proj_obs_noisy)
    
    xm, ym = jnp.meshgrid(
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        indexing='ij',
    )
    grid = jnp.stack([xm, ym], axis=-1)


    sliced_wasserstein = config.objective.sliced_wasserstein_list
    for idx, sliced_wasserstein in enumerate(sliced_wasserstein):
        print(f"Computing SW landscape for sw objective: {sliced_wasserstein}...")
        n_projections, n_pixels = sliced_wasserstein.n_projections, sliced_wasserstein.n_pixels
        key = jax.random.PRNGKey(idx)
        

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
                # from numpyro.distributions.directional import ProjectedNormal

                # def quaternion_rejection_sampling(min_geodesic_angle_in_degrees, max_geodesic_angle_in_degrees, n_points, key):
                #     wxyz_list = []
                #     pn = ProjectedNormal(concentration=var_config.concentration)
                #     while len(wxyz_list) < n_points:
                #         unit_quaternions = pn.sample(key, sample_shape=(var_config.n_batch,))
                #         geodesic_distance_in_degrees = jnp.arccos(jnp.abs(unit_quaternions[:, 0])) * (180.0 / jnp.pi) * 2.0
                #         mask = (geodesic_distance_in_degrees >= min_geodesic_angle_in_degrees) & (geodesic_distance_in_degrees <= max_geodesic_angle_in_degrees)
                #         wxyz_list.extend(unit_quaternions[mask].tolist())
                #         key, _ = jax.random.split(key)
                #     return jnp.array(wxyz_list[:n_points])
                # var_config['wxyz_list'] = quaternion_rejection_sampling(
                #     var_config['min_geodesic_angle_in_degrees'],
                #     var_config['max_geodesic_angle_in_degrees'],
                #     var_config['n_points'],
                #     key,
                # )
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
        print(f"cv_dict: {cv_dict}")

        key1, key2 = config.landscape.landscape_variables.keys()
        cv_grid = cv_dict[key1]
        other_grid = cv_dict[key2]

        def eval_sw(cv, other, key):
            wxyz = pose.wxyz
            defocus = config.setup.ctf.defocus_in_angstroms
            offset_x = config.setup.pose.translation.offset_x_in_angstroms
            offset_y = config.setup.pose.translation.offset_y_in_angstroms
            cv = config.setup.cv.value
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
                cv_scalar=cv,
                defocus_scalar=defocus,
                offset_x_in_angstroms=offset_x,
                offset_y_in_angstroms=offset_y,
            )
            return sw_sorted_jax_differentiable(
                config.setup.noise.dose * new_image_model.simulate() + config.setup.noise.background,
                proj_obs_noisy,
                grid,
                n_projections,
                n_pixels,
                sliced_wasserstein.normalize_safely_epsilon,
                key,
            )

        # make a full grid of (cv, other) pairs
        cv_grid = jnp.hstack([cv_grid, cv_grid])

        vmap_size = len(cv_grid) * len(other_grid)
        vmap_shape = (len(cv_grid), len(other_grid))

        # vmap correctly over cv and other
        eval_sw_vmapped = jax.vmap(
            jax.vmap(eval_sw, in_axes=(None, 0, 0)),  # inner: over OTHER axis=0
            in_axes=(0, None, 0)                      # outer: over CV axis=0
        )

        eval_sw_vmapped_jit = jax.jit(eval_sw_vmapped)

        # multiple trials
        key_main = jax.random.PRNGKey(42)
        rep_keys = jax.random.split(key_main, config.landscape.n_trials)

        def run_trial(k):
            ks = jax.random.split(k, vmap_size).reshape(vmap_shape + (2,))
            return eval_sw_vmapped_jit(cv_grid, other_grid, ks)

        sw_landscapes = jax.vmap(run_trial)(rep_keys)
        jnp.save(os.path.join(run_dir, "data/sw_landscape.npy"), sw_landscapes)

if __name__ == "__main__":
    main()