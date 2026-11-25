import pandas as pd
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

from goto_swap.backends.jax.emd import emd_1d_sorted_jax_differentiable
from goto_swap import get_backend
jax_backend = get_backend("jax")

from cryojax.io import read_atoms_from_pdb
import cryojax.simulator as cxs
from cryojax.ndimage.operators import FourierGaussian
from cryojax.rotations import convert_quaternion_to_euler_angles

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

@dataclass
class SpecimenConfig:
    pdb_file: str
    center: bool 
    selection_string: str


@dataclass
class NoiseConfig:
    dose: float
    background: float
    random_seed: int = 0


@dataclass
class ImageConfig:
    grid_size: int
    pixel_size: float
    pad_size: int


@dataclass
class IntegratorConfig:
    atom_width: float
    atom_amplitude: float

@dataclass
class EulerAnglesDegConfig:
    phi_angle: float
    theta_angle: float
    psi_angle: float


@dataclass
class TranslationConfig:
    offset_x_in_angstroms: float
    offset_y_in_angstroms: float


@dataclass
class PoseConfig:
    euler_angles_deg: EulerAnglesDegConfig
    translation: TranslationConfig


@dataclass
class CTFConfig:
    defocus_in_angstroms: float
    voltage: float


@dataclass
class CVConfig:
    type: str
    value: float

    def __post_init__(self):
        if self.type not in ['tear', 'squeeze']:
            raise ValueError("CV type must be either 'tear' or 'squeeze'")


@dataclass
class EnvelopeConfig:
    amplitude: float
    b_factor: float


@dataclass
class SetupConfig:
    specimen: SpecimenConfig
    image: ImageConfig
    noise: NoiseConfig
    integrator: IntegratorConfig
    pose: PoseConfig
    ctf: CTFConfig
    envelope: EnvelopeConfig
    cv: CVConfig


@dataclass
class SlicedWassersteinConfig:
    n_projections: int
    n_pixels: int
    normalize_safely_epsilon: float
    projection_method: str


@dataclass
class PreconditionerConfig:
    cv: float
    defocus: float
    quaternion: float
    offset: float
    noise_dose: float
    noise_background: float


@dataclass
class OptimizeTheseVariablesConfig:
    cv: bool
    defocus: bool
    rotation: bool
    translation: bool
    noise_dose: bool
    noise_background: bool


@dataclass
class OptimizationConfig:
    use_this_objective: str
    learning_rate: float
    n_iterations: int
    sliced_wasserstein: SlicedWassersteinConfig
    print_every: int
    preconditioner: dict[str,PreconditionerConfig]
    optimize_these_variables: OptimizeTheseVariablesConfig

    # post init for use this objective sw or l2
    def __post_init__(self):
        if self.use_this_objective not in ['sw_cdf', 'l2', 'sw', 'sw_dsort', 'sw_icdf']:
            raise ValueError("use_this_objective must be either 'sw', 'sw_cdf', 'l2', 'sw_dsort' or 'sw_icdf'")
        if self.preconditioner:
            for key in self.preconditioner:
                if key not in ['sw_cdf', 'sw', 'l2', 'sw_dsort', 'sw_icdf']:
                    raise ValueError("preconditioner keys must be either 'sw_cdf', 'sw', 'l2', 'sw_dsort' or 'sw_icdf'")
@dataclass
class InitialEstimates:
    cv: CVConfig
    pose: PoseConfig
    ctf: CTFConfig
    integrator: IntegratorConfig


@dataclass
class InferenceModelConfig:
    noise: NoiseConfig
    initial_estimates: InitialEstimates
    integrator: IntegratorConfig


@dataclass
class FullConfig:
    setup: SetupConfig
    optimization: OptimizationConfig
    inference_model: InferenceModelConfig


# def tear_cv(positions: jnp.ndarray, normal: jnp.ndarray, offset: float, tear_amount: float):
#     """
#     Apply a 'tear' collective variable: separate the molecule into two halves
#     along a plane defined by (normal, offset).

#     Args:
#         positions: (N, 3) array of atom positions.
#         normal: (3,) array, unit normal vector defining the splitting plane.
#         offset: float, plane offset so that the plane is {x | n·x = offset}.
#         tear_amount: float, distance to separate top and bottom halves along the normal.

#     Returns:
#         displaced_positions: (N, 3) array after applying the tear CV.
#     """
#     # Compute signed distance of each point from the plane
#     signed_dist = jnp.dot(positions, normal) - offset

#     # Continuous smooth mask: positive side → +1, negative side → -1
#     # Use tanh for differentiability instead of hard sign
#     smooth_sign = jnp.tanh(10.0 * signed_dist)

#     # Displace along the normal vector
#     displacement = (tear_amount / 2.0) * smooth_sign[:, None] * normal[None, :]

#     displaced_positions = positions + displacement
#     return displaced_positions


def separate_cv(positions: jnp.ndarray, normal: jnp.ndarray, tear_amount: float):
    """
    Separate the molecule into two halves by displacing atoms with indices in idx 
    by tear_amount along the axis specified by normal.
    
    Args:
        positions: (N, 3) array of atom positions.
        normal: (3,) unit vector defining the separation axis.
        idx: (M,) array of atom indices to displace (the "half" to move).
        tear_amount: float, distance to displace the selected atoms along normal.
    
    Returns:
        displaced_positions: (N, 3) array after displacing selected atoms.
    """
    # Normalize the normal vector to unit length
    normal = normal / jnp.linalg.norm(normal)
    
    # Create a displacement vector for the selected atoms
    displacement = tear_amount * normal[None, :] / 2  # shape (1, 3)
    
    # Use at/set to displace only atoms at indices in idx
    idx1 = jnp.arange(863)
    idx2 = jnp.arange(863, 1558)
    displaced_positions = positions.at[idx1].add(-displacement)
    displaced_positions = displaced_positions.at[idx2].add(displacement)
    
    return displaced_positions


def apply_all_cv(image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms):
    # new_positions = tear_cv(image_model.volume.positions, normal=jnp.array([1., 1, 0.]), offset=0.0, tear_amount=cv_scalar)
    # n_atoms = image_model.volume.positions.shape[0]
    # bool_mask = jnp.arange(n_atoms) < 863
    # integer_indices = jnp.where(bool_mask)[0]
    print('BEFORE BUG', image_model)
    new_positions = separate_cv(image_model.volume_parametrisation.positions, normal=jnp.array([18.338715,   -9.431992,  -10.780163]), 
                                tear_amount=cv_scalar)
    image_model_updated = eqx.tree_at(lambda im: im.volume_parametrisation.positions, image_model, new_positions)
    image_model_updated = eqx.tree_at(lambda im: im.pose.offset_x_in_angstroms, image_model_updated, offset_x_in_angstroms)
    image_model_updated = eqx.tree_at(lambda im: im.pose.offset_y_in_angstroms, image_model_updated, offset_y_in_angstroms)
    image_model_updated = eqx.tree_at(lambda im: im.pose.wxyz, image_model_updated, wxyz)
    image_model_updated = eqx.tree_at(lambda im: im.scattering_theory.transfer_theory.ctf.defocus_in_angstroms, image_model_updated, defocus_scalar)
    return image_model_updated


def sw_sorted_jax_differentiable(image_1, image_2, grid, n_projections, n_pixels, normalize_safely_epsilon, key):
    
    v_weights = image_1.flatten()
    u_weights = image_2.flatten()

    thetas = jnp.linspace(0, jnp.pi, n_projections, endpoint=False) 
    keys = jax.random.split(key, len(thetas))

    def loss_per_theta(theta, key):
        project = jnp.tensordot(grid, jnp.array([jnp.cos(theta), jnp.sin(theta)]), axes=1).flatten()
        perm = jax.random.permutation(key, len(project))[:n_pixels]
        x_ab = project[perm]
        sort_idx = jnp.argsort(x_ab)
        x_ab_sorted = x_ab[sort_idx]
        a_sorted = u_weights[perm][sort_idx] + normalize_safely_epsilon
        b_sorted = v_weights[perm][sort_idx] + normalize_safely_epsilon
        return emd_1d_sorted_jax_differentiable(x_ab_sorted, x_ab_sorted, a_sorted / a_sorted.sum(), b_sorted / b_sorted.sum())

    losses = jax.vmap(loss_per_theta)(thetas, keys)
    return losses

def apply_background_and_dose(image, dose, background):
    """
    Apply background and dose to the image.

    Args:
        image: (H, W) array representing the image.
        dose: float, the dose factor to scale the image.
        background: float, the background level to add to the image.

    Returns:
        processed_image: (H, W) array after applying background and dose.
    """
    return dose * (image + background)
    

def loss_for_pose_and_cv(image_ref, image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms, use_this_objective, grid, sliced_wasserstein, noise_dose, noise_background, key):
    benchmark_time = False
    if benchmark_time: s = time()
    image_1 = apply_all_cv(image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms).simulate()

    optimal_b_d = False
    if optimal_b_d:
        def optimal_noise_d_b(image_ref, image_1):
            # undo image_1 = dose * (image_1 + background)
            image_1_bar = jnp.mean(image_1)
            image_ref_bar = jnp.mean(image_ref)
            image_ref_centered = image_ref - image_ref_bar
            image_1_centered = image_1 - image_1_bar
            optimal_dose = jnp.sum(image_ref_centered * image_1_centered) / jnp.sum(image_1_centered ** 2)
            optimal_background = image_ref_bar/optimal_dose - image_1_bar
            image_1_optimal = optimal_dose * (image_1 + optimal_background)
            return optimal_dose, optimal_background, image_1_optimal

        optimal_dose, optimal_background, image_1_optimal = optimal_noise_d_b(image_ref, image_1)
        l2_loss_optimal = jnp.linalg.norm(image_1_optimal - image_ref)
        l2_loss_unoptimal = jnp.linalg.norm(noise_dose * (image_1 + noise_background) - image_ref)
        jax.debug.print(
            "dose: {noise_dose:.4f}, background: {noise_background:.4f}, "
            "optimal_dose: {optimal_dose:.4f}, "
            "optimal_background: {optimal_background:.4f}, "
            "l2_loss_optimal: {l2_loss_optimal:.4f}, "
            "l2_loss_unoptimal: {l2_loss_unoptimal:.4f}",
            noise_dose=noise_dose,
            noise_background=noise_background,
            optimal_dose=optimal_dose,
            optimal_background=optimal_background,
            l2_loss_optimal=l2_loss_optimal,
            l2_loss_unoptimal=l2_loss_unoptimal,
        )    
    if benchmark_time: 
        simulate_time = time() - s
        logger.debug(f"simulate_time: {simulate_time:.4f} seconds")
    n_pix = image_ref.shape[-1]
    
    def l2_loss(image_1, image_2):
        # jax.debug.print("std: {s1}, {s2}", 
        #                 s1=jnp.std(image_1), 
        #                 s2=jnp.std(image_2))
        # jax.debug.print("max: {m1}, {m2}", 
        #                 m1=jnp.max(image_1), 
        #                 m2=jnp.max(image_2))
        # jax.debug.print("min: {m1}, {m2}", 
        #                 m1=jnp.min(image_1), 
        #                 m2=jnp.min(image_2))
        # # mean
        # jax.debug.print("mean: {m1}, {m2}", 
        #                 m1=jnp.mean(image_1), 
        #                 m2=jnp.mean(image_2))
        return jnp.linalg.norm(image_1 - image_2)

    projection_method = sliced_wasserstein.projection_method
    assert projection_method in ['fourier_slice', 'real_rotate'], f"projection_method must be either 'fourier_slice' or 'real_rotate', got {projection_method}"
    if use_this_objective == 'sw':
        if benchmark_time: s = time()

        result = sw_sorted_jax_differentiable(apply_background_and_dose(image_1, noise_dose, noise_background), image_ref, grid, sliced_wasserstein.n_projections, sliced_wasserstein.n_pixels, sliced_wasserstein.normalize_safely_epsilon, key).mean()
        if benchmark_time:
            sw_time = time() - s
            logger.debug(f"sw_time: {sw_time:.4f} seconds")

        return result
    elif use_this_objective == 'l2':
        # import numpy as np
        # plt.imshow(np.array(image_1), cmap='gray')
        # plt.colorbar()
        # plt.savefig("debug_image_1.png")
        # plt.clf()
        # plt.imshow(np.array(image_ref), cmap='gray')
        # plt.colorbar()
        # plt.savefig("debug_image_ref.png")
        # plt.clf()
        if benchmark_time: s = time()
        result = l2_loss(apply_background_and_dose(image_1, noise_dose, noise_background), image_ref) # normalize by dose to not affect learning rate
        if benchmark_time: 
            l2_time = time() - s
            logger.debug(f"l2_time: {l2_time:.4f} seconds")
        return result
    elif use_this_objective == 'sw_cdf':
        if benchmark_time: s = time()
        result = jax_backend.wasserstein_distance_loss_images(apply_background_and_dose(image_1, noise_dose, noise_background).reshape(1, n_pix, n_pix), 
                                                     image_ref.reshape(1, n_pix, n_pix),
                                                     num_rotations=sliced_wasserstein.n_projections,
                                                    projection_method=projection_method,
                                                     method='cdf'
                                                     ).mean()
        if benchmark_time: 
            cdf_time = time() - s
            logger.debug(f"cdf_time: {cdf_time:.4f} seconds")
        return result
    elif use_this_objective == 'sw_dsort':
        if benchmark_time: s = time()
        result = jax_backend.wasserstein_distance_loss_images(apply_background_and_dose(image_1, noise_dose, noise_background).reshape(1, n_pix, n_pix), 
                                                     image_ref.reshape(1, n_pix, n_pix),
                                                     num_rotations=sliced_wasserstein.n_projections,
                                                     projection_method=projection_method,
                                                     method='dsort'
                                                     ).mean()
        if benchmark_time: 
            dsort_time = time() - s
            logger.debug(f"dsort_time: {dsort_time:.4f} seconds")
        return result
    elif use_this_objective == 'sw_icdf':
        if benchmark_time: s = time()
        result = jax_backend.wasserstein_distance_loss_images(apply_background_and_dose(image_1, noise_dose, noise_background).reshape(1, n_pix, n_pix), 
                                                     image_ref.reshape(1, n_pix, n_pix),
                                                     num_rotations=sliced_wasserstein.n_projections,
                                                     projection_method=projection_method,
                                                     method='icdf'
                                                     ).mean()
        if benchmark_time: 
            icdf_time = time() - s
            logger.debug(f"icdf_time: {icdf_time:.4f} seconds")
        return result
    else:
        raise ValueError(f"use_this_objective not implemented for {use_this_objective}")


def unnormalized_quaternion_to_geodesic(wxyz_raw):
    wxyz_raw /= jnp.linalg.norm(wxyz_raw)
    return jnp.rad2deg(2*jnp.arccos(wxyz_raw[0]))


@hydra.main(version_base=None, config_path="./", config_name="config_joint_gradient_descent")
def main(cfg: FullConfig):
    plt.style.use("seaborn-v0_8-poster")

    # logger.info(OmegaConf.to_yaml(cfg))
    config = OmegaConf.merge(OmegaConf.structured(FullConfig), cfg)
    
    run_dir = HydraConfig.get().run.dir 
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    logger.info(f"Saving traces to {run_dir}")
    logger.info(OmegaConf.to_yaml(OmegaConf.to_container(config)))

    def setup_specimen(specimen: SpecimenConfig):
        atom_positions, _ = read_atoms_from_pdb(
            specimen.pdb_file, 
            center=specimen.center, 
            selection_string=specimen.selection_string,
            # loads_properties=False
        )
        return atom_positions
    atom_positions = setup_specimen(config.setup.specimen)
    # print(f"Loaded {len(atom_positions)} atoms from {config.setup.specimen.pdb_file} with selection '{config.setup.specimen.selection_string}'")

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
    # print(image_model)
    
    logger.info(f"image_model: {image_model}")
    jnp.save(os.path.join(run_dir, "data/initial_image_model.npy"), image_model)
    image_model_obs = apply_all_cv(
        image_model=image_model,
        wxyz=pose.wxyz,
        offset_x_in_angstroms=config.setup.pose.translation.offset_x_in_angstroms,
        offset_y_in_angstroms=config.setup.pose.translation.offset_y_in_angstroms,
        cv_scalar=config.setup.cv.value,
        defocus_scalar=config.setup.ctf.defocus_in_angstroms,
    )

    proj_obs = image_model_obs.simulate()
    proj_obs_noisy = jax.random.poisson(jax.random.PRNGKey(config.setup.noise.random_seed),  config.setup.noise.dose*(proj_obs + config.setup.noise.background))

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
    # assert False

    @dataclass
    class OptParams(eqx.Module):
        cv: jnp.ndarray
        defocus: jnp.ndarray
        wxyz: jnp.ndarray
        offset_x_in_angstroms: jnp.ndarray
        offset_y_in_angstroms: jnp.ndarray
        noise_dose: jnp.ndarray
        noise_background: jnp.ndarray

    def loss_for_params(params: OptParams, key):
        return loss_for_pose_and_cv(
            image_ref=proj_obs_noisy,
            image_model=image_model,
            cv_scalar=params.cv,
            defocus_scalar=params.defocus,
            wxyz=params.wxyz,
            offset_x_in_angstroms=params.offset_x_in_angstroms,
            offset_y_in_angstroms=params.offset_y_in_angstroms,
            use_this_objective=config.optimization.use_this_objective,
            grid=grid,
            sliced_wasserstein=config.optimization.sliced_wasserstein,
            noise_dose=params.noise_dose,
            noise_background=params.noise_background,
            key=key,
        )

    value_and_grad_for_params = eqx.filter_value_and_grad(loss_for_params)  
    value_and_grad_for_params_jit = eqx.filter_jit(value_and_grad_for_params)

    key = jax.random.PRNGKey(0)
    cv = jnp.array(config.inference_model.initial_estimates.cv.value)
    defocus = jnp.array(config.inference_model.initial_estimates.ctf.defocus_in_angstroms)
    wxyz = cxs.EulerAnglePose(
        phi_angle=config.inference_model.initial_estimates.pose.euler_angles_deg.phi_angle,
        theta_angle=config.inference_model.initial_estimates.pose.euler_angles_deg.theta_angle,
        psi_angle=config.inference_model.initial_estimates.pose.euler_angles_deg.psi_angle,
    ).rotation.wxyz
    offset_x_in_angstroms = jnp.array(config.inference_model.initial_estimates.pose.translation.offset_x_in_angstroms)
    offset_y_in_angstroms = jnp.array(config.inference_model.initial_estimates.pose.translation.offset_y_in_angstroms)
    noise_dose = jnp.array(config.inference_model.noise.dose)
    noise_background = jnp.array(config.inference_model.noise.background)

    params0 = OptParams(
        cv=cv,
        defocus=defocus,
        wxyz=wxyz,
        offset_x_in_angstroms=offset_x_in_angstroms,
        offset_y_in_angstroms=offset_y_in_angstroms,
        noise_dose=noise_dose,
        noise_background=noise_background,
    )

    def optimization(optimization_cfg, key, initial_params: OptParams):
        lr = optimization_cfg.learning_rate
        n_iters = optimization_cfg.n_iterations

        losses = []
        cvs = []
        defoci = []
        wxyzs = []
        offset_xs = []
        offset_ys = []
        geodesic_angles = [] # from unit quaternion
        noise_doses = []
        noise_backgrounds = []

        params = initial_params
        preconditioner = optimization_cfg.preconditioner[optimization_cfg.use_this_objective]


        total_time = 0
        for i in range(n_iters):
            key, subkey = jax.random.split(key)
            s = time()
            loss_val, grads = value_and_grad_for_params_jit(params, subkey)
            iter_time = time() - s
            total_time += iter_time
            logger.debug(f"value_and_grad_for_params_jit time: {iter_time:.4f} seconds")

            losses.append(float(loss_val))
            cvs.append(float(params.cv))
            defoci.append(float(params.defocus))
            wxyzs.append(jnp.array(params.wxyz))  # store array form for quaternion
            offset_xs.append(float(params.offset_x_in_angstroms))
            offset_ys.append(float(params.offset_y_in_angstroms))
            geodesic_angles.append(float(unnormalized_quaternion_to_geodesic(params.wxyz)))
            noise_doses.append(float(params.noise_dose))
            noise_backgrounds.append(float(params.noise_background))

            # SGD-like update (apply preconditioning factors from config)
            if optimization_cfg.optimize_these_variables.cv:
                cv_new = params.cv - lr * grads.cv * preconditioner.cv
            else:
                cv_new = params.cv
            if optimization_cfg.optimize_these_variables.defocus:
                defocus_new = params.defocus - lr * grads.defocus * preconditioner.defocus
            else:
                defocus_new = params.defocus
            if optimization_cfg.optimize_these_variables.rotation:
                quaternion_normalize_safely_epsilon = 1e-12
                wxyz_new = params.wxyz - lr * grads.wxyz * preconditioner.quaternion + quaternion_normalize_safely_epsilon
                wxyz_new = wxyz_new / (jnp.linalg.norm(wxyz_new) )
            else:
                wxyz_new = params.wxyz
            if optimization_cfg.optimize_these_variables.translation:
                offset_x_new = params.offset_x_in_angstroms - lr * grads.offset_x_in_angstroms * preconditioner.offset
                offset_y_new = params.offset_y_in_angstroms - lr * grads.offset_y_in_angstroms * preconditioner.offset
            else:
                offset_x_new = params.offset_x_in_angstroms
                offset_y_new = params.offset_y_in_angstroms
            if optimization_cfg.optimize_these_variables.noise_dose:
                noise_dose_new = params.noise_dose - lr * grads.noise_dose * preconditioner.noise_dose
                noise_dose_new = jnp.clip(noise_dose_new, a_min=1e-6)  # Ensure noise dose remains positive
            else:
                noise_dose_new = params.noise_dose
            if optimization_cfg.optimize_these_variables.noise_background:
                noise_background_new = params.noise_background - lr * grads.noise_background * preconditioner.noise_background
                noise_background_new = jnp.clip(noise_background_new, a_min=0.0)  # Ensure noise background remains non-negative
            else:
                noise_background_new = params.noise_background
            params = OptParams(
                cv=cv_new,
                defocus=defocus_new,
                wxyz=wxyz_new,
                offset_x_in_angstroms=offset_x_new,
                offset_y_in_angstroms=offset_y_new,
                noise_dose=noise_dose_new,
                noise_background=noise_background_new,
            )

            if i % optimization_cfg.print_every == 0:
                euler_angles_deg = convert_quaternion_to_euler_angles(params.wxyz)
                logger.info(
                    f"Step {i:03d}: loss={loss_val:.6e}, cv={float(params.cv):.6f}, defocus={float(params.defocus):.6f}, "
                    f"offset_x={float(params.offset_x_in_angstroms):.6f}, offset_y={float(params.offset_y_in_angstroms):.6f}, "
                    f"wxyz=({float(params.wxyz[0]):.6f}, {float(params.wxyz[1]):.6f}, {float(params.wxyz[2]):.6f}, {float(params.wxyz[3]):.6f}), "
                    f"geodesic_angle={geodesic_angles[-1]:.3f} degrees, "
                    f"euler_angles_deg=({euler_angles_deg[0]:.2f}, {euler_angles_deg[1]:.2f}, {euler_angles_deg[2]:.2f}), "
                    f"noise_dose={float(params.noise_dose):.6f}, noise_background={float(params.noise_background):.6f}, "
                    f"grad_cv={float(grads.cv):.2e}, "
                    f"grad_defocus={float(grads.defocus):.2e}, "
                    f"grad_offset_x={float(grads.offset_x_in_angstroms):.2e}, "
                    f"grad_offset_y={float(grads.offset_y_in_angstroms):.2e}, "
                    f"grad_wxyz=({float(grads.wxyz[0]):.2e}, {float(grads.wxyz[1]):.2e}, {float(grads.wxyz[2]):.2e}, {float(grads.wxyz[3]):.2e}), "
                    f"grad_noise_dose={float(grads.noise_dose):.2e}, "
                    f"grad_noise_background={float(grads.noise_background):.2e}",
                )
        logger.info(f"Average time per iteration for {optimization_cfg.use_this_objective}: {total_time / n_iters:.4f} seconds")

        return (
            jnp.array(losses),
            jnp.array(cvs),
            jnp.array(defoci),  
            jnp.stack(wxyzs),
            jnp.array(geodesic_angles),
            jnp.array(offset_xs),
            jnp.array(offset_ys),
            jnp.array(noise_doses),
            jnp.array(noise_backgrounds),
        )

    start = time()
    losses, cvs, defoci, wxyzs, geodesic_angles, offset_xs, offset_ys, noise_doses, noise_backgrounds = optimization(config.optimization, key, params0)
    end = time()
    logger.info(f"Optimization took {end - start:.2f} seconds.")
    logger.info(f'Output in {run_dir}')

    # images with initial and final parameters
    for idx, label in zip([0, -1], ['initial', 'final']):
        image_model_i = apply_all_cv(
            image_model=image_model,
            wxyz=wxyzs[idx],
            offset_x_in_angstroms=offset_xs[idx],
            offset_y_in_angstroms=offset_ys[idx],
            cv_scalar=cvs[idx],
            defocus_scalar=defoci[idx],
        )
        proj_i = image_model_i.simulate()
        plt.imshow(proj_i, cmap='gray')
        plt.colorbar()
        plt.savefig(os.path.join(run_dir, f"plots/proj_{label}.png"))
        plt.clf()
        jnp.save(os.path.join(run_dir, f"data/proj_{label}.npy"), proj_i)
    # benchmark forward model evalualtion
    @eqx.filter_jit
    def forward_model_trial():
        return apply_all_cv(
            image_model=image_model,
            wxyz=wxyzs[-1],
            offset_x_in_angstroms=offset_xs[-1],
            offset_y_in_angstroms=offset_ys[-1],
            cv_scalar=cvs[-1],
            defocus_scalar=defoci[-1],
        ).simulate()
    n_trials_forward = 100
    times = []
    for _ in range(n_trials_forward + 1):
        s = time()
        _ = forward_model_trial()
        times.append(time() - s)
    logger.info(f"Forward model evaluation: Averaged over {n_trials_forward} trials. Average time: {sum(times[1:]) / n_trials_forward:.4f} seconds")

    jnp.savez(os.path.join(run_dir, "data/joint_gradient_descent_traces.npz"),
             losses=losses,
             cvs=cvs,
             defoci=defoci,
             wxyzs=wxyzs,
             geodesic_angles=geodesic_angles,
             offset_xs=offset_xs,
             offset_ys=offset_ys,
             noise_doses=noise_doses,
             noise_backgrounds=noise_backgrounds
    )
    euler_angles = jnp.array([convert_quaternion_to_euler_angles(w) for w in wxyzs])
    pd.DataFrame({
        "losses": losses,
        "cvs": cvs,
        "defoci": defoci,
        "wxyz_0": wxyzs[:,0],
        "wxyz_1": wxyzs[:,1],
        "wxyz_2": wxyzs[:,2],
        "wxyz_3": wxyzs[:,3],
        'euler_angle_0': euler_angles[:,0],
        'euler_angle_1': euler_angles[:,1],
        'euler_angle_2': euler_angles[:,2],
        "geodesic_angles": geodesic_angles,
        "offset_xs": offset_xs,
        "offset_ys": offset_ys,
        "noise_doses": noise_doses,
        "noise_backgrounds": noise_backgrounds
    }).to_csv(os.path.join(run_dir, "data/joint_gradient_descent_traces.csv"), index=False)
    label = config.optimization.use_this_objective
    plt.plot(losses, label=label)
    plt.xlabel("Gradient step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/loss_curve.png"))
    plt.clf()
    plt.plot(cvs, label=label)
    plt.axhline(y=config.setup.cv.value, color='gray', linestyle='--', label='Ground Truth')
    plt.xlabel("Gradient step")
    plt.ylabel("CV")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/cv_curve.png"))
    plt.clf()
    plt.plot(defoci, label=label)
    plt.xlabel("Gradient step")
    plt.ylabel("Defocus (Angstroms)")
    plt.axhline(y=config.setup.ctf.defocus_in_angstroms, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/defocus_curve.png"))
    plt.clf()
    plt.plot(wxyzs, label=label)
    plt.xlabel("Gradient step")
    plt.ylabel("Rotation (quaternion components)")
    for quaternion_component in setup_pose_wxyz:
        plt.axhline(y=quaternion_component, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/wxyz_curve.png"))
    plt.clf()
    plt.plot(geodesic_angles, label=label)
    plt.xlabel("Gradient step")
    plt.ylabel("Geodesic Angle (degrees)")
    plt.axhline(y=0, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/geodesic_angle_curve.png"))
    plt.clf()
    plt.plot(offset_xs, label=label)
    plt.xlabel("Gradient step")
    plt.ylabel("Offset X (Angstroms)")
    plt.axhline(y=config.setup.pose.translation.offset_x_in_angstroms, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/offset_x_curve.png"))
    plt.clf()
    plt.plot(offset_ys, label=label)
    plt.xlabel("Gradient step")
    plt.ylabel("Offset Y (Angstroms)")
    plt.axhline(y=config.setup.pose.translation.offset_y_in_angstroms, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/offset_y_curve.png"))
    plt.clf()

    plt.plot(noise_doses, label=label)
    plt.axhline(
        y=config.setup.noise.dose,
        color='gray',
        linestyle='--',
        label='Ground Truth',
    )
    plt.xlabel("Gradient step")
    plt.ylabel("Noise dose")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/noise_dose_curve.png"))
    plt.clf()

    plt.plot(noise_backgrounds, label=label)
    plt.axhline(
        y=config.setup.noise.background,
        color='gray',
        linestyle='--',
        label='Ground Truth',
    )
    plt.xlabel("Gradient step")
    plt.ylabel("Noise background")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/noise_background_curve.png"))
    plt.clf()

    
    return None

if __name__ == "__main__":
    main()