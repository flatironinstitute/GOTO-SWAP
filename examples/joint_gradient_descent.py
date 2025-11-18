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
from cryojax.io import read_atoms_from_pdb
import cryojax.simulator as cxs
from cryojax.ndimage.operators import FourierGaussian


@dataclass
class SpecimenConfig:
    pdb_file: str
    center: bool 
    selection_string: str


@dataclass
class NoiseConfig:
    dose: float
    background: float


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


@dataclass
class PreconditionerConfig:
    cv: float
    defocus: float
    quaternion: float
    offset: float


@dataclass
class OptimizeTheseVariablesConfig:
    cv: bool
    defocus: bool
    rotation: bool
    translation: bool


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
        if self.use_this_objective not in ['sliced_wasserstein', 'l2']:
            raise ValueError("use_this_objective must be either 'sliced_wasserstein' or 'l2'")
        if self.preconditioner:
            for key in self.preconditioner:
                if key not in ['sliced_wasserstein', 'l2']:
                    raise ValueError("preconditioner keys must be either 'sliced_wasserstein' or 'l2'")

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


def tear_cv(positions: jnp.ndarray, normal: jnp.ndarray, offset: float, tear_amount: float):
    """
    Apply a 'tear' collective variable: separate the molecule into two halves
    along a plane defined by (normal, offset).

    Args:
        positions: (N, 3) array of atom positions.
        normal: (3,) array, unit normal vector defining the splitting plane.
        offset: float, plane offset so that the plane is {x | n·x = offset}.
        tear_amount: float, distance to separate top and bottom halves along the normal.

    Returns:
        displaced_positions: (N, 3) array after applying the tear CV.
    """
    # Compute signed distance of each point from the plane
    signed_dist = jnp.dot(positions, normal) - offset

    # Continuous smooth mask: positive side → +1, negative side → -1
    # Use tanh for differentiability instead of hard sign
    smooth_sign = jnp.tanh(10.0 * signed_dist)

    # Displace along the normal vector
    displacement = (tear_amount / 2.0) * smooth_sign[:, None] * normal[None, :]

    displaced_positions = positions + displacement
    return displaced_positions


def apply_all_cv(image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms):
    new_positions = tear_cv(image_model.volume_parametrization.positions, normal=jnp.array([1., 1, 0.]), offset=0.0, tear_amount=cv_scalar)
    image_model_updated = eqx.tree_at(lambda im: im.volume_parametrization.positions, image_model, new_positions)
    image_model_updated = eqx.tree_at(lambda im: im.pose.offset_in_angstroms, image_model_updated, jnp.array([offset_x_in_angstroms, offset_y_in_angstroms]))
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
    

def loss_for_pose_and_cv(image_ref, image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms, use_this_objective, grid, sliced_wasserstein, noise, key):
    image_1 = apply_all_cv(image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms).simulate()
    
    def l2_loss(image_1, image_2):
        return jnp.linalg.norm(image_1 - image_2)

    if use_this_objective == 'sw':
        return sw_sorted_jax_differentiable(noise.dose*(image_1 + noise.background), image_ref, grid, sliced_wasserstein.n_projections, sliced_wasserstein.n_pixels, sliced_wasserstein.normalize_safely_epsilon, key).mean()
    elif use_this_objective == 'l2':
        return l2_loss(noise.dose*(image_1 + noise.background)/noise.dose, image_ref/noise.dose) # normalize by dose to not affect learning rate


def unnormalized_quaternion_to_geodesic(wxyz_raw):
    wxyz_raw /= jnp.linalg.norm(wxyz_raw)
    return jnp.rad2deg(2*jnp.arccos(wxyz_raw[0]))


@hydra.main(version_base=None, config_path="./", config_name="config_joint_gradient_descent")
def main(cfg: FullConfig):

    logger = logging.getLogger(__name__)
    # logger.info(OmegaConf.to_yaml(cfg))
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
    jnp.savez(os.path.join(run_dir, "data/initial_image_model.npz"), image_model=image_model, )
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


    @dataclass
    class OptParams(eqx.Module):
        cv: jnp.ndarray
        defocus: jnp.ndarray
        wxyz: jnp.ndarray
        offset_x_in_angstroms: jnp.ndarray
        offset_y_in_angstroms: jnp.ndarray

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
            noise=config.inference_model.noise,
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

    params0 = OptParams(
        cv=cv,
        defocus=defocus,
        wxyz=wxyz,
        offset_x_in_angstroms=offset_x_in_angstroms,
        offset_y_in_angstroms=offset_y_in_angstroms,
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

        params = initial_params
        preconditioner = optimization_cfg.preconditioner[optimization_cfg.use_this_objective]

        for i in range(n_iters):
            key, subkey = jax.random.split(key)
            loss_val, grads = value_and_grad_for_params_jit(params, subkey)

            losses.append(float(loss_val))
            cvs.append(float(params.cv))
            defoci.append(float(params.defocus))
            wxyzs.append(jnp.array(params.wxyz))  # store array form for quaternion
            offset_xs.append(float(params.offset_x_in_angstroms))
            offset_ys.append(float(params.offset_y_in_angstroms))
            geodesic_angles.append(float(unnormalized_quaternion_to_geodesic(params.wxyz)))

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
            params = OptParams(
                cv=cv_new,
                defocus=defocus_new,
                wxyz=wxyz_new,
                offset_x_in_angstroms=offset_x_new,
                offset_y_in_angstroms=offset_y_new,
            )

            if i % optimization_cfg.print_every == 0:
                logger.info(
                    f"Step {i:03d}: loss={loss_val:.6e}, cv={float(params.cv):.6f}, defocus={float(params.defocus):.6f}, "
                    f"offset_x={float(params.offset_x_in_angstroms):.6f}, offset_y={float(params.offset_y_in_angstroms):.6f}, "
                    f"wxyz=({float(params.wxyz[0]):.6f}, {float(params.wxyz[1]):.6f}, {float(params.wxyz[2]):.6f}, {float(params.wxyz[3]):.6f}), "
                    f"geodesic_angle={geodesic_angles[-1]:.3f} degrees, "
                    f"grad_cv={float(grads.cv):.2e}, "
                    f"grad_defocus={float(grads.defocus):.2e}, "
                    f"grad_offset_x={float(grads.offset_x_in_angstroms):.2e}, "
                    f"grad_offset_y={float(grads.offset_y_in_angstroms):.2e}, "
                    f"grad_wxyz=({float(grads.wxyz[0]):.2e}, {float(grads.wxyz[1]):.2e}, {float(grads.wxyz[2]):.2e}, {float(grads.wxyz[3]):.2e})"
                )

        return (
            jnp.array(losses),
            jnp.array(cvs),
            jnp.array(defoci),  
            jnp.stack(wxyzs),
            jnp.array(geodesic_angles),
            jnp.array(offset_xs),
            jnp.array(offset_ys),
        )

    start = time()
    losses, cvs, defoci, wxyzs, geodesic_angles, offset_xs, offset_ys = optimization(config.optimization, key, params0)
    end = time()
    logger.info(f"Optimization took {end - start:.2f} seconds.")
    logger.info(f'Output in {run_dir}')

    jnp.savez(os.path.join(run_dir, "data/joint_gradient_descent_traces.npz"),
             losses=losses,
             cvs=cvs,
             defoci=defoci,
             wxyzs=wxyzs,
             geodesic_angles=geodesic_angles,
             offset_xs=offset_xs,
             offset_ys=offset_ys,
    )

    plt.plot(losses, label="Loss")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/loss_curve.png"))
    plt.clf()
    plt.plot(cvs, label="CV")
    plt.axhline(y=config.setup.cv.value, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/cv_curve.png"))
    plt.clf()
    plt.plot(defoci, label="Defocus")
    plt.axhline(y=config.setup.ctf.defocus_in_angstroms, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/defocus_curve.png"))
    plt.clf()
    plt.plot(wxyzs, label="WXYZ")
    for quaternion_component in setup_pose_wxyz:
        plt.axhline(y=quaternion_component, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/wxyz_curve.png"))
    plt.clf()
    plt.plot(geodesic_angles, label="Geodesic Angle (degrees)")
    plt.axhline(y=0, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/geodesic_angle_curve.png"))
    plt.clf()
    plt.plot(offset_xs, label="Offset X")
    plt.axhline(y=config.setup.pose.translation.offset_x_in_angstroms, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/offset_x_curve.png"))
    plt.clf()
    plt.plot(offset_ys, label="Offset Y")
    plt.axhline(y=config.setup.pose.translation.offset_y_in_angstroms, color='gray', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(run_dir, "plots/offset_y_curve.png"))
    plt.clf()
    
    return None

if __name__ == "__main__":
    main()