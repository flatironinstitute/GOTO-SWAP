import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass
from dataclasses import field
from typing import List
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from goto_swap.backends.jax.emd import emd_1d_sorted_jax_differentiable
from cryojax.io import read_atoms_from_pdb
import cryojax.simulator as cxs
from cryojax.ndimage.operators import FourierGaussian
from cryojax.simulator import WeakPhaseScatteringTheory
from cryojax.ndimage import irfftn

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
    defocus_to_cv_ratio: float
    quaternion_to_cv_ratio: float
    offset_to_cv_ratio: float

@dataclass
class OptimizationConfig:
    learning_rate: float
    n_iterations: int
    sliced_wasserstein: SlicedWassersteinConfig
    print_every: int
    preconditioner: PreconditionerConfig


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


def apply_rotation_and_cv(image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms):
    new_positions = tear_cv(image_model.volume_parametrization.positions, normal=jnp.array([1., 1, 0.]), offset=0.0, tear_amount=cv_scalar)
    image_model_updated = eqx.tree_at(lambda im: im.volume_parametrization.positions, image_model, new_positions)
    image_model_updated = eqx.tree_at(lambda im: im.pose.offset_in_angstroms, image_model_updated, jnp.array([offset_x_in_angstroms, offset_y_in_angstroms]))
    image_model_updated = eqx.tree_at(lambda im: im.pose.wxyz, image_model_updated, wxyz)
    image_model_updated = eqx.tree_at(lambda im: im.scattering_theory.transfer_theory.ctf.defocus_in_angstroms, image_model_updated, defocus_scalar)
    image_ref = image_model_updated.simulate()
    return image_ref


def loss_for_pose_and_cv(image_ref, image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms, grid, sliced_wasserstein, key):
    image_1 = apply_rotation_and_cv(image_model, wxyz, cv_scalar, defocus_scalar, offset_x_in_angstroms, offset_y_in_angstroms) 
    

    def wasserstein_random_projections(image_1, image_2, key):
        return jnp.linalg.norm(image_1 - image_2)
        
        # v_weights = image_1.flatten()
        # u_weights = image_2.flatten()

        # thetas = jnp.linspace(0, jnp.pi, sliced_wasserstein.n_projections, endpoint=False) 
        # keys = jax.random.split(key, len(thetas))

        # def loss_per_theta(theta, key):
        #     project = jnp.tensordot(grid, jnp.array([jnp.cos(theta), jnp.sin(theta)]), axes=1).flatten()
        #     perm = jax.random.permutation(key, len(project))[:sliced_wasserstein.n_pixels]
        #     x_ab = project[perm]
        #     sort_idx = jnp.argsort(x_ab)
        #     x_ab_sorted = x_ab[sort_idx]
        #     a_sorted = u_weights[perm][sort_idx] + sliced_wasserstein.normalize_safely_epsilon
        #     b_sorted = v_weights[perm][sort_idx] + sliced_wasserstein.normalize_safely_epsilon
        #     return emd_1d_sorted_jax_differentiable(x_ab_sorted, x_ab_sorted, a_sorted / a_sorted.sum(), b_sorted / b_sorted.sum())

        # losses = jax.vmap(loss_per_theta)(thetas, keys)
        # return losses.mean()
    return wasserstein_random_projections(image_1, image_ref, key)

@hydra.main(version_base=None, config_path="./", config_name="config_joint_gradient_descent")
def main(cfg: FullConfig):
    print(OmegaConf.to_yaml(cfg))
    config = OmegaConf.merge(OmegaConf.structured(FullConfig), cfg)

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
                variances=config.setup.integrator.atom_width**2,
            )
    # gaussian_integrator = cxs.GaussianMixtureProjection()
    pose = cxs.QuaternionPose(wxyz=cxs.EulerAnglePose(
        phi_angle=config.setup.pose.euler_angles_deg.phi_angle,
        theta_angle=config.setup.pose.euler_angles_deg.theta_angle,
        psi_angle=config.setup.pose.euler_angles_deg.psi_angle,
        ).rotation.wxyz, 
        offset_x_in_angstroms=config.setup.pose.translation.offset_x_in_angstroms, 
        offset_y_in_angstroms=config.setup.pose.translation.offset_y_in_angstroms
        )
    image_model = cxs.make_image_model(
        volume,
        image_config,
        pose=pose,
        transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF(defocus_in_angstroms=config.setup.ctf.defocus_in_angstroms)),
        simulates_quantity=True,
        quantity_mode='intensity'
    )
    image_ref = image_model.simulate()


    # envelope = FourierGaussian(amplitude=config.setup.envelope.amplitude, 
    #                            b_factor=config.setup.envelope.b_factor)
    # transfer_theory = cxs.ContrastTransferTheory(
    #         cxs.AstigmaticCTF(defocus_in_angstroms=config.setup.ctf.defocus_in_angstroms),
    #         envelope=envelope,
    #     )
    # scattering_theory = WeakPhaseScatteringTheory(gaussian_integrator, transfer_theory)
    # image_ref = irfftn(scattering_theory.compute_intensity_spectrum(volume, image_config))


    plt.imshow(image_ref, cmap='gray')
    plt.show()

    
    xm, ym = jnp.meshgrid(
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        indexing='ij',
    )
    grid = jnp.stack([xm, ym], axis=-1)
    proj_obs = apply_rotation_and_cv(
        image_model=image_model,
        wxyz=pose.wxyz,
        offset_x_in_angstroms=config.setup.pose.translation.offset_x_in_angstroms,
        offset_y_in_angstroms=config.setup.pose.translation.offset_y_in_angstroms,
        cv_scalar=config.setup.cv.value,
        defocus_scalar=config.setup.ctf.defocus_in_angstroms,
    )

    # Pack differentiable variables into a dataclass so equinox can return
    # a matching gradient structure (vectorized gradient).
    # Make OptParams an Equinox pytree (dataclass subclassing eqx.Module)
    @dataclass
    class OptParams(eqx.Module):
        cv: jnp.ndarray
        defocus: jnp.ndarray
        wxyz: jnp.ndarray
        offset_x_in_angstroms: jnp.ndarray
        offset_y_in_angstroms: jnp.ndarray

    def loss_for_params(params: OptParams, key):
        return loss_for_pose_and_cv(
            image_ref=proj_obs,
            image_model=image_model,
            cv_scalar=params.cv,
            defocus_scalar=params.defocus,
            wxyz=params.wxyz,
            offset_x_in_angstroms=params.offset_x_in_angstroms,
            offset_y_in_angstroms=params.offset_y_in_angstroms,
            grid=grid,
            sliced_wasserstein=config.optimization.sliced_wasserstein,
            key=key,
        )

    # JIT the gradient function: use jax.jit on the Equinox-filtered grad to
    # avoid equinox's static-arg hashing issues with non-hashable dataclass
    # instances.
    grad_loss_for_params = eqx.filter_jit(eqx.filter_grad(loss_for_params))

    # --- Example gradient descent loop ---
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

    # initial params packed as a dataclass
    params0 = OptParams(
        cv=cv,
        defocus=defocus,
        wxyz=wxyz,
        offset_x_in_angstroms=offset_x_in_angstroms,
        offset_y_in_angstroms=offset_y_in_angstroms,
    )

    def optimization(optimization, key, initial_params: OptParams):
        lr = optimization.learning_rate
        losses = []
        cvs = []
        defoci = []
        wxyzs = []
        offset_xs = []
        offset_ys = []
        params = initial_params
        for i in range(optimization.n_iterations):
            key, subkey = jax.random.split(key)
            # compute loss and gradients w.r.t. the packed params
            loss_val = loss_for_params(params, subkey)
            grads = grad_loss_for_params(params, subkey)

            losses.append(loss_val)
            cvs.append(params.cv)
            defoci.append(params.defocus)
            wxyzs.append(params.wxyz)
            offset_xs.append(params.offset_x_in_angstroms)
            offset_ys.append(params.offset_y_in_angstroms)

            # apply simple SGD updates (add preconditioning/scaling as needed)
            cv_new = params.cv - lr * grads.cv
            defocus_new = params.defocus - lr * grads.defocus * optimization.preconditioner.defocus_to_cv_ratio
            wxyz_new = params.wxyz - lr * grads.wxyz * optimization.preconditioner.quaternion_to_cv_ratio
            wxyz_new = wxyz_new / (jnp.linalg.norm(wxyz_new) + 1e-12)
            offset_x_new = params.offset_x_in_angstroms - lr * grads.offset_x_in_angstroms * optimization.preconditioner.offset_to_cv_ratio
            offset_y_new = params.offset_y_in_angstroms - lr * grads.offset_y_in_angstroms * optimization.preconditioner.offset_to_cv_ratio

            params = OptParams(
                cv=cv_new,
                defocus=defocus_new,
                wxyz=wxyz_new,
                offset_x_in_angstroms=offset_x_new,
                offset_y_in_angstroms=offset_y_new,
            )

            if i % optimization.print_every == 0:
                print(
                    f"Step {i:02d}: loss={float(loss_val):.6e}, cv={float(params.cv):.6f}, defocus={float(params.defocus):.6f}, offset_x={float(params.offset_x_in_angstroms):.6f}, offset_y={float(params.offset_y_in_angstroms):.6f}"
                )

        return jnp.array(losses), jnp.array(cvs), jnp.array(defoci), jnp.array(wxyzs), jnp.array(offset_xs), jnp.array(offset_ys)

    losses, cvs, defoci, wxyzs, offset_xs, offset_ys = optimization(config.optimization, key, params0)

    plt.plot(losses, label="Loss")
    plt.legend()
    plt.show()
    plt.plot(cvs, label="CV")
    plt.legend()
    plt.show()
    plt.plot(defoci, label="Defocus")
    plt.legend()
    plt.show()
    plt.plot(wxyzs, label="WXYZ")
    plt.legend()
    plt.show()
    plt.plot(offset_xs, label="Offset X")
    plt.legend()
    plt.show()
    plt.plot(offset_ys, label="Offset Y")
    plt.legend()
    plt.show()
    
    return None

if __name__ == "__main__":
    main()