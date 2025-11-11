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
class PoseConfig:
    rotation: List[float] = field(default_factory=list)
    translation: List[float] = field(default_factory=list)


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
    rotation: List[float]
    translation: List[float]
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


def apply_rotation_and_cv(volume, scattering_theory, image_config, rotation_matrix, cv_scalar, defocus_scalar):
    new_positions = tear_cv(volume.positions, normal=jnp.array([1., 1, 0.]), offset=0.0, tear_amount=cv_scalar)
    rotated_positions = new_positions @ rotation_matrix.T
    volume_updated = eqx.tree_at(lambda v: v.positions, volume, rotated_positions)
    # scattering_theory = WeakPhaseScatteringTheory(integrator, transfer_theory)
    scattering_theory = eqx.tree_at(lambda m: m.transfer_theory.ctf.defocus_in_angstroms, scattering_theory, defocus_scalar)

    image_ref = irfftn(scattering_theory.compute_intensity_spectrum(volume_updated, image_config))
    return image_ref


def loss_for_pose_and_cv(image_ref, volume, scattering_theory, image_config, rotation_matrix, cv_scalar, defocus_scalar, grid, sliced_wasserstein, key):
    image_1 = apply_rotation_and_cv(volume, scattering_theory, image_config, rotation_matrix, cv_scalar, defocus_scalar) 
    

    def wasserstein_random_projections(image_1, image_2, key):
        # return jnp.linalg.norm(image_1 - image_2)
        
        v_weights = image_1.flatten()
        u_weights = image_2.flatten()

        thetas = jnp.linspace(0, jnp.pi, sliced_wasserstein.n_projections, endpoint=False) 
        keys = jax.random.split(key, len(thetas))

        def loss_per_theta(theta, key):
            project = jnp.tensordot(grid, jnp.array([jnp.cos(theta), jnp.sin(theta)]), axes=1).flatten()
            perm = jax.random.permutation(key, len(project))[:sliced_wasserstein.n_pixels]
            x_ab = project[perm]
            sort_idx = jnp.argsort(x_ab)
            x_ab_sorted = x_ab[sort_idx]
            a_sorted = u_weights[perm][sort_idx] + sliced_wasserstein.normalize_safely_epsilon
            b_sorted = v_weights[perm][sort_idx] + sliced_wasserstein.normalize_safely_epsilon
            return emd_1d_sorted_jax_differentiable(x_ab_sorted, x_ab_sorted, a_sorted / a_sorted.sum(), b_sorted / b_sorted.sum())

        losses = jax.vmap(loss_per_theta)(thetas, keys)
        return losses.mean()
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
    gaussian_integrator = cxs.GaussianMixtureProjection()
    # image_model = cxs.make_image_model(
    #     volume,
    #     image_config,
    #     pose=cxs.EulerAnglePose(),
    #     transfer_theory=cxs.ContrastTransferTheory(cxs.AstigmaticCTF(defocus_in_angstroms=config.setup.ctf.defocus_in_angstroms)),
    # )
    # image_ref = image_model.simulate()
    # image_ref = apply_rotation_and_cv(volume, 
    #                                  gaussian_integrator, 
    #                                  image_config, 
    #                                  rotation_matrix=jnp.eye(3), 
    #                                  cv_scalar=config.setup.cv.value)

    envelope = FourierGaussian(amplitude=config.setup.envelope.amplitude, 
                               b_factor=config.setup.envelope.b_factor)
    transfer_theory = cxs.ContrastTransferTheory(
            cxs.AstigmaticCTF(defocus_in_angstroms=config.setup.ctf.defocus_in_angstroms),
            envelope=envelope,
        )
    scattering_theory = WeakPhaseScatteringTheory(gaussian_integrator, transfer_theory)
    image_ref = irfftn(scattering_theory.compute_intensity_spectrum(volume, image_config))
    plt.imshow(image_ref, cmap='gray')
    plt.show()

    
    xm, ym = jnp.meshgrid(
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        indexing='ij',
    )
    grid = jnp.stack([xm, ym], axis=-1)
    proj_obs = apply_rotation_and_cv(volume=volume, 
                                     scattering_theory=scattering_theory,
                                     image_config=image_config, 
                                     rotation_matrix=jnp.eye(3), 
                                     cv_scalar=config.setup.cv.value, 
                                     defocus_scalar=config.setup.ctf.defocus_in_angstroms) 
    def loss_for_cv(cv, defocus, key):
        return loss_for_pose_and_cv(image_ref=proj_obs, 
                                    volume=volume, 
                                    scattering_theory=scattering_theory, 
                                    image_config=image_config, 
                                    rotation_matrix=jnp.eye(3), 
                                    cv_scalar=cv, 
                                    defocus_scalar=defocus,
                                    grid=grid, 
                                    sliced_wasserstein=config.optimization.sliced_wasserstein, 
                                    key=key)

    grad_loss_for_cv = eqx.filter_jit(eqx.filter_grad(loss_for_cv))

    # --- Example gradient descent loop ---
    key = jax.random.key(0)
    cv = jnp.array(config.inference_model.initial_estimates.cv.value)
    defocus = jnp.array(config.inference_model.initial_estimates.ctf.defocus_in_angstroms)

    def optimization(optimization, key, initial_cv, initial_defocus):
        lr = optimization.learning_rate
        losses = []
        cvs = []
        cv = initial_cv
        defocus = initial_defocus
        for i in range(optimization.n_iterations):
            key, subkey = jax.random.split(key)
            loss_val = loss_for_cv(cv, defocus, subkey)
            losses.append(loss_val)
            cvs.append(cv)
            grad_val = grad_loss_for_cv(cv, defocus, subkey)
            cv = cv - lr * grad_val
            defocus = defocus - lr * grad_val * optimization.preconditioner.defocus_to_cv_ratio
            if i % optimization.print_every == 0: 
                print(f"Step {i:02d}: loss={loss_val:.6f}, grad={grad_val:.6f}, cv={cv:.3f}, defocus={defocus:.3f}")
        return jnp.array(losses), jnp.array(cvs)
    losses, cvs = optimization(config.optimization, key, cv, defocus)

    plt.plot(losses)
    plt.show()
    plt.plot(cvs)
    plt.show()
    return None

if __name__ == "__main__":
    main()