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


@dataclass
class SpecimenConfig:
    pdb_file: str
    atomic_number_exclusion: List[int] = field(default_factory=list)


@dataclass
class NoiseConfig:
    dose: float
    background: float


@dataclass
class IntegratorConfig:
    grid_size: int
    pixel_size: float
    atom_width: float
    atom_amplitude: float


@dataclass
class PoseConfig:
    rotation: List[float] = field(default_factory=list)
    translation: List[float] = field(default_factory=list)


@dataclass
class CTFConfig:
    defocus_u: float
    defocus_v: float
    astigmatism_angle: float
    cs: float
    voltage: float
    amplitude_contrast: float


@dataclass
class CVConfig:
    type: str
    value: float

    # post init that type comes from list of tear and squeeze
    def __post_init__(self):
        if self.type not in ['tear', 'squeeze']:
            raise ValueError("CV type must be either 'tear' or 'squeeze'")


@dataclass
class SetupConfig:
    specimen: SpecimenConfig
    noise: NoiseConfig
    integrator: IntegratorConfig
    pose: PoseConfig
    ctf: CTFConfig
    cv: CVConfig


@dataclass
class SlicedWassersteinConfig:
    n_projections: int
    n_pixels: int


@dataclass
class OptimizationConfig:
    learning_rate: float
    n_iterations: int
    sliced_wasserstein: SlicedWassersteinConfig
    print_every: int


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

def apply_rotation_and_cv(volume, integrator, image_config, rotation_matrix, cv_scalar):
    # atom_positions = apply_squeeze(atom_volume.position_pytree, cv_scalar)
    # rotated_positions = atom_positions @ rotation_matrix.T
    # volume = eqx.tree_at(
    #     lambda v: v.position_pytree, volume, rotated_positions
    # )
    # proj_ref = fft_integrator.integrate(
    #     volume, image_config, outputs_real_space=True
    # )
    # print(volume.positions)
    # atom_positions = apply_squeeze(volume.positions, cv_scalar)
    new_positions = tear_cv(volume.positions, normal=jnp.array([1., 1, 0.]), offset=0.0, tear_amount=cv_scalar)
    rotated_positions = new_positions @ rotation_matrix.T
    volume_updated = eqx.tree_at(lambda v: v.positions, volume, rotated_positions)
    proj_ref = integrator.integrate(
            volume_updated, image_config, outputs_real_space=True
        )
    return proj_ref

def loss_for_pose_and_cv(image_ref, volume, integrator, image_config, rotation_matrix, cv_scalar, grid, n_rotations, n_pix_samples, key):
    image_1 = apply_rotation_and_cv(volume, integrator, image_config, rotation_matrix, cv_scalar) 
    
    def wasserstein_random_projections(image_1, image_2, key):
        
        # return jnp.linalg.norm(image - proj_inf)
        v_weights = image_1.flatten()
        u_weights = image_2.flatten()

        thetas = jnp.linspace(0, jnp.pi, n_rotations, endpoint=False) 
        keys = jax.random.split(key, len(thetas))

        def loss_per_theta(theta, key):
            project = jnp.tensordot(grid, jnp.array([jnp.cos(theta), jnp.sin(theta)]), axes=1).flatten()
            perm = jax.random.permutation(key, len(project))[:n_pix_samples]
            x_ab = project[perm]
            sort_idx = jnp.argsort(x_ab)
            x_ab_sorted = x_ab[sort_idx]
            a_sorted = u_weights[perm][sort_idx] + 1e-8
            b_sorted = v_weights[perm][sort_idx] + 1e-8
            return emd_1d_sorted_jax_differentiable(x_ab_sorted, x_ab_sorted, a_sorted / a_sorted.sum(), b_sorted / b_sorted.sum())

        losses = jax.vmap(loss_per_theta)(thetas, keys)
        return losses.mean()
    return wasserstein_random_projections(image_1, image_ref, key)

@hydra.main(version_base=None, config_path="./", config_name="config_joint_gradient_descent")
def main(cfg: FullConfig):
    print(OmegaConf.to_yaml(cfg))
    config = OmegaConf.merge(OmegaConf.structured(FullConfig), cfg)

    atom_positions, atom_type = read_atoms_from_pdb('1uao_assembly.pdb', center=True, loads_properties=False)
    atom_positions = atom_positions[atom_type != 1]
    pixel_size, shape = 0.35, (32, 32)
    pad_options = dict(shape=(64, 64))
    image_config = cxs.BasicImageConfig(
        shape, pixel_size, voltage_in_kilovolts=300.0, pad_options=pad_options
    )
    width = 0.5
    volume = gaussian_volume = cxs.GaussianMixtureVolume(
                atom_positions,
                amplitudes=1.0,
                variances=width**2,
            )
    gaussian_integrator = cxs.GaussianMixtureProjection()
    proj_ref = gaussian_integrator.integrate(
                gaussian_volume, image_config, outputs_real_space=True
            )
    from goto_swap.backends.jax import emd_1d_sorted_jax_differentiable
    theta = 0 # 45 degrees
    n_pix = pad_options['shape'][0]
    xm, ym = jnp.meshgrid(
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        jnp.linspace(-n_pix // 2, n_pix // 2 - 1, n_pix),
        indexing='ij',
    )
    xy = jnp.stack([xm, ym], axis=-1)
    proj_inf = apply_rotation_and_cv(volume, gaussian_integrator, image_config, rotation_matrix=jnp.eye(3), cv_scalar=2.5) 
    def loss_for_cv(cv, key):
        return loss_for_pose_and_cv(proj_inf, volume, gaussian_integrator, image_config, jnp.eye(3), cv, xy, n_rotations=30, n_pix_samples=100, key=key)

    grad_loss_for_cv = eqx.filter_jit(eqx.filter_grad(loss_for_cv))

    # --- Example gradient descent loop ---
    key = jax.random.key(0)
    cv = jnp.array(config.inference_model.initial_estimates.cv.value)

    def optimization(optimization, key, initial_cv):
        lr = optimization.learning_rate
        losses = []
        cvs = []
        cv = initial_cv
        for i in range(optimization.n_iterations):
            key, subkey = jax.random.split(key)
            loss_val = loss_for_cv(cv, subkey)
            losses.append(loss_val)
            cvs.append(cv)
            grad_val = grad_loss_for_cv(cv, subkey)
            cv = cv - lr * grad_val
            if i % optimization.print_every == 0: 
                print(f"Step {i:02d}: loss={loss_val:.6f}, grad={grad_val:.6f}, cv={cv:.3f}")
        return jnp.array(losses), jnp.array(cvs)
    losses, cvs = optimization(config.optimization, key, cv)

    plt.plot(losses)
    plt.show()
    plt.plot(cvs)
    plt.show()
    return None

if __name__ == "__main__":
    main()