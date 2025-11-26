import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import Tuple, List, Optional
import pandas as pd
from tqdm import tqdm
import seaborn as sns


def compute_high_power_frac(arr, threshold):
    ny, nx = arr.shape
    # FFT high-frequency fraction:
    F = np.fft.fftshift(np.fft.fft2(arr - np.mean(arr)))
    ps = np.abs(F)**2
    # build frequency grid
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    kr = np.sqrt(KX**2 + KY**2)
    # choose threshold at e.g. 0.25 * max_freq (empirical)
    thresh = threshold * np.max(kr)
    high_power_frac = np.sum(ps[kr >= thresh]) / np.sum(ps)
    return high_power_frac


@dataclass
class ByTrialConfig:
    n_pixels_plot: int
    dose_to_n_trials: dict[int, int]
    dose_to_roughness: dict[int, float]


@dataclass
class EMDLandscapeConfig:
    fname: str
    average_over_trial: bool


@dataclass
class AxVLine:
    label: str
    value: float


@dataclass
class PlottingConfig:
    figure_size: Tuple[float, float]
    dpi: int
    marker: str
    palette: str
    axvline: Optional[List[AxVLine]] = None
    axhline: Optional[List[AxVLine]] = None
    skip_single_plots: bool = True


@dataclass
class ByProjectionConfig:
    emd_landscape: EMDLandscapeConfig
    average_sw_landscape_over_trial: bool
    plotting: PlottingConfig
    n_projections_smaller_list_size: int


@dataclass
class FullConfig:
    precomputed_landscape_paths: Tuple[str]
    roughness_threshold: float
    mode: str  
    by_trial: ByTrialConfig
    by_projection: ByProjectionConfig

    def __post_init__(self):
        valid_modes = ['by_trial', 'by_projection']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}.")


def plot_data(roughness_trends_fname, landscapes_and_images_fname, dose_to_n_trials, dose_to_roughness, output_dir, n_pixels_plot):

    roughness_trends_data_df = pd.read_csv(roughness_trends_fname)
    landscape_data = jnp.load(landscapes_and_images_fname)

    for noise_dose, df_dose_group in roughness_trends_data_df.groupby('noise_dose'):
        plt.figure()
        for (n_pixels, n_projections), df_pixel_group in df_dose_group.groupby(['n_pixels', 'n_projections']):
            label = f'n_pixels={n_pixels}'
            plt.plot((n_pixels * n_projections * df_pixel_group.running_average_index.values).tolist(), 
                     df_pixel_group.high_power_frac.values.tolist(), label=label)
        plt.xscale('log')
        plt.title(f'Roughness trend for noise dose={noise_dose}')
        plt.xlabel('Time per SW evaluation (n_trials x n_projections x n_pixels, arbitrary units)')
        plt.ylabel('Roughness')
        plt.axhline(y=dose_to_roughness[int(noise_dose)], color='tab:blue', linestyle='--', alpha=0.5)
        plt.axvline(x=n_pixels_plot * n_projections * dose_to_n_trials[int(noise_dose)], color='tab:blue', linestyle='--', alpha=0.5)
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, f'roughness_trends_dose{noise_dose}.png'))
        plt.clf()

    landscapes = landscape_data['landscapes']
    noise_dose_array = landscape_data['noise_dose']
    noisy_images = landscape_data['noisy_images']
    for i in range(len(noise_dose_array)):
        noise_dose = noise_dose_array[i]
        averaged_landscape = landscapes[i][:dose_to_n_trials[int(noise_dose)]].mean(axis=0)
        noisy_image = noisy_images[i]

        plt.figure()
        plt.imshow(averaged_landscape, 
                    extent=[df_dose_group.offset_x_in_angstroms_min.unique().item(), 
                            df_dose_group.offset_x_in_angstroms_max.unique().item(), 
                            df_dose_group.rotation_min_geodesic_angle_in_degrees.unique().item(), 
                            df_dose_group.rotation_max_geodesic_angle_in_degrees.unique().item()
                            ],
                    origin='lower',
                    aspect='auto',
                    cmap='jet')
        plt.colorbar()
        plt.title(f'Averaged landscape n_pixels={n_pixels_plot} | dose={noise_dose} | n_trials={dose_to_n_trials[int(noise_dose)]}')
        plt.xlabel('Offset x (angstroms)')
        plt.ylabel('Rotation (geodesic distance in degrees)')
        plt.savefig(os.path.join(output_dir, f'averaged_landscape_n_pixels_{n_pixels_plot}_dose{noise_dose}.png'))
        plt.clf()

        plt.figure()
        plt.imshow(noisy_image, cmap='gray_r')
        plt.colorbar()
        plt.title(f'Noisy image dose={noise_dose}')
        plt.savefig(os.path.join(output_dir, f'noisy_image_dose{noise_dose}.png'))
        plt.clf()


@hydra.main(version_base=None, config_path="./", config_name="config_balancing_tradeoff_loss_landscape_figure")
def main(cfg: FullConfig):
    logger = logging.getLogger(__name__)
    config = OmegaConf.merge(OmegaConf.structured(FullConfig), cfg)
    logger.info(f"Config: {config}")
    plt.style.use("seaborn-v0_8-poster")
    
    run_dir = HydraConfig.get().run.dir 
    logger.info(f"Run directory: {run_dir}")

    def parse_runs(precomputed_landscape_paths):
        d_list = []
        for dir in precomputed_landscape_paths:
            landscape_path = os.path.join('/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/', dir, 'data/sw_landscape.npy')
            logger.info(f'Loading {landscape_path}')
            config_fname = os.path.join('/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/', dir, '.hydra/config.yaml')
            config = OmegaConf.load(config_fname)
            noise_background = config.setup.noise.background
            noise_dose = config.setup.noise.dose
            if config.objective.use_this_objective == 'sw_random_pixels':
                sw_n_pixels = config.objective.sliced_wasserstein_random_pixels.n_pixels
                sw_n_projections = config.objective.sliced_wasserstein_random_pixels.n_projections
            elif config.objective.use_this_objective == 'sw':
                sw_n_pixels = config.objective.sliced_wasserstein.n_pixels
                sw_n_projections = config.objective.sliced_wasserstein.n_projections
            elif config.objective.use_this_objective == 'sw_cdf':
                sw_n_pixels = 1
                sw_n_projections = config.objective.sliced_wasserstein_cdf.n_projections
            elif config.objective.use_this_objective == 'sw_icdf':
                sw_n_pixels =  config.objective.sliced_wasserstein_icdf.n_interpolation_points
                sw_n_projections = config.objective.sliced_wasserstein_icdf.n_projections
            elif config.objective.use_this_objective == 'sw_dsort':
                sw_n_pixels = 1
                sw_n_projections = config.objective.sliced_wasserstein_dsort.n_projections
            else:
                raise ValueError(f"Invalid objective: {config.objective.use_this_objective}")
            d_list.append({
                'use_this_objective': config.objective.use_this_objective,
                'noise_background': noise_background,
                'noise_dose': noise_dose,
                'sw_n_pixels': sw_n_pixels,
                'sw_n_projections': sw_n_projections,
                'n_trials': config.landscape.n_trials,
                'noisy_image_path': os.path.join('/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/', dir, 'data/proj_obs_noisy.npy'),
                'landscape_path': landscape_path,
                'offset_x_in_angstroms_min': config.landscape.landscape_variables.offset_x_in_angstroms.min_value_in_angstroms,
                'offset_x_in_angstroms_max': config.landscape.landscape_variables.offset_x_in_angstroms.max_value_in_angstroms,
                'rotation_min_geodesic_angle_in_degrees': config.landscape.landscape_variables.rotation.min_geodesic_angle_in_degrees,
                'rotation_max_geodesic_angle_in_degrees': config.landscape.landscape_variables.rotation.max_geodesic_angle_in_degrees,
            })

        df = pd.DataFrame(d_list)
        return df
    df = parse_runs(config.precomputed_landscape_paths)
    print(df)

    if config.mode == 'by_projection':

        def zscore(arr):
            return (arr - arr.mean()) / arr.std()
        emd_landscape = jnp.load(config.by_projection.emd_landscape.fname)
        if config.by_projection.emd_landscape.average_over_trial:
            emd_landscape = emd_landscape.mean(axis=0)
        emd_landscape = zscore(emd_landscape)

        def compute_accuacy_and_roughness_and_make_single_plots(df, outdir, skip_single_plots):
            run_dir = outdir
            df_list = []
            for (use_this_objective, noise_dose), df_group in df.groupby(['use_this_objective', 'noise_dose']):
                for n_pixels, df_subgroup in df_group.groupby('sw_n_pixels'):
                    logger.info(f'Accuracy metrics for noise_dose={noise_dose}, n_pixels={n_pixels}')
                    sw_landscape_loaded = [jnp.load(d['landscape_path']) for _, d in df_subgroup.iterrows()]
                    

                    accuracies_all = []
                    roughness_all = []
                    avg_landacapes_all = []

                    for idx, sw_landscape in enumerate(sw_landscape_loaded):
                        if config.by_projection.average_sw_landscape_over_trial:
                            sw_landscape = sw_landscape.mean(axis=0)
                        

                        accuracy = []
                        accuracy_std = []
                        roughness_std = []
                        roughness = []
                        n_projections_list = []


                        n_projections = sw_landscape.shape[-1]
                        average_landacapes = []
                        K = 1  # number of cyclic shifts
                        rolls = jnp.linspace(0, n_projections, K, endpoint=False).astype(int)

                        n_projections_smaller_list = jnp.exp(jnp.linspace(jnp.log(3), jnp.log(n_projections), num=config.by_projection.n_projections_smaller_list_size)).astype(int)
                        for n_projections_smaller in tqdm(n_projections_smaller_list, desc="Downsampling projections", unit="n_projections"):
                            idxs = np.floor(np.arange(n_projections_smaller) * (n_projections / n_projections_smaller)).astype(int)

                            n_projections_list.append(len(idxs))

                            roll_accuracy = []
                            roll_roughness = []
                            roll_avg_land = []
                            for r in rolls:
                                avg_land = jnp.roll(sw_landscape, shift=-r, axis=-1)[:, :, idxs].mean(-1) 
                                roll_avg_land.append(avg_land)
                                accuracy_val = jnp.linalg.norm(emd_landscape - zscore(avg_land))
                                roll_accuracy.append(accuracy_val)
                                roughness_val = compute_high_power_frac(zscore(avg_land), config.roughness_threshold)
                                roll_roughness.append(roughness_val)
                            average_landacapes.append(jnp.stack(roll_avg_land).mean(0))
                            accuracy.append(jnp.stack(roll_accuracy).mean())
                            accuracy_std.append(jnp.stack(roll_accuracy).std())
                            roughness.append(jnp.stack(roll_roughness).mean())
                            roughness_std.append(jnp.stack(roll_roughness).std())

                        avg_landacapes_all.append((df_subgroup.iloc[idx]["sw_n_pixels"], n_projections_list, average_landacapes))
                        accuracies_all.append((df_subgroup.iloc[idx]["sw_n_pixels"], n_projections_list, accuracy))
                        roughness_all.append((df_subgroup.iloc[idx]["sw_n_pixels"], n_projections_list, roughness))

                    df_list.append(pd.DataFrame({
                        "noise_dose": noise_dose,
                        "n_pixels": n_pixels,
                        "n_projections": n_projections_list,
                        "average_landacapes": average_landacapes,
                        "accuracy": accuracy,
                        "accuracy_std": accuracy_std,
                        "roughness": roughness,
                        "roughness_std": roughness_std,
                        "use_this_objective": use_this_objective,
                    }))
                    
                    if skip_single_plots:
                        continue
                    logger.info(f'roughness_all: {roughness_all}')
                    logger.info(f'accuracies_all: {accuracies_all}')
                    # --------------------------------------------------
                    # 1) ACCURACY PLOT
                    # --------------------------------------------------
                    plt.figure(figsize=config.by_projection.plotting.figure_size)
                    for npix, nproj, acc in accuracies_all:
                        plt.plot(nproj, acc, marker='o', label=f'n_pixels={npix}')

                    plt.title(f'Accuracy to EMD landscape (dose={noise_dose})')
                    plt.xlabel('n_projections')
                    plt.ylabel('L2 error to EMD landscape (z-scored)')
                    plt.legend()
                    plt.xscale('log')
                    fname_acc = os.path.join(run_dir, f"accuracy_dose{noise_dose}_objective{use_this_objective}_npix{n_pixels}.png")
                    plt.savefig(fname_acc, dpi=config.by_projection.plotting.dpi, bbox_inches='tight')
                    plt.close()

                    # --------------------------------------------------
                    # 2) ROUGHNESS PLOT
                    # --------------------------------------------------
                    plt.figure(figsize=config.by_projection.plotting.figure_size)
                    for npix, nproj, rough in roughness_all:
                        plt.plot(nproj, rough, marker='o', label=f'n_pixels={npix}')

                    plt.title(f'Roughness vs n_projections (dose={noise_dose})')
                    plt.xlabel('n_projections')
                    plt.ylabel('Roughness')
                    plt.legend()
                    plt.xscale('log')
                    fname_rough = os.path.join(run_dir, f"roughness_dose{noise_dose}_objective{use_this_objective}_npix{n_pixels}.png")
                    plt.savefig(fname_rough, dpi=config.by_projection.plotting.dpi, bbox_inches='tight')
                    plt.close()

                    # --------------------------------------------------
                    # 2) joint plot SW vs EMD landscape
                    # --------------------------------------------------
                    import seaborn as sns
                    for npix, nproj, average_landacapes in avg_landacapes_all:
                        logger.info(f'Joint plot SW vs EMD landscape (dose={noise_dose}, n_pixels={npix})')
                        for n_projections_small, avg_land in zip(nproj, average_landacapes):
                            g = sns.jointplot(x=zscore(avg_land.flatten()), 
                                              y=zscore(emd_landscape.flatten()), 
                                              kind='scatter', 
                                              color='k',
                                              s=25,)
                            g.set_axis_labels(r"$SW_2$ Landscape", r"$W_2$ Landscape values")
                            # Set x & y ticks
                            ticks = [-2, -1, 0, 1, 2]
                            g.ax_joint.set_xticks(ticks)
                            g.ax_joint.set_yticks(ticks)
                            plt.suptitle(f'Joint plot SW vs EMD landscape (dose={noise_dose}, n_pixels={npix}, n_projections={n_projections_small})')
                            fname_joint = os.path.join(run_dir, f"jointplot_dose{noise_dose}_objective{use_this_objective}_npix{npix}_np{n_projections_small}.png")
                            plt.savefig(fname_joint, dpi=config.by_projection.plotting.dpi, bbox_inches='tight')
                            plt.close()
            
                    # --------------------------------------------------
                    # 2) landscape plots
                    # --------------------------------------------------
                    for npix, nproj, average_landacapes in avg_landacapes_all:
                        logger.info(f'Landscape plots (dose={noise_dose}, n_pixels={npix})')
                        for n_projections_small, avg_land in zip(nproj, average_landacapes):
                            plt.figure()
                            avg_land = zscore(np.array(avg_land))
                            plt.imshow(avg_land, 
                                        extent=[df_subgroup.offset_x_in_angstroms_min.unique().item(), 
                                                df_subgroup.offset_x_in_angstroms_max.unique().item(), 
                                                df_subgroup.rotation_min_geodesic_angle_in_degrees.unique().item(), 
                                                df_subgroup.rotation_max_geodesic_angle_in_degrees.unique().item()
                                                ],
                                        origin='lower',
                                        aspect='auto',
                                        cmap='jet')
                            plt.colorbar()
                            plt.title(f'Averaged landscape n_pixels={npix} | dose={noise_dose} | n_projections={n_projections_small}')
                            plt.xlabel('Offset x (angstroms)')
                            plt.ylabel('Rotation (geodesic distance in degrees)')
                            fname_landscape = os.path.join(run_dir, f"landscape_dose{noise_dose}_objective{use_this_objective}_npix{npix}_np{n_projections_small}.png")
                            plt.savefig(fname_landscape, dpi=config.by_projection.plotting.dpi, bbox_inches='tight')
                            plt.close()

            df = pd.concat(df_list, ignore_index=True)
            return df
        single_plots_dir = os.path.join(run_dir, "single_plots")
        os.makedirs(single_plots_dir, exist_ok=True)
        df_roughness_accuracy = compute_accuacy_and_roughness_and_make_single_plots(df, single_plots_dir, skip_single_plots=config.by_projection.plotting.skip_single_plots)

        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        df_fname = os.path.join(data_dir, f'by_projection_analysis.csv')
        df_roughness_accuracy['accuracy'] = df_roughness_accuracy['accuracy'].astype(float)
        df_roughness_accuracy['accuracy_std'] = df_roughness_accuracy['accuracy_std'].astype(float)
        df_roughness_accuracy['roughness'] = df_roughness_accuracy['roughness'].astype(float)
        df_roughness_accuracy['compute_cost'] = (df_roughness_accuracy['n_pixels'] * df_roughness_accuracy['n_projections']).astype(float)
        df_roughness_accuracy['plot_label'] = df_roughness_accuracy['use_this_objective'].astype(str).map({'sw_cdf': 'CDF', 'sw_icdf': 'iCDF', 'sw_dsort': 'Greedy Matching'})
        df_roughness_accuracy.to_csv(df_fname, index=False)    

        def plot_compute_accuracy_tradeoff(df, out_dir, dpi, marker, palette, axvline=None, axhline=None):
            

            for x, y in [
                ('compute_cost', 'roughness'),
                ('compute_cost', 'accuracy'),
                ('n_projections', 'accuracy'),
                ('n_projections', 'roughness')
            ]:
                print(f'Plotting {x} vs {y}')
                fig, ax = plt.subplots(figsize=(6, 6))

                # Draw main lines
                sns.lineplot(
                    data=df,
                    x=x,
                    y=y,
                    hue='plot_label',
                    marker=marker,
                    palette=palette,
                    ax=ax
                )

                # --- Extract actual colors seaborn assigned ---
                handles, labels = ax.get_legend_handles_labels()
                color_map = {label: h.get_color() for label, h in zip(labels, handles)}

                # Add shading per group
                for label, df_group in df.groupby('plot_label'):
                    if y == 'accuracy':
                        std_col = 'accuracy_std'
                    else:
                        continue
                    # elif y == 'roughness':
                    #     std_col = 'roughness_std'
                    # else:
                    #     raise ValueError(f'Invalid y: {y}')
                    ax.fill_between(
                            df_group[x],
                            df_group[y] - df_group[std_col],
                            df_group[y] + df_group[std_col],
                            alpha=0.2,
                            color=color_map[label]   # <-- Fixed
                        )

                plt.xscale('log')
                plt.xlabel(x.replace('_', ' ').title())
                plt.ylabel(y.replace('_', ' ').title())
                fontsize = ax.xaxis.label.get_fontsize()

                leg = ax.legend(title="SW Type", fontsize=fontsize)
                leg.get_title().set_fontsize(fontsize)

                if axvline is not None: 
                    for axvline_dict in axvline:
                        if axvline_dict['label'] == x:
                            plt.axvline(x=axvline_dict['value'], color='gray', linestyle='--', alpha=0.5)
                if axhline is not None:
                    for axhline_dict in axhline:
                        if axhline_dict['label'] == y:
                            plt.axhline(y=axhline_dict['value'], color='gray', linestyle='--', alpha=0.5)
                    
                plt.savefig(os.path.join(out_dir, f'by_projection_{x}_vs_{y}.png'), dpi=dpi, bbox_inches='tight')
                plt.close(fig)
        plot_compute_accuracy_tradeoff(df_roughness_accuracy, 
                                       run_dir, 
                                       config.by_projection.plotting.dpi,
                                       config.by_projection.plotting.marker, 
                                       config.by_projection.plotting.palette,
                                       config.by_projection.plotting.axvline,
                                       config.by_projection.plotting.axhline,
                                       )

        return 
    elif config.mode == 'by_trial':

        roughness_trends_data = []
        landscape_data = []
        noise_dose_list = []
        noisy_image_list = []
        for noise_dose, df_group in df.groupby('noise_dose'):
            noise_dose_list.append(noise_dose)
            # group by noise_dose
            sw_landscape_loaded = [jnp.load(d['landscape_path']) for _, d in df_group.iterrows()]

            # plot roughness trends
            for idx, sw_landscape in enumerate(sw_landscape_loaded):
                logger.info(f'Roughness metrics for n_pixels={df_group.iloc[idx]["sw_n_pixels"]}')
                normalized = (sw_landscape - sw_landscape.mean()) / sw_landscape.std()
                running_average_index = jnp.arange(1, normalized.shape[0]+1)
                high_power_frac = jnp.stack([compute_high_power_frac(normalized[:i].mean(axis=0), config.roughness_threshold) for i in running_average_index])
                roughness_trends_data.append(pd.DataFrame({
                    'noise_dose': noise_dose,
                    'n_pixels': df_group.iloc[idx]['sw_n_pixels'],
                    'n_projections': df_group.iloc[idx]['sw_n_projections'],
                    'high_power_frac': high_power_frac,
                    'running_average_index': running_average_index,
                    'dose': noise_dose,
                    'rotation_max_geodesic_angle_in_degrees': df_group.iloc[idx]['rotation_max_geodesic_angle_in_degrees'],
                    'rotation_min_geodesic_angle_in_degrees': df_group.iloc[idx]['rotation_min_geodesic_angle_in_degrees'],
                    'offset_x_in_angstroms_max': df_group.iloc[idx]['offset_x_in_angstroms_max'],
                    'offset_x_in_angstroms_min': df_group.iloc[idx]['offset_x_in_angstroms_min'],
                }))

            # plot averaged landscape
            for idx, sw_landscape in enumerate(sw_landscape_loaded):
                if df_group.iloc[idx]['sw_n_pixels'] == config.n_pixels_plot:
                    normalized = (sw_landscape - sw_landscape.mean()) / sw_landscape.std()
                    # averaged_landscape = normalized.mean(axis=0)
                    landscape_data.append(normalized)
                

            # noisy image
            noisy_image_path = df_group.iloc[0]['noisy_image_path']
            noisy_image = jnp.load(noisy_image_path)
            noisy_image_list.append(noisy_image)

        # save data
        roughness_trends_data_df = pd.concat(roughness_trends_data, ignore_index=True)
        roughness_trends_fname = os.path.join(run_dir, f'roughness_trends.csv')
        roughness_trends_data_df.to_csv(roughness_trends_fname, index=False)
        landscapes_and_images_fname = os.path.join(run_dir, f'landscapes_and_images.npz')
        jnp.savez(landscapes_and_images_fname, 
                landscapes=jnp.stack(landscape_data),
                noise_dose=jnp.array(noise_dose_list),
                noisy_images=jnp.stack(noisy_image_list),
                n_pixels_plot=config.n_pixels_plot,
                )
        plot_data(roughness_trends_fname, landscapes_and_images_fname, config.dose_to_n_trials, config.dose_to_roughness, run_dir, config.n_pixels_plot)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")


if __name__ == "__main__":
    main()