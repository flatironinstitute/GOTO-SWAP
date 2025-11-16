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
from typing import Any, Union, Tuple
import pandas as pd



def compute_roughness_metrics(arr, threshold):
    ny, nx = arr.shape
    # FFT high-frequency fraction:
    F = np.fft.fftshift(np.fft.fft2(arr - np.mean(arr)))
    ps = np.abs(F)**2
    # build frequency grid
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)
    # choose threshold at e.g. 0.25 * max_freq (empirical)
    thresh = threshold * np.max(kr)
    high_power_frac = np.sum(ps[kr >= thresh]) / np.sum(ps)
    return high_power_frac


@dataclass
class FullConfig:
    precomputed_landscape_paths: Tuple[str]
    n_pixels_plot: int
    roughness_threshold: float

@hydra.main(version_base=None, config_path="./", config_name="config_balancing_tradeoff_loss_landscape_figure")
def main(cfg: FullConfig):
    logger = logging.getLogger(__name__)
    config = OmegaConf.merge(OmegaConf.structured(FullConfig), cfg)
    
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
            sw_n_pixels = config.objective.sliced_wasserstein.n_pixels
            sw_n_projections = config.objective.sliced_wasserstein.n_projections
            d_list.append({
                'noise_background': noise_background,
                'noise_dose': noise_dose,
                'sw_n_pixels': sw_n_pixels,
                'sw_n_projections': sw_n_projections,
                'n_trials': config.landscape.n_trials,
                'noisy_image_path': os.path.join('/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/', dir, 'data/proj_obs_noisy.npy'),
                'landscape_path': landscape_path,
            })

        df = pd.DataFrame(d_list)
        return df
    df = parse_runs(config.precomputed_landscape_paths)
    print(df)

    roughness_trends_data = []
    average_landscapes_data = []
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
            trend = jnp.stack([compute_roughness_metrics(normalized[:i].mean(axis=0), config.roughness_threshold) for i in range(1, normalized.shape[0]+1)])
            label = 'n_pixels='+str(df_group.iloc[idx]['sw_n_pixels'])
            roughness_trends_data.append(pd.DataFrame({
                'noise_dose': noise_dose,
                'n_pixels': df_group.iloc[idx]['sw_n_pixels'],
                'trend': trend,
                'dose': noise_dose,
            }))
            plt.plot(trend, label=label)
        plt.legend()
        plt.savefig(os.path.join(run_dir, f'roughness_trends_dose{noise_dose}.png'))
        plt.clf()

        # plot averaged landscape
        for idx, sw_landscape in enumerate(sw_landscape_loaded):
            if df_group.iloc[idx]['sw_n_pixels'] == config.n_pixels_plot:
                normalized = (sw_landscape - sw_landscape.mean()) / sw_landscape.std()
                averaged_landscape = normalized.mean(axis=0)
                average_landscapes_data.append(averaged_landscape)
                plt.figure()
                plt.imshow(averaged_landscape, cmap='jet')
                plt.colorbar()
                plt.title(f'n_pixels={config.n_pixels_plot}')
                plt.savefig(os.path.join(run_dir, f'averaged_landscape_n_pixels_{config.n_pixels_plot}_dose{noise_dose}.png'))
                plt.clf()

        # noisy image
        noisy_image_path = df_group.iloc[0]['noisy_image_path']
        noisy_image = jnp.load(noisy_image_path)
        noisy_image_list.append(noisy_image)
        plt.figure()
        plt.imshow(noisy_image, cmap='gray_r')
        plt.colorbar()
        plt.title(f'Noisy image dose={noise_dose}')
        plt.savefig(os.path.join(run_dir, f'noisy_image_dose{noise_dose}.png'))
        plt.clf()

    # save data
    roughness_trends_data_df = pd.concat(roughness_trends_data, ignore_index=True)
    roughness_trends_data_df.to_csv(os.path.join(run_dir, f'roughness_trends.csv'))
    jnp.savez(os.path.join(run_dir, f'landscapes_and_images_{config.n_pixels_plot}.npz'), 
              average_landscape=jnp.stack(average_landscapes_data),
              noise_dose=jnp.array(noise_dose_list),
              noisy_images=jnp.stack(noisy_image_list)
              )
    
    # plot
    # nrows = 3
    # ncols = len(noise_dose_list)
    # base_figsize = 4
    # figsize = (base_figsize*ncols, base_figsize*nrows)
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for noise_dose, df_dose_group in roughness_trends_data_df.groupby('noise_dose'):
        plt.figure()
        for n_pixels, df_pixel_group in df_dose_group.groupby('n_pixels'):
            label = 'n_pixels='+str(n_pixels)
            plt.plot(df_pixel_group.trend.values.tolist(), label=label)
        plt.title(f'Roughness trend for noise dose={noise_dose}')
        plt.xlabel('Number of averaged trials')
        plt.ylabel('High-frequency power fraction')
        plt.legend()
        plt.savefig(os.path.join(run_dir, f'roughness_trends_dose{noise_dose}_2.png'))
        plt.clf()

    landscape_data = jnp.load(os.path.join(run_dir, f'landscapes_and_images_{config.n_pixels_plot}.npz'))
    average_landscapes = landscape_data['average_landscape']
    noise_dose_array = landscape_data['noise_dose']
    noisy_images = landscape_data['noisy_images']
    for i in range(len(noise_dose_array)):
        noise_dose = noise_dose_array[i]
        averaged_landscape = average_landscapes[i]
        noisy_image = noisy_images[i]

        plt.figure()
        plt.imshow(averaged_landscape, cmap='jet')
        plt.colorbar()
        plt.title(f'Averaged landscape n_pixels={config.n_pixels_plot} dose={noise_dose}')
        plt.savefig(os.path.join(run_dir, f'averaged_landscape_n_pixels_{config.n_pixels_plot}_dose{noise_dose}_2.png'))
        plt.clf()

        plt.figure()
        plt.imshow(noisy_image, cmap='gray_r')
        plt.colorbar()
        plt.title(f'Noisy image dose={noise_dose}')
        plt.savefig(os.path.join(run_dir, f'noisy_image_dose{noise_dose}_2.png'))
        plt.clf()


if __name__ == "__main__":
    main()