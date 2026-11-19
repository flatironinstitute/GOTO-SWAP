import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATHS = [
    ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/13-23-14/data/','sw_cdf'],
    ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/13-31-17/data/','sw_icdf'],
    ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/13-32-59/data/','sw_dsort'],

    ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/14-23-57/data/','mse__from_sw_cdf_endpoint'],
    ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/14-27-37/data/','mse__from_sw_icdf_endpoint'],
    ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/14-30-28/data/','mse__from_sw_dsort_endpoint'],
    
    ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/14-31-43/data/','mse'],
]

odir = '/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/figure3'



if __name__ == '__main__':
    parameters = ['losses','cvs','defoci','geodesic_angles','offset_xs','offset_ys']

    def panel_e():
        for directory, label in PATHS:
            path = f"{directory}/joint_gradient_descent_traces.csv"
            df = pd.read_csv(path)
            for param in parameters:
                plt.plot(df[param], label=label)
                plt.savefig(f"{odir}/panele_figure_{label}_{param}.png")
                plt.close()

    def panel_d():
        # ground truth
        fname_ground_truth = '/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/13-23-14/data/proj_obs.npy'
        ground_truth = np.load(fname_ground_truth)
        plt.imshow(ground_truth, cmap='gray')
        plt.savefig(f"{odir}/paneld_ground_truth.png")
        plt.close()

        # observation
        fname_obs = '/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/13-23-14/data/proj_obs_noisy.npy'
        observation = np.load(fname_obs)
        plt.imshow(observation, cmap='gray')
        plt.savefig(f"{odir}/paneld_observation.png")
        plt.close()

        # initial prediction
        fname_obs = '/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/13-23-14/data/proj_initial.npy'
        observation = np.load(fname_obs)
        plt.imshow(observation, cmap='gray')
        plt.savefig(f"{odir}/paneld_initial_prediction.png")
        plt.close()

    def panel_f():

        for directory, label in PATHS:
            path = f"{directory}/proj_final.npy"
            image = np.load(path)
            plt.imshow(image, cmap='gray')
            plt.savefig(f"{odir}/panelf_figure_{label}.png")

    def panel_b():

        PATHS = [
            ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/16-06-08/data/','l2_landscape', 'l2', False],
            ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/16-07-07/data/','sw_landscape', 'sw_cdf', True],
            ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/16-08-06/data/','sw_landscape', 'sw_icdf', True],
            ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/16-09-05/data/','sw_landscape', 'sw_dsort', True],
            ['/mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-05-18/16-10-06/data/','emd_landscape', 'EMD', False],
        ]
        for directory, fname, label, reshape in PATHS:
            print(f"Processing {label}...")
            path = f"{directory}/{fname}.npy"
            landscape = np.load(path)
            if reshape: landscape = landscape.mean(axis=(0,-1))
            plt.imshow(landscape, cmap='jet')
            plt.savefig(f"{odir}/panelb_landscape_{label}.png")
            plt.close()

            # path = f"{directory}/emd_landscape.npy"
            # landscape = np.load(path).mean(axis=(0,-1))
            # plt.imshow(landscape, cmap='jet')
            # plt.savefig(f"{odir}/panelb_emd_landscape.png")

    def panel_c():
        pass

    # panel_b()