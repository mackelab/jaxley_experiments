from rfest import ASD
from sklearn.linear_model import LinearRegression
import numpy as np


def get_rf_linreg(ca, images, noise_std=0.0, **kwargs):
    test_images_T = images.T
    noise = np.reshape(test_images_T, (test_images_T.shape[0], 20 * 15))
    X = noise

    ca = ca.T
    ca = np.random.randn(*ca.shape) * noise_std + ca
    y = ca

    dims = (20, 15)
    
    reg = LinearRegression().fit(X, y)
    rf = np.reshape(reg.coef_, dims)

    return rf


def get_rf_asd(ca, images, noise_std=0.0, num_iter=100):
    test_images_T = images.T
    noise = np.reshape(test_images_T, (test_images_T.shape[0], 20 * 15))
    X = noise

    ca = ca.T
    ca = np.random.randn(*ca.shape) * noise_std + ca
    y = ca
    
    dims = (20, 15)
    
    # Run ASD.
    asd = ASD(X, y, dims=dims)
    asd.fit(p0=[1.2, 1.2, 1.2, 1.2, 1.2], num_iters=num_iter, verbose=0, step_size=2e-2)
    rf = np.reshape(asd.w_opt, dims)

    rf -= np.min(rf)
    rf /= np.max(rf)
    
    return rf


def recover_raw_rf(rec_id, roi_id, rfs_raw):
    rf_raw = rfs_raw[rfs_raw["rec_id"] == rec_id]
    rf_raw = rf_raw[rf_raw["roi_id"] == roi_id]
    rf = np.stack(rf_raw["rf"].to_numpy()).reshape((20, 15))
    rf = rf - np.min(rf)
    rf = rf / np.max(rf)
    return rf


def compute_all_trained_rfs(counters, loss_weights_eval, all_ca_predictions, all_images, noise_mag, num_iter):
    # Compute receptive fields.
    rfs_trained = []

    for counter in counters:
        _ = np.random.seed(counter)
        roi_was_measured = loss_weights_eval[:, counter].astype(bool)
        rfs_trained.append(
            get_rf_asd(
                all_ca_predictions[roi_was_measured, counter],
                all_images[:, :, roi_was_measured],
                noise_std=noise_mag,
                num_iter=num_iter,
            )
        )
    return rfs_trained