import functools
import os.path

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import initializers
from functools import partial
from scipy.stats import cauchy, norm

sns.set(style="whitegrid", font_scale=2.25)
# plt.rcParams.update({"text.usetex": True})
import sys

sys.path.append('../')
from src.vimlts_fast import VimltsLinear, ConjungateDenseViGauss

print(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('Tensorflow version: ', tf.__version__, )


# %%

def softplus_inv(y):
    return np.log(np.exp(y) - 1)


class LogKL(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epochs'] = epoch

# %%


def initialize_models(seed=2, prior_dist=tfd.Normal(loc=0.,scale=1.)):
    models = {}
    # Ms = [1, 3, 10, 30, 100, 300]
    Ms = [2, 6, 20, 60, 200,]
    theta_start = -5
    theta_stop = 5
    for M in Ms:
        # init params
        kernel_initializers=dict(kernel_init_alpha_w = initializers.Constant(1.),
                                 kernel_init_beta_w = initializers.Constant(0.),
                                 kernel_init_alpha_z = initializers.Constant(1.),
                                 kernel_init_beta_z = initializers.Constant(0.),
                                 kernel_init_thetas = [initializers.Constant(theta_start)] + [initializers.Constant(softplus_inv((theta_stop-theta_start)/(M))) for i in range(M)])
        # define model
        tf.random.set_seed(seed)
        np.random.seed(seed)
        layer = VimltsLinear(1,
                             activation=lambda x: x,
                             **kernel_initializers,
                             num_samples=10000,
                             prior_dist=prior_dist,
                             input_shape=(1,))
        model = tf.keras.Sequential([layer], name=f"VIMLTS-degree{M}")
        model.build(input_shape=(None, 1))
        models[f"TM-VI, M={M}"] = model
    # Add MFVI
    tf.random.set_seed(seed)
    np.random.seed(seed)
    vi_gauss_l = ConjungateDenseViGauss(1,
                                        activation=lambda x: x,
                                        num_samples=10000,
                                        kernel_init_mu_w=initializers.Constant(0.),
                                        kernel_init_rhosigma_w=initializers.Constant(softplus_inv(2.5)),
                                        prior_dist=prior_dist)
    vi_gauss = tf.keras.Sequential([vi_gauss_l], name="Gauss-VI")
    vi_gauss.build(input_shape=(None, 1))
    vi_gauss.summary()
    models["Gauss-VI"] = vi_gauss
    return models


@tf.function
def sample_nll(y_obs, y_pred, scale):
    """
    Args:
        y_obs: true labels. Expected shape (#batch, 1) or (#batch)
        y_pred: model prediction. Expected shape (#samples, #batch, 1) or (#samples, #batch)

    Returns: sum of Nll
    """
    if len(y_pred.shape) == 2:  # Bug tf?! If we have a single output it squeezes y_pred. I did not want this behaviour.
        y_pred = y_pred[..., None]
    tf.debugging.check_numerics(y_pred, "Prediction for nll computation contains NaNs or Infs")
    error_str = f"Expected one of the above defined shapes. Got shapes: y_obs: {y_obs.shape}; y_pred: {y_pred.shape}"
    assert y_pred.shape[-1] == y_obs.shape[-1] or ((len(y_pred.shape) == 3) and y_pred.shape[-1] == 1), error_str

    # dist = tfp.distributions.Normal(loc=y_pred, scale=scale)
    dist = tfd.Cauchy(loc=y_pred, scale=scale)
    nll_per_sample = -dist.log_prob(y_obs)
    nlls = tf.reduce_mean(nll_per_sample, axis=0)
    tf.debugging.check_numerics(nlls, "NLL contains NaNs or Infs")
    return tf.reduce_sum(nlls)

def train(models, data, epochs):
    for name, model in models.items():
        print(f"Start experiment with model {name}")
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(
        #     partial(scheduler, lr_start=0.1, lr_stop=0.025, epochs=epochs))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=sample_nll_p,
                      run_eagerly=False)
        model.fit(tf.ones(data.shape), data, epochs=epochs, verbose=False, callbacks=[LogKL()])


def eval(models, ytensor, prior_dist):
    # Compute posterior from samples
    def posterior_unnormalized(samples, data, prior_dist=tfd.Normal(loc=0, scale=1.)):
        assert len(samples.shape) == 1, "Expected samples to be one dimensional"
        log_liks = np.sum(cauchy.logpdf(data.reshape(1,-1), loc=samples.reshape(-1,1), scale=sigma), axis=1)
        priors = prior_dist.log_prob(samples).numpy()
        post_un = log_liks + priors
        sort = np.argsort(samples)
        samples_s = samples[sort]
        post_un_s = post_un[sort]
        return post_un_s

    # compute log(p(D))
    x = np.linspace(-10,10, 1000000)
    log_ps = posterior_unnormalized(x, ytensor)
    log_p_D = np.log(np.trapz(np.exp(log_ps), x))
    del x
    del log_ps

    kl_summary = {}
    samples = {}
    for name, model in models.items():
        layer = model.layers[0]
        w, log_qw = layer.sample(num=100000)
        w = w.numpy().squeeze()
        sort = np.argsort(w)
        w = w[sort]
        log_qw = log_qw.numpy().squeeze()[sort]
        # compute KL
        log_p_w_D_unnorm = posterior_unnormalized(w, ytensor, prior_dist=prior_dist)
        kl_unnorm = np.mean(log_qw - log_p_w_D_unnorm)
        # 1/T \sum_{w_i \sim q(w_i}} log(q(w_i)/p(w_i|D)) => log(p(D)) + 1/T \sum_{w_i \sim q(w_i}} log(q(w_i)) - log(p(D|w_i)*p(w_i))
        kl_summary[name] = np.array(log_p_D + kl_unnorm)
        samples[name] = dict(w=w, log_q_w=log_qw)
        print(f"{name} KL: {kl_summary[name]:.2e}")
    return kl_summary, samples

# %%
if __name__ == '__main__':
    data_file = "../ipynb/01_singel_weight_data.npy"
    data = np.load(data_file)
    print(data.shape)
    data = data.reshape(-1,1)
    num = len(data)
    sigma = 0.5
    sample_nll_p = partial(sample_nll, scale=sigma)
    sample_nll_p.__name__ = "sample_nll_p"
    sample_nll_p.__module__ = sample_nll.__module__
    prior_dist=tfd.Normal(loc=0.,scale=1.)

    seeds = np.arange(2, 41, 2)
    df = None
    samples_all = {}
    for s in seeds:
        print(f"------ Run exp with seed {s} ------")
        models = initialize_models(s, prior_dist)
        train(models, data, epochs=1000)
        kl_summary, samples = eval(models, data, prior_dist)
        samples_all[s] = samples
        if df is None:
            df = pd.DataFrame(kl_summary, index=[s])
        else:
            df_sub = pd.DataFrame(kl_summary, index=[s])
            df = pd.concat([df, df_sub])
    df = df.reset_index().rename(columns={"index": "seed"})
    print(df)
    df.to_csv(f"{os.path.splitext(__file__)[0]}_kl26.csv")
    np.savez(f"{os.path.splitext(__file__)[0]}_samples26.npz", samples=samples_all)

# %%
