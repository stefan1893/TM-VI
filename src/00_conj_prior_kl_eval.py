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

sns.set(style="whitegrid", font_scale=2.25)
# plt.rcParams.update({"text.usetex": True})
import sys

sys.path.append('../')
from src.vimlts_fast import VimltsLinear, ConjungateDenseViGauss

print(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('Tensorflow version: ', tf.__version__, )


# %%


def scheduler(epoch, lr, lr_start, lr_stop, epochs):
    return lr_start + (lr_stop - lr_start) * (epoch / epochs)


@tf.function
def sample_bernoulli_nll(y_obs, y_pred):
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

    dist = tfp.distributions.Bernoulli(probs=y_pred)
    nll_per_sample = -dist.log_prob(y_obs)
    nlls = tf.reduce_mean(nll_per_sample, axis=0)
    tf.debugging.check_numerics(nlls, "NLL contains NaNs or Infs")
    return tf.reduce_sum(nlls)
    # return 0.


def softplus_inv(y):
    return np.log(np.exp(y) - 1)


# %%

def initialize_models(seed=2, prior_dist=tfd.Beta(concentration1=1.1, concentration0=1.1)):
    models = {}
    # Ms = [1, 3, 10, 30, 100, 300]
    Ms = [2, 4, 5, 6, 7, 8, 9, 15]
    for M in Ms:
        # init params
        kernel_initializers = dict(kernel_init_alpha_w=initializers.RandomNormal(mean=1.5),
                                   kernel_init_beta_w=initializers.RandomNormal(),
                                   kernel_init_alpha_z=initializers.RandomNormal(mean=1),
                                   kernel_init_beta_z=initializers.RandomNormal(),
                                   kernel_init_thetas=[initializers.RandomNormal(mean=-1.5, stddev=.3)] + [
                                       initializers.RandomNormal(mean=softplus_inv((2 + 1.5) / M), stddev=.5) for i in
                                       range(M)])
        # define model
        tf.random.set_seed(seed)
        np.random.seed(seed)
        layer = VimltsLinear(1,
                             activation=tfp.bijectors.Sigmoid(low=1e-6, high=1. - 1e-6),
                             **kernel_initializers,
                             num_samples=10000,
                             prior_dist=prior_dist,
                             input_shape=(1,), )
        model = tf.keras.Sequential([layer], name=f"TM-VI-degree{M}")
        model.build(input_shape=(None, 1))
        models[f"TM-VI M={M}"] = model

    tf.random.set_seed(seed)
    np.random.seed(seed)
    vi_gauss_l = ConjungateDenseViGauss(1,
                                        activation=tfp.bijectors.Sigmoid(low=1e-6, high=1. - 1e-6),
                                        num_samples=10000,
                                        kernel_init_mu_w=initializers.Constant(0.),
                                        kernel_init_rhosigma_w=initializers.Constant(0.2),
                                        prior_dist=prior_dist)
    vi_gauss = tf.keras.Sequential([vi_gauss_l], name="Gauss-VI")
    vi_gauss.build(input_shape=(None, 1))
    vi_gauss.summary()
    models["Gauss-VI"] = vi_gauss
    return models


class LogKL(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epochs'] = epoch


def train(models, data, epochs):
    for name, model in models.items():
        print(f"Start experiment with model {name}")
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(
        #     partial(scheduler, lr_start=0.1, lr_stop=0.025, epochs=epochs))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=sample_bernoulli_nll,
                      run_eagerly=False)
        model.fit(tf.ones((len(data), 1)), data, epochs=epochs, verbose=False, callbacks=[LogKL()])


def eval(models, post_analytic):
    kl_summary = {}
    samples = {}
    for name, model in models.items():
        layer = model.layers[0]
        pi, log_q_pi = layer.sample(num=100000, bijector=tfp.bijectors.Sigmoid(low=1e-6, high=1. - 1e-6))
        kl = tf.reduce_mean(log_q_pi - post_analytic.log_prob(pi), axis=0)
        print(f"{name} KL: {kl.numpy().squeeze():.2e}")
        kl_summary[name] = kl.numpy().squeeze()
        samples[name] = dict(pi=pi.numpy(), log_q_pi=log_q_pi.numpy())
    return kl_summary, samples


# %%
if __name__ == '__main__':
    data = np.ones((2))
    data = data.reshape(-1, 1)
    alpha = 1.1
    beta = 1.1
    prior_dist = tfd.Beta(concentration1=alpha, concentration0=beta)
    alpha_post = alpha + np.sum(data)
    beta_post = beta + len(data) - np.sum(data)
    post_analytic = tfd.Beta(alpha_post.astype(np.float32), beta_post.astype(np.float32))

    seeds = np.arange(2, 41, 2)
    df = None
    samples_all = {}
    for s in seeds:
        print(f"------ Run exp with seed {s} ------")
        models = initialize_models(s, prior_dist)
        train(models, data, epochs=1000)
        kl_summary, samples = eval(models, post_analytic)
        samples_all[s] = samples
        if df is None:
            df = pd.DataFrame(kl_summary, index=[s])
        else:
            df_sub = pd.DataFrame(kl_summary, index=[s])
            df = pd.concat([df, df_sub])
    df = df.reset_index().rename(columns={"index": "seed"})
    print(df)
    df.to_csv(f"{os.path.splitext(__file__)[0]}_kl.csv")
    np.savez(f"{os.path.splitext(__file__)[0]}_samples.npz", samples=samples_all)

# %%
