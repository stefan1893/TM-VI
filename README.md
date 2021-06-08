# Transformation Models for Flexible Posteriors in Variational Bayes

**Authors:** Stefan Hörtling, Daniel Dold, Oliver Dürr, Beate Sick

**Description:** The repository contains code for the conducted experiments and for a customized Keras layer for T;-VI.

## Experiments

|No.|Experiment| Motivation |
|--|--|--|
|01|Simple regression  | - Demonstrate basic functionality <br /> - Comparison of the approaches
|02|Single weight behavior  | - Prove the ability of VIMLTS to fit a multimodal posterior according to MCMC <br /> - Independence assumption of MFVI should not play a role
|03|Small and shallow networks | Check assumption: posterior approximations could be complex in small and shallow networks
|04|Going deeper | Check behavior in deeper BNNs


## Customized Keras layer for TM-VI
To use the TM-VI Keras layer, you have to import the 

>  src.vimlts_keras.py

file to your Notebook and create a layer instance for your architecture.
Please see the *Small and shallow networks* or the *Going deeper* experiment for an example.

**Example:**

    from src.vimlts_keras import DenseVIMLTS

    x_in = Input(shape=(1,),name="VIMLTS_il")
    x_arch = DenseVIMLTS(units=num_hidden_units, num_samples_per_epoch=num_samples_per_epoch, activation='relu', kl_weight=kl_weight, name="VIMLTS_hl_1", **prior_params)(x_in)
    x_arch = DenseVIMLTS(units=1, num_samples_per_epoch=num_samples_per_epoch, kl_weight=kl_weight, name="VIMLTS_ol", **prior_params)(x_arch)
    
    model_VIMLTS = Model(x_in, x_arch,name="model_VIMLTS")
