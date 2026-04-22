import numpy as onp
from jax import numpy as np, random as jr, tree as jtu
from dorito.stats import apply_regularisers




def ff_reg(model, exposure, ff_std=0.035):
    ff_norm = model.FF - 1
    return -np.mean((ff_norm / ff_std) ** 2)

def nl_reg(model, exposure, nl_std=0.025):
    nl_norm = model.non_linearity - model.non_linearity.mean()
    return -np.mean((nl_norm / nl_std) ** 2)

def loss_fn(model, exposure, args={}):

    # calculating likelihood
    likelihood = -np.nanmean(exposure.mv_zscore(model))

    # applying regularisers to calculate prior
    prior = apply_regularisers(model, exposure, args)

    # summing to posterior
    posterior = likelihood + prior

    aux = (likelihood, prior)
    return posterior, aux


def args_fn(model, args, epoch):
    """
    Custom args function to handle the learning
    rate warm-up and temperature decay.
    """
    args["l2"] = args["l2_schedule"][epoch]
    return model, args


def cosine_warmup(t, t0, n_max):
    # Make the cosine curve
    x = (t - t0) * np.pi / n_max
    half_cos = 0.5 * (1 + (np.cos(x + np.pi)))

    # Set all values > n to 1.
    half_cos = np.where(t > n_max + t0, 1, half_cos)
    half_cos = np.where(t < t0, 0.0, half_cos)
    return half_cos


def temp_decay(T0, k, t):
    return T0 * np.exp(-k * t)


def get_warmup(args):
    return cosine_warmup(args["t"], args["t0"], args["n_max"])


def get_temperature(args):
    return temp_decay(args["T0"], args["k"], args["t"])


def grads_fn(model, grads, args):

    if "nn_weights" not in grads.params.keys():
        return grads, args

    # Get the parameters
    grad_params = grads.params

    # Get the key and update args with new key
    key, subkey = jr.split(args["key"], 2)
    args["key"] = subkey

    # Adds a temperature to the NN gradients
    values = grad_params["nn_weights"]

    # Add the learning rate warm-up (we also warm up the temperature here)
    values *= args["max_lr"] * get_warmup(args)

    rand_vals = get_temperature(args) * jr.normal(key, values.shape)
    values += rand_vals

    # Increment the t parameter
    args["t"] += 1.0 / args["n_batch"]

    # Update with the new values
    grad_params["nn_weights"] = values
    grads = grads.set("params", grad_params)
    return grads, args

def aux_fn(batch_key, aux_dict, aux):
    # Aux should have exposure keys, with values (likelihood, prior)
    for exp_key, val in aux.items():
        aux_key = (batch_key, exp_key)
        aux_dict["loglike"][aux_key].append(onp.array(val[0]))
        # NOTE "l2_reg" is just all priors
        # TODO Edit Trainer class so this is not hardcoded
        aux_dict["l2_reg"][aux_key].append(onp.array(val[1]))
    return aux_dict

def looper_fn(loss_dict, aux_dict):

    cal_losses, flat_losses, val_losses = {}, {}, {}
    for key, value in aux_dict["loglike"].items():
        batch_key, exp_key = key
        if "cal" in batch_key:
            cal_losses[key] = value
        if "flat" in batch_key:
            flat_losses[key] = value
        if "val" in batch_key:
            val_losses[key] = value

    print_str = ""
    if len(cal_losses) > 0:
        print_str += "Cal: "

        losses = np.array(list(cal_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    if len(val_losses) > 0:
        print_str += " | Val: "

        losses = np.array(list(val_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    if len(flat_losses) > 0:
        print_str += " | Flat: "
        losses = np.array(list(flat_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    # NOTE "l2_reg" is just all priors
    # TODO Edit Trainer class so this is not hardcoded
    prior = np.array(jtu.leaves(aux_dict["l2_reg"]))
    if len(prior) > 0:
        print_str += f" | Prior: {prior[-1]:.2f}"
        if len(prior) > 1:
            print_str += f" \u0394 {np.diff(prior)[-1]:.2f}"

    return print_str