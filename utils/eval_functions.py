import jax
import jax.numpy as jnp
from functools import partial
from utils.vorticity import velocity_to_vorticity_fwd, velocity_to_vorticity_rev, vorx, vory, vorz
import pdb


def relative_l2(u, u_gt):
    return jnp.linalg.norm(u-u_gt) / jnp.linalg.norm(u_gt)

def mse(u, u_gt):
    return jnp.mean((u-u_gt)**2)

@partial(jax.jit, static_argnums=(0,))
def _eval2d(apply_fn, params, *test_data):
    x, y, u_gt = test_data
    return relative_l2(apply_fn(params, x, y), u_gt)

@partial(jax.jit, static_argnums=(0,))
def _eval2d_mask(apply_fn, mask, params, *test_data):
    x, y, u_gt = test_data
    nx, ny = u_gt.shape
    pred = apply_fn(params, x, y).reshape(nx, ny)
    pred = pred * mask
    return relative_l2(pred, u_gt.reshape(nx, ny))


@partial(jax.jit, static_argnums=(0,))
def _eval3d(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = apply_fn(params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_pinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_rev(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_spinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_fwd(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval4d(apply_fn, params, *test_data):
    t, x, y, z, u_gt = test_data
    return relative_l2(apply_fn(params, t, x, y, z), u_gt)

@partial(jax.jit, static_argnums=(0,))
def _eval_ns4d(apply_fn, params, *test_data):
    t, x, y, z, w_gt = test_data
    error = 0
    wx = vorx(apply_fn, params, t, x, y, z)
    wy = vory(apply_fn, params, t, x, y, z)
    wz = vorz(apply_fn, params, t, x, y, z)
    error = relative_l2(wx, w_gt[0]) + relative_l2(wy, w_gt[1]) + relative_l2(wz, w_gt[2])
    return error / 3


# temporary code
def _batch_eval4d(apply_fn, params, *test_data):
    t, x, y, z, u_gt = test_data
    error, batch_size = 0., 100000
    n_iters = len(u_gt) // batch_size
    for i in range(n_iters):
        begin, end = i*batch_size, (i+1)*batch_size
        u = apply_fn(params, t[begin:end], x[begin:end], y[begin:end], z[begin:end])
        error += jnp.sum((u - u_gt[begin:end])**2)
    error = jnp.sqrt(error) / jnp.linalg.norm(u_gt)
    return error

@partial(jax.jit, static_argnums=(0,))
def _evalnd(apply_fn, params, *test_data):
    t, x_list, u_gt = test_data
    return relative_l2(apply_fn(params, t, *x_list), u_gt)


def setup_eval_function(model, equation):
    dim = equation[-2:]
    if dim == '2d':
        if equation == 'poisson2d':
            fn = _eval2d_mask
        else:
            fn = _eval2d
    elif dim == '3d':
        if model == 'pinn' and equation == 'navier_stokes3d':
            fn = _eval3d_ns_pinn
        elif model == 'spinn' and equation == 'navier_stokes3d':
            fn = _eval3d_ns_spinn
        else:
            fn = _eval3d
    elif dim == '4d':
        if model == 'pinn':
            fn = _batch_eval4d
        if model == 'spinn' and equation == 'navier_stokes4d':
            fn = _eval_ns4d
        else:
            fn = _eval4d
    elif dim == 'nd':
        if model == 'spinn':
            fn = _evalnd
    else:
        raise NotImplementedError
    return fn