import jax.numpy as jnp
from jax import jvp, vjp


def velocity_to_vorticity_fwd(apply_fn, params, t, x, y):
    # t, x, y, _ = train_data
    vec = jnp.ones(x.shape)
    # w = v_x - u_y
    v_x = jvp(lambda x: apply_fn(params, t, x, y)[1], (x,), (vec,))[1]
    u_y = jvp(lambda y: apply_fn(params, t, x, y)[0], (y,), (vec,))[1]
    
    return v_x - u_y


def velocity_to_vorticity_rev(apply_fn, params, t, x, y):    
    # w = v_x - u_y
    v, vjp_fn = vjp(lambda x: apply_fn(params, t, x, y)[..., 1], x)
    v_x = vjp_fn(jnp.ones(v.shape))[0]
    u, vjp_fn = vjp(lambda x: apply_fn(params, t, x, y)[..., 0], y)
    u_y = vjp_fn(jnp.ones(u.shape))[0]
    
    return v_x - u_y


def vorx(apply_fn, params, t, x, y, z):
    # vorticity vector w/ forward-mode AD
    # w_x = uz_y - uy_z
    vec_z = jnp.ones(z.shape)
    vec_y = jnp.ones(y.shape)
    uy_z = jvp(lambda z: apply_fn(params, t, x, y, z)[1], (z,), (vec_z,))[1]
    uz_y = jvp(lambda y: apply_fn(params, t, x, y, z)[2], (y,), (vec_y,))[1]
    wx = uz_y - uy_z
    return wx


def vory(apply_fn, params, t, x, y, z):
    # vorticity vector w/ forward-mode AD
    # w_y = ux_z - uz_x
    vec_z = jnp.ones(z.shape)
    vec_x = jnp.ones(x.shape)
    ux_z = jvp(lambda z: apply_fn(params, t, x, y, z)[0], (z,), (vec_z,))[1]
    uz_x = jvp(lambda x: apply_fn(params, t, x, y, z)[2], (x,), (vec_x,))[1]
    wy = ux_z - uz_x
    return wy


def vorz(apply_fn, params, t, x, y, z):
    # vorticity vector w/ forward-mode AD
    # w_z = uy_x - ux_y
    vec_y = jnp.ones(y.shape)
    vec_x = jnp.ones(x.shape)
    ux_y = jvp(lambda y: apply_fn(params, t, x, y, z)[0], (y,), (vec_y,))[1]
    uy_x = jvp(lambda x: apply_fn(params, t, x, y, z)[1], (x,), (vec_x,))[1]
    wz = uy_x - ux_y
    return wz