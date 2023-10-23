from functools import partial

import jax
import jax.numpy as jnp


# 3d time-independent helmholtz exact u
@partial(jax.jit, static_argnums=(0, 1, 2,))
def helmholtz3d_exact_u(a1, a2, a3, x, y, z):
    return jnp.sin(a1*jnp.pi*x) * jnp.sin(a2*jnp.pi*y) * jnp.sin(a3*jnp.pi*z)


# 3d time-independent helmholtz source term
@partial(jax.jit, static_argnums=(0, 1, 2,))
def helmholtz3d_source_term(a1, a2, a3, x, y, z, lda=1.):
    u_gt = helmholtz3d_exact_u(a1, a2, a3, x, y, z)
    uxx = -(a1*jnp.pi)**2 * u_gt
    uyy = -(a2*jnp.pi)**2 * u_gt
    uzz = -(a3*jnp.pi)**2 * u_gt
    return uxx + uyy + uzz + lda*u_gt


# 2d time-dependent klein-gordon exact u
def klein_gordon3d_exact_u(t, x, y, k):
    return (x + y) * jnp.cos(k * t) + (x * y) * jnp.sin(k * t)


# 2d time-dependent klein-gordon source term
def klein_gordon3d_source_term(t, x, y, k):
    u = klein_gordon3d_exact_u(t, x, y, k)
    return u**2 - (k**2)*u


# 3d time-dependent klein-gordon exact u
def klein_gordon4d_exact_u(t, x, y, z, k):
    return (x + y + z) * jnp.cos(k*t) + (x * y * z) * jnp.sin(k*t)


# 3d time-dependent klein-gordon source term
def klein_gordon4d_source_term(t, x, y, z, k):
    u = klein_gordon4d_exact_u(t, x, y, z, k)
    return u**2 - (k**2)*u


# 3d time-dependent navier-stokes forcing term
def navier_stokes4d_forcing_term(t, x, y, z, nu):
    # forcing terms in the PDE
    # f_x = -24*jnp.exp(-18*nu*t)*jnp.sin(2*y)*jnp.cos(2*y)*jnp.sin(z)*jnp.cos(z)
    f_x = -6*jnp.exp(-18*nu*t)*jnp.sin(4*y)*jnp.sin(2*z)
    # f_y = -24*jnp.exp(-18*nu*t)*jnp.sin(2*x)*jnp.cos(2*x)*jnp.sin(z)*jnp.cos(z)
    f_y = -6*jnp.exp(-18*nu*t)*jnp.sin(4*x)*jnp.sin(2*z)
    # f_z = 24*jnp.exp(-18*nu*t)*jnp.sin(2*x)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.cos(2*y)
    f_z = 6*jnp.exp(-18*nu*t)*jnp.sin(4*x)*jnp.sin(4*y)
    return f_x, f_y, f_z


# 3d time-dependent navier-stokes exact vorticity
def navier_stokes4d_exact_w(t, x, y, z, nu):
    # analytic form of vorticity
    w_x = -3*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.cos(z)
    w_y = 6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.cos(z)
    w_z = -6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.cos(2*y)*jnp.sin(z)
    return w_x, w_y, w_z


# 3d time-dependent navier-stokes exact velocity
def navier_stokes4d_exact_u(t, x, y, z, nu):
    # analytic form of velocity
    u_x = 2*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.sin(z)
    u_y = -1*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.sin(z)
    u_z = -2*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.sin(2*y)*jnp.cos(z)
    return u_x, u_y, u_z


# 3d time-dependent flow-mixing exact solution
def flow_mixing3d_exact_u(t, x, y, omega):
    return -jnp.tanh((y/2)*jnp.cos(omega*t) - (x/2)*jnp.sin(omega*t))


# 3d time-dependent flow-mixing parameters
def flow_mixing3d_params(t, x, y, v_max, require_ab=False):
    # t, x, y must be meshgrid
    r = jnp.sqrt(x**2 + y**2)
    v_t = ((1/jnp.cosh(r))**2) * jnp.tanh(r)
    omega = (1/r)*(v_t/v_max)
    a, b = None, None
    if require_ab:
        a = -(v_t/v_max)*(y/r)
        b = (v_t/v_max)*(x/r)
    return omega, a, b