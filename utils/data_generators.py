import os
from utils.data_utils import *

import jax
import scipy.io


#========================== diffusion equation 3-d =========================#
#---------------------------------- PINN -----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_diffusion3d(nc, key):
    keys = jax.random.split(key, 13)
    ni, nb = nc**2, nc**2

    # colocation points
    tc = jax.random.uniform(keys[0], (nc**3, 1), minval=0., maxval=1.)
    xc = jax.random.uniform(keys[1], (nc**3, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc**3, 1), minval=-1., maxval=1.)
    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-1., maxval=1.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-1., maxval=1.)
    ui = 0.25 * jnp.exp(-((xi - 0.3)**2 + (yi - 0.2)**2) / 0.1) + \
         0.4 * jnp.exp(-((xi + 0.5)**2 + (yi + 0.1)**2) * 15) + \
         0.3 * jnp.exp(-(xi**2 + (yi + 0.5)**2) * 20)
    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=1.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=1.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=1.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=1.)
    ]
    xb = [
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[10], (nb, 1), minval=-1., maxval=1.)
    ]
    yb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[12], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb)
    ]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb


#---------------------------------- SPINN ----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_diffusion3d(nc, key):
    keys = jax.random.split(key, 3)
    # colocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=1.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-1., maxval=1.)
    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    xi_mesh, yi_mesh = jnp.meshgrid(xi.ravel(), yi.ravel(), indexing='ij')
    ui = 0.25 * jnp.exp(-((xi_mesh - 0.3)**2 + (yi_mesh - 0.2)**2) / 0.1) + \
         0.4 * jnp.exp(-((xi_mesh + 0.5)**2 + (yi_mesh + 0.1)**2) * 15) + \
         0.3 * jnp.exp(-(xi_mesh**2 + (yi_mesh + 0.5)**2) * 20)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-1.]]), jnp.array([[1.]]), xc, xc]
    yb = [yc, yc, jnp.array([[-1.]]), jnp.array([[1.]])]
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb


#========================== Helmholtz equation 3-d =========================#
#---------------------------------- PINN -----------------------------------#
@partial(jax.jit, static_argnums=(0, 1, 2, 3,))
def _pinn_train_generator_helmholtz3d(a1, a2, a3, nc, key):
    keys = jax.random.split(key, 15)
    nb = nc**2
    # collocation points
    xc = jax.random.uniform(keys[0], (nc**3, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[1], (nc**3, 1), minval=-1., maxval=1.)
    zc = jax.random.uniform(keys[2], (nc**3, 1), minval=-1., maxval=1.)
    uc = helmholtz3d_source_term(a1, a2, a3, xc, yc, zc)
    # boundary points (hard-coded)
    xb = [
        jnp.array([[1.]]*nb),
        jnp.array([[-1.]]*nb),
        jax.random.uniform(keys[3], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[4], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[5], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[6], (nb, 1), minval=-1., maxval=1.)
    ]
    yb = [
        jax.random.uniform(keys[7], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[8], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[1.]]*nb),
        jnp.array([[-1.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[10], (nb, 1), minval=-1., maxval=1.),
    ]
    zb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[12], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[13], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[14], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[1.]]*nb),
        jnp.array([[-1.]]*nb)
    ]
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    zb = jnp.concatenate(zb)
    return xc, yc, zc, uc, xb, yb, zb



#---------------------------------- SPINN ----------------------------------#
@partial(jax.jit, static_argnums=(0, 1, 2, 3,))
def _spinn_train_generator_helmholtz3d(a1, a2, a3, nc, key):
    keys = jax.random.split(key, 3)
    # collocation points
    xc = jax.random.uniform(keys[0], (nc,), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[1], (nc,), minval=-1., maxval=1.)
    zc = jax.random.uniform(keys[2], (nc,), minval=-1., maxval=1.)
    # source term
    xcm, ycm, zcm = jnp.meshgrid(xc, yc, zc, indexing='ij')
    uc = helmholtz3d_source_term(a1, a2, a3, xcm, ycm, zcm)
    xc, yc, zc = xc.reshape(-1, 1), yc.reshape(-1, 1), zc.reshape(-1, 1)
    # boundary (hard-coded)
    xb = [jnp.array([[1.]]), jnp.array([[-1.]]), xc, xc, xc, xc]
    yb = [yc, yc, jnp.array([[1.]]), jnp.array([[-1.]]), yc, yc]
    zb = [zc, zc, zc, zc, jnp.array([[1.]]), jnp.array([[-1.]])]
    return xc, yc, zc, uc, xb, yb, zb


#======================== Klein-Gordon equation 3-d ========================#
#---------------------------------- PINN -----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_klein_gordon3d(nc, k, key):
    ni, nb = nc**2, nc**2
    keys = jax.random.split(key, 13)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc**3, 1), minval=0., maxval=10.)
    xc = jax.random.uniform(keys[1], (nc**3, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc**3, 1), minval=-1., maxval=1.)
    uc = klein_gordon3d_source_term(tc, xc, yc, k)
    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-1., maxval=1.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-1., maxval=1.)
    ui = klein_gordon3d_exact_u(ti, xi, yi, k)
    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=10.)
    ]
    xb = [
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[10], (nb, 1), minval=-1., maxval=1.)
    ]
    yb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[12], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb)
    ]
    ub = []
    for i in range(4):
        ub += [klein_gordon3d_exact_u(tb[i], xb[i], yb[i], k)]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    ub = jnp.concatenate(ub)
    return tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub


#---------------------------------- SPINN ----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_klein_gordon3d(nc, k, key):
    keys = jax.random.split(key, 3)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=10.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-1., maxval=1.)
    tc_mesh, xc_mesh, yc_mesh = jnp.meshgrid(tc.ravel(), xc.ravel(), yc.ravel(), indexing='ij')
    uc = klein_gordon3d_source_term(tc_mesh, xc_mesh, yc_mesh, k)
    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    ti_mesh, xi_mesh, yi_mesh = jnp.meshgrid(ti.ravel(), xi.ravel(), yi.ravel(), indexing='ij')
    ui = klein_gordon3d_exact_u(ti_mesh, xi_mesh, yi_mesh, k)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-1.]]), jnp.array([[1.]]), xc, xc]
    yb = [yc, yc, jnp.array([[-1.]]), jnp.array([[1.]])]
    ub = []
    for i in range(4):
        tb_mesh, xb_mesh, yb_mesh = jnp.meshgrid(tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), indexing='ij')
        ub += [klein_gordon3d_exact_u(tb_mesh, xb_mesh, yb_mesh, k)]
    return tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub


#======================== Klein-Gordon equation 4-d ========================#
#---------------------------------- PINN -----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_klein_gordon4d(nc, k, key):
    ni, nb = nc**3, nc**3
    keys = jax.random.split(key, 24)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc**4, 1), minval=0., maxval=10.)
    xc = jax.random.uniform(keys[1], (nc**4, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc**4, 1), minval=-1., maxval=1.)
    zc = jax.random.uniform(keys[3], (nc**4, 1), minval=-1., maxval=1.)
    uc = klein_gordon4d_source_term(tc, xc, yc, zc, k)
    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[4], (ni, 1), minval=-1., maxval=1.)
    yi = jax.random.uniform(keys[5], (ni, 1), minval=-1., maxval=1.)
    zi = jax.random.uniform(keys[6], (ni, 1), minval=-1., maxval=1.)
    ui = klein_gordon4d_exact_u(ti, xi, yi, zi, k)
    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[9], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[10], (nb, 1), minval=0., maxval=10.),
        jax.random.uniform(keys[11], (nb, 1), minval=0., maxval=10.)
    ]
    xb = [
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb),
        jax.random.uniform(keys[12], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[13], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[14], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[15], (nb, 1), minval=-1., maxval=1.)
    ]
    yb = [
        jax.random.uniform(keys[16], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[17], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb),
        jax.random.uniform(keys[18], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[19], (nb, 1), minval=-1., maxval=1.)
    ]
    zb = [
        jax.random.uniform(keys[20], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[21], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[22], (nb, 1), minval=-1., maxval=1.),
        jax.random.uniform(keys[23], (nb, 1), minval=-1., maxval=1.),
        jnp.array([[-1.]]*nb),
        jnp.array([[1.]]*nb),
    ]
    ub = []
    for i in range(6):
        ub += [klein_gordon4d_exact_u(tb[i], xb[i], yb[i], zb[i], k)]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    zb = jnp.concatenate(zb)
    ub = jnp.concatenate(ub)
    return tc, xc, yc, zc, uc, ti, xi, yi, zi, ui, tb, xb, yb, zb, ub


#---------------------------------- SPINN ----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_klein_gordon4d(nc, k, key):
    keys = jax.random.split(key, 4)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=10.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-1., maxval=1.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-1., maxval=1.)
    zc = jax.random.uniform(keys[3], (nc, 1), minval=-1., maxval=1.)
    tcm, xcm, ycm, zcm = jnp.meshgrid(
        tc.ravel(), xc.ravel(), yc.ravel(), zc.ravel(), indexing='ij'
    )
    uc = klein_gordon4d_source_term(tcm, xcm, ycm, zcm, k)
    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    zi = zc
    tim, xim, yim, zim = jnp.meshgrid(
        ti.ravel(), xi.ravel(), yi.ravel(), zi.ravel(), indexing='ij'
    )
    ui = klein_gordon4d_exact_u(tim, xim, yim, zim, k)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc, tc, tc]
    xb = [jnp.array([[-1.]]), jnp.array([[1.]]), xc, xc, xc, xc]
    yb = [yc, yc, jnp.array([[-1.]]), jnp.array([[1.]]), yc, yc]
    zb = [zc, zc, zc, zc, jnp.array([[-1.]]), jnp.array([[1.]])]
    ub = []
    for i in range(6):
        tbm, xbm, ybm, zbm = jnp.meshgrid(
            tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), zb[i].ravel(), indexing='ij'
        )
        ub += [klein_gordon4d_exact_u(tbm, xbm, ybm, zbm, k)]
    return tc, xc, yc, zc, uc, ti, xi, yi, zi, ui, tb, xb, yb, zb, ub


#======================== Navier-Stokes equation 3-d ========================#
#---------------------------------- SPINN -----------------------------------#
def _spinn_train_generator_navier_stokes3d(nt, nxy, data_dir, result_dir, marching_steps, step_idx, offset_num, key):
    keys = jax.random.split(key, 2)
    gt_data = scipy.io.loadmat(os.path.join(data_dir, 'w_data.mat'))
    t = gt_data['t']

    # initial points
    ti = jnp.zeros((1, 1))
    xi = gt_data['x']
    yi = gt_data['y']
    if step_idx == 0:
        # get data from ground truth
        w0 = gt_data
    else:
        # get data from previous time window prediction
        w0 = scipy.io.loadmat(os.path.join(result_dir, '..', f'IC_pred/w0_{step_idx}.mat'))
        ti = w0['t']

    # collocation points
    tc = jnp.expand_dims(jnp.linspace(start=0., stop=t[0][-1], num=nt, endpoint=False), axis=1)
    xc = jnp.expand_dims(jnp.linspace(start=0., stop=2.*jnp.pi, num=nxy, endpoint=False), axis=1)
    yc = jnp.expand_dims(jnp.linspace(start=0., stop=2.*jnp.pi, num=nxy, endpoint=False), axis=1)

    if marching_steps != 0:
        # when using time marching
        Dt = t[0][-1] / marching_steps  # interval of a single time window
        # generate temporal coordinates within current time window
        if step_idx == 0:
            tc = jnp.expand_dims(jnp.linspace(start=0., stop=Dt*(step_idx+1), num=nt, endpoint=False), axis=1)
        else:
            tc = jnp.expand_dims(jnp.linspace(start=w0['t'][0][0], stop=Dt*(step_idx+1), num=nt, endpoint=False), axis=1)

    # for stacking multi-input grid
    tc_mult = jnp.expand_dims(tc, axis=0)
    xc_mult = jnp.expand_dims(xc, axis=0)
    yc_mult = jnp.expand_dims(yc, axis=0)

    # maximum value of offsets
    dt = tc[1][0] - tc[0][0]
    dxy = xc[1][0] - xc[0][0]

    # create offset values (zero is included by default)
    offset_t = jax.random.uniform(keys[0], (offset_num-1,), minval=0., maxval=dt)
    offset_xy = jax.random.uniform(keys[1], (offset_num-1,), minval=0., maxval=dxy)

    # make multi-grid
    for i in range(offset_num-1):
        tc_mult = jnp.concatenate((tc_mult, jnp.expand_dims(tc + offset_t[i], axis= 0)), axis=0)
        xc_mult = jnp.concatenate((xc_mult, jnp.expand_dims(xc + offset_xy[i], axis=0)), axis=0)
        yc_mult = jnp.concatenate((yc_mult, jnp.expand_dims(yc + offset_xy[i], axis=0)), axis=0)

    return tc_mult, xc_mult, yc_mult, ti, xi, yi, w0['w0'], w0['u0'], w0['v0']


#======================== Navier-Stokes equation 4-d ========================#
#---------------------------------- SPINN -----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_navier_stokes4d(nc, nu, key):
    keys = jax.random.split(key, 4)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=5.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=0., maxval=2.*jnp.pi)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=0., maxval=2.*jnp.pi)
    zc = jax.random.uniform(keys[3], (nc, 1), minval=0., maxval=2.*jnp.pi)

    tcm, xcm, ycm, zcm = jnp.meshgrid(
        tc.ravel(), xc.ravel(), yc.ravel(), zc.ravel(), indexing='ij'
    )
    fc = navier_stokes4d_forcing_term(tcm, xcm, ycm, zcm, nu)

    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    zi = zc
    tim, xim, yim, zim = jnp.meshgrid(
        ti.ravel(), xi.ravel(), yi.ravel(), zi.ravel(), indexing='ij'
    )
    wi = navier_stokes4d_exact_w(tim, xim, yim, zim, nu)
    ui = navier_stokes4d_exact_u(tim, xim, yim, zim, nu)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc, tc, tc]
    xb = [jnp.array([[-1.]]), jnp.array([[1.]]), xc, xc, xc, xc]
    yb = [yc, yc, jnp.array([[-1.]]), jnp.array([[1.]]), yc, yc]
    zb = [zc, zc, zc, zc, jnp.array([[-1.]]), jnp.array([[1.]])]
    wb = []
    for i in range(6):
        tbm, xbm, ybm, zbm = jnp.meshgrid(
            tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), zb[i].ravel(), indexing='ij'
        )
        wb += [navier_stokes4d_exact_w(tbm, xbm, ybm, zbm, nu)]
    return tc, xc, yc, zc, fc, ti, xi, yi, zi, wi, ui, tb, xb, yb, zb, wb


#======================== Flow-Mixing 3-d ========================#
#----------------------------- PINN ------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_flow_mixing3d(nc, v_max, key):
    ni, nb = nc**2, nc**2

    keys = jax.random.split(key, 13)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc**3, 1), minval=0., maxval=4.)
    xc = jax.random.uniform(keys[1], (nc**3, 1), minval=-4., maxval=4.)
    yc = jax.random.uniform(keys[2], (nc**3, 1), minval=-4., maxval=4.)
    _, a, b = flow_mixing3d_params(tc, xc, yc, v_max, require_ab=True)

    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-4., maxval=4.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-4., maxval=4.)
    omega_i, _, _ = flow_mixing3d_params(ti, xi, yi, v_max)
    ui = flow_mixing3d_exact_u(ti, xi, yi, omega_i)

    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=4.)
    ]
    xb = [
        jnp.array([[-4.]]*nb),
        jnp.array([[4.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-4., maxval=4.),
        jax.random.uniform(keys[10], (nb, 1), minval=-4., maxval=4.)
    ]
    yb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-4., maxval=4.),
        jax.random.uniform(keys[12], (nb, 1), minval=-4., maxval=4.),
        jnp.array([[-4.]]*nb),
        jnp.array([[4.]]*nb)
    ]
    ub = []
    for i in range(4):
        omega_b, _, _ = flow_mixing3d_params(tb[i], xb[i], yb[i], v_max)
        ub += [flow_mixing3d_exact_u(tb[i], xb[i], yb[i], omega_b)]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    ub = jnp.concatenate(ub)
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b


#----------------------------- SPINN -----------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_flow_mixing3d(nc, v_max, key):
    keys = jax.random.split(key, 3)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=4.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-4., maxval=4.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-4., maxval=4.)
    tc_mesh, xc_mesh, yc_mesh = jnp.meshgrid(tc.ravel(), xc.ravel(), yc.ravel(), indexing='ij')

    _, a, b = flow_mixing3d_params(tc_mesh, xc_mesh, yc_mesh, v_max, require_ab=True)

    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    ti_mesh, xi_mesh, yi_mesh = jnp.meshgrid(ti.ravel(), xi.ravel(), yi.ravel(), indexing='ij')
    omega_i, _, _ = flow_mixing3d_params(ti_mesh, xi_mesh, yi_mesh, v_max)
    ui = flow_mixing3d_exact_u(ti_mesh, xi_mesh, yi_mesh, omega_i)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-4.]]), jnp.array([[4.]]), xc, xc]
    yb = [yc, yc, jnp.array([[-4.]]), jnp.array([[4.]])]
    ub = []
    for i in range(4):
        tb_mesh, xb_mesh, yb_mesh = jnp.meshgrid(tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), indexing='ij')
        omega_b, _, _ = flow_mixing3d_params(tb_mesh, xb_mesh, yb_mesh, v_max)
        ub += [flow_mixing3d_exact_u(tb_mesh, xb_mesh, yb_mesh, omega_b)]
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b


#=========================== Poisson equation 2-d ==========================#
#---------------------------------- SPINN -----------------------------------#
def _spinn_train_generator_poisson2d(nx, key):
    keys = jax.random.split(key, 10)
    # collocation points
    xc1 = jnp.expand_dims(jnp.linspace(start=-1., stop=0., num=nx, endpoint=False), axis=1)
    yc1 = jnp.expand_dims(jnp.linspace(start=-1., stop=1., num=nx, endpoint=False), axis=1)
    xc2 = jnp.expand_dims(jnp.linspace(start=0., stop=1., num=nx, endpoint=False), axis=1)
    yc2 = jnp.expand_dims(jnp.linspace(start=-1., stop=0., num=nx, endpoint=False), axis=1)

    xb = [
        jnp.expand_dims(jnp.linspace(start=-1., stop=1., num=nx, endpoint=True), axis=1),
        jnp.array([[1.]]),
        jnp.expand_dims(jnp.linspace(start=0., stop=1., num=nx, endpoint=True), axis=1),
        jnp.array([[0.]]),
        jnp.expand_dims(jnp.linspace(start=-1., stop=0., num=nx, endpoint=True), axis=1),
        jnp.array([[-1.]]),
    ]

    yb = [
        jnp.array([[-1.]]),
        jnp.expand_dims(jnp.linspace(start=-1., stop=0., num=nx, endpoint=True), axis=1),
        jnp.array([[0.]]),
        jnp.expand_dims(jnp.linspace(start=0., stop=1., num=nx, endpoint=True), axis=1),
        jnp.array([[1.]]),
        jnp.expand_dims(jnp.linspace(start=-1., stop=1., num=nx, endpoint=True), axis=1),
    ]

    xc1_mult = jnp.expand_dims(xc1, axis=0)
    yc1_mult = jnp.expand_dims(yc1, axis=0)
    xc2_mult = jnp.expand_dims(xc2, axis=0)
    yc2_mult = jnp.expand_dims(yc2, axis=0)

    dx1 = xc1[1][0] - xc1[0][0]
    dy1 = yc1[1][0] - yc1[0][0]
    dx2 = xc2[1][0] - xc2[0][0]
    dy2 = yc2[1][0] - yc2[0][0]

    offset_x1 = jax.random.uniform(keys[0], (8-1,), minval=0., maxval=dx1)
    offset_y1 = jax.random.uniform(keys[1], (8-1,), minval=0., maxval=dy1)
    offset_x2 = jax.random.uniform(keys[2], (8-1,), minval=0., maxval=dx2)
    offset_y2 = jax.random.uniform(keys[3], (8-1,), minval=0., maxval=dy2)

    # make multi-grid
    for i in range(8-1):
        xc1_mult = jnp.concatenate((xc1_mult, jnp.expand_dims(xc1 + offset_x1[i], axis= 0)), axis=0)
        yc1_mult = jnp.concatenate((yc1_mult, jnp.expand_dims(yc1 + offset_y1[i], axis=0)), axis=0)
        xc2_mult = jnp.concatenate((xc2_mult, jnp.expand_dims(xc2 + offset_x2[i], axis= 0)), axis=0)
        yc2_mult = jnp.concatenate((yc2_mult, jnp.expand_dims(yc2 + offset_y2[i], axis=0)), axis=0)

    return xc1_mult, yc1_mult, xc2_mult, yc2_mult, xb, yb


def generate_train_data(args, key, result_dir=None):
    eqn = args.equation
    if args.model == 'pinn':
        if eqn == 'diffusion3d':
            data = _pinn_train_generator_diffusion3d(
                args.nc, key
            )
        elif eqn == 'helmholtz3d':
            data = _pinn_train_generator_helmholtz3d(
                args.a1, args.a2, args.a3, args.nc, key
            )
        elif eqn == 'klein_gordon3d':
            data = _pinn_train_generator_klein_gordon3d(
                args.nc, args.k, key
            )
        elif eqn == 'klein_gordon4d':
            data = _pinn_train_generator_klein_gordon4d(
                args.nc, args.k, key
            )
        elif eqn == 'flow_mixing3d':
            data = _pinn_train_generator_flow_mixing3d(
                args.nc, args.vmax, key
            )
        else:
            raise NotImplementedError
    elif args.model == 'spinn':
        if eqn == 'diffusion3d':
            data = _spinn_train_generator_diffusion3d(
                args.nc, key
            )
        elif eqn == 'helmholtz3d':
            data = _spinn_train_generator_helmholtz3d(
                args.a1, args.a2, args.a3, args.nc, key
            )
        elif eqn == 'klein_gordon3d':
            data = _spinn_train_generator_klein_gordon3d(
                args.nc, args.k, key
            )
        elif eqn == 'klein_gordon4d':
            data = _spinn_train_generator_klein_gordon4d(
                args.nc, args.k, key
            )
        elif eqn == 'navier_stokes3d':
            data = _spinn_train_generator_navier_stokes3d(
                args.nt, args.nxy, args.data_dir, result_dir, args.marching_steps, args.step_idx, args.offset_num, key
            )
        elif eqn == 'navier_stokes4d':
            data = _spinn_train_generator_navier_stokes4d(
                args.nc, args.nu, key
            )
        elif eqn == 'flow_mixing3d':
            data = _spinn_train_generator_flow_mixing3d(
                args.nc, args.vmax, key
            )
        elif eqn == 'poisson2d':
            data = _spinn_train_generator_poisson2d(
                args.nc, key
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return data


#============================== test dataset ===============================#
#------------------------- diffusion equation 3-d --------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_diffusion3d(model, data_dir):
    u_gt, tt = [], 0.
    for _ in range(101):
        u_gt += [jnp.load(os.path.join(data_dir, f'heat_gaussian_{tt:.2f}.npy'))]
        tt += 0.01
    u_gt = jnp.stack(u_gt)
    t = jnp.linspace(0., 1., u_gt.shape[0])
    x = jnp.linspace(-1., 1., u_gt.shape[1])
    y = jnp.linspace(-1., 1., u_gt.shape[2])
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    return t, x, y, u_gt


#------------------------- Helmholtz equation 3-d --------------------------#
@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4,))
def _test_generator_helmholtz3d(model, a1, a2, a3, nc_test):
    x = jnp.linspace(-1., 1., nc_test)
    y = jnp.linspace(-1., 1., nc_test)
    z = jnp.linspace(-1., 1., nc_test)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    z = jax.lax.stop_gradient(z)
    xm, ym, zm = jnp.meshgrid(x, y, z, indexing='ij')
    u_gt = helmholtz3d_exact_u(a1, a2, a3, xm, ym, zm)
    if model == 'pinn':
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        z = zm.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
    return x, y, z, u_gt


#----------------------- Klein-Gordon equation 3-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_klein_gordon3d(model, nc_test, k):
    t = jnp.linspace(0, 10, nc_test)
    x = jnp.linspace(-1, 1, nc_test)
    y = jnp.linspace(-1, 1, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    u_gt = klein_gordon3d_exact_u(tm, xm, ym, k)
    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    return t, x, y, u_gt


#----------------------- Klein-Gordon equation 4-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_klein_gordon4d(model, nc_test, k):
    t = jnp.linspace(0, 10, nc_test)
    x = jnp.linspace(-1, 1, nc_test)
    y = jnp.linspace(-1, 1, nc_test)
    z = jnp.linspace(-1, 1, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    z = jax.lax.stop_gradient(z)
    tm, xm, ym, zm = jnp.meshgrid(
        t, x, y, z, indexing='ij'
    )
    u_gt = klein_gordon4d_exact_u(tm, xm, ym, zm, k)
    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        z = zm.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
    return t, x, y, z, u_gt


#----------------------- Navier-Stokes equation 3-d -------------------------#
def _test_generator_navier_stokes3d(model, data_dir, result_dir, marching_steps, step_idx):
    ns_data = scipy.io.loadmat(os.path.join(data_dir, 'w_data.mat'))
    t = ns_data['t'].reshape(-1, 1)
    x = ns_data['x'].reshape(-1, 1)
    y = ns_data['y'].reshape(-1, 1)
    t = jnp.insert(t, 0, jnp.array([0.]), axis=0)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)

    gt = ns_data['w']   # without t=0
    gt = jnp.insert(gt, 0, ns_data['w0'], axis=0)

    # get data within current time window
    if marching_steps != 0:
        Dt = t[-1][0] / marching_steps  # interval of time window
        i = 0
        while Dt*(step_idx+1) > t[i][0]:
            i+=1
        t = t[:i]
        gt = gt[:i]

    # get data within current time window
    if step_idx > 0:
        w0_pred = scipy.io.loadmat(os.path.join(result_dir, '..', f'IC_pred/w0_{step_idx}.mat'))
        i = 0
        while t[i] != w0_pred['t'][0][0]:
            i+=1
        t = t[i:]
        gt = gt[i:]

    if model == 'pinn':
        tm, xm, ym = jnp.meshgrid(t.ravel(), x.ravel(), y.ravel(), indexing='ij')
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
    
    return t, x, y, gt


#----------------------- Navier-Stokes equation 4-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_navier_stokes4d(model, nc_test, nu):
    t = jnp.linspace(0, 5, nc_test)
    x = jnp.linspace(0, 2*jnp.pi, nc_test)
    y = jnp.linspace(0, 2*jnp.pi, nc_test)
    z = jnp.linspace(0, 2*jnp.pi, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    z = jax.lax.stop_gradient(z)
    tm, xm, ym, zm = jnp.meshgrid(
        t, x, y, z, indexing='ij'
    )
    w_gt = navier_stokes4d_exact_w(tm, xm, ym, zm, nu)
    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        z = zm.reshape(-1, 1)
        w_gt = w_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
    return t, x, y, z, w_gt


#----------------------- Flow-Mixing 3-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_flow_mixing3d(model, nc_test, v_max):
    t = jnp.linspace(0, 4, nc_test)
    x = jnp.linspace(-4, 4, nc_test)
    y = jnp.linspace(-4, 4, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')

    omega, _, _ = flow_mixing3d_params(tm, xm, ym, v_max)
    u_gt = flow_mixing3d_exact_u(tm, xm, ym, omega)

    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    return t, x, y, u_gt


#----------------------- Poisson 2-d -------------------------#
def _test_generator_poisson2d(model, data_dir):
    data = jnp.load(os.path.join(data_dir, 'Poisson_Lshape.npz'))
    x = data['X_test'][::161][:,0]
    y = data['X_test'][::161][:,0]
    u_gt = data['y_ref'].reshape(161, 161)
    u_gt = jnp.nan_to_num(u_gt, nan=0.)

    xm, ym = jnp.meshgrid(x, y, indexing='ij')
    if model == 'pinn':
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    # pdb.set_trace()
    return x, y, u_gt


def generate_test_data(args, result_dir):
    eqn = args.equation
    if eqn == 'diffusion3d':
        data = _test_generator_diffusion3d(
            args.model, args.data_dir
        )
    elif eqn == 'helmholtz3d':
        data = _test_generator_helmholtz3d(
            args.model, args.a1, args.a2, args.a3, args.nc_test
        )
    elif eqn == 'klein_gordon3d':
        data = _test_generator_klein_gordon3d(
            args.model, args.nc_test, args.k
        )
    elif eqn == 'klein_gordon4d':
        data = _test_generator_klein_gordon4d(
            args.model, args.nc_test, args.k
        )
    elif eqn == 'navier_stokes3d':
        data = _test_generator_navier_stokes3d(
            args.model, args.data_dir, result_dir, args.marching_steps, args.step_idx
        )
    elif eqn == 'navier_stokes4d':
        data = _test_generator_navier_stokes4d(
            args.model, args.nc_test, args.nu
        )
    elif eqn == 'flow_mixing3d':
        data = _test_generator_flow_mixing3d(
            args.model, args.nc_test, args.vmax
        )
    elif eqn == 'poisson2d':
        data = _test_generator_poisson2d(
            args.model, args.data_dir
        )
    else:
        raise NotImplementedError
    return data