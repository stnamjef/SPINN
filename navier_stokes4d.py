import argparse
import os
import time

import jax
import numpy as np
import optax
from networks.hessian_vector_products import *
from tqdm import trange
from utils.data_generators import generate_test_data, generate_train_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import *
from utils.vorticity import vorx, vory, vorz
from utils.visualizer import show_solution


@partial(jax.jit, static_argnums=(0,))
def apply_model_spinn(apply_fn, params, nu, lbda_c, lbda_ic, *train_data):
    def residual_loss(params, t, x, y, z, f):
        # calculate u
        ux, uy, uz = apply_fn(params, t, x, y, z)
        # pdb.set_trace()
        # calculate w (3D vorticity vector)
        wx = vorx(apply_fn, params, t, x, y, z)
        wy = vory(apply_fn, params, t, x, y, z)
        wz = vorz(apply_fn, params, t, x, y, z)
        # tangent vector dx/dx
        vec_t = jnp.ones(t.shape)
        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)
        vec_z = jnp.ones(z.shape)

        # x-component
        wx_t = jvp(lambda t: vorx(apply_fn, params, t, x, y, z), (t,), (vec_t,))[1]
        wx_x, wx_xx = hvp_fwdfwd(lambda x: vorx(apply_fn, params, t, x, y, z), (x,), (vec_x,), True)
        wx_y, wx_yy = hvp_fwdfwd(lambda y: vorx(apply_fn, params, t, x, y, z), (y,), (vec_y,), True)
        wx_z, wx_zz = hvp_fwdfwd(lambda z: vorx(apply_fn, params, t, x, y, z), (z,), (vec_z,), True)
        
        ux_x = jvp(lambda x: apply_fn(params, t, x, y, z)[0], (x,), (vec_x,))[1]
        ux_y = jvp(lambda y: apply_fn(params, t, x, y, z)[0], (y,), (vec_y,))[1]
        ux_z = jvp(lambda z: apply_fn(params, t, x, y, z)[0], (z,), (vec_z,))[1]

        loss_x = jnp.mean((wx_t + ux*wx_x + uy*wx_y + uz*wx_z - \
             (wx*ux_x + wy*ux_y + wz*ux_z) - \
                nu*(wx_xx + wx_yy + wx_zz) - \
                    f[0])**2)

        # y-component
        wy_t = jvp(lambda t: vory(apply_fn, params, t, x, y, z), (t,), (vec_t,))[1]
        wy_x, wy_xx = hvp_fwdfwd(lambda x: vory(apply_fn, params, t, x, y, z), (x,), (vec_x,), True)
        wy_y, wy_yy = hvp_fwdfwd(lambda y: vory(apply_fn, params, t, x, y, z), (y,), (vec_y,), True)
        wy_z, wy_zz = hvp_fwdfwd(lambda z: vory(apply_fn, params, t, x, y, z), (z,), (vec_z,), True)
        
        uy_x = jvp(lambda x: apply_fn(params, t, x, y, z)[1], (x,), (vec_x,))[1]
        uy_y = jvp(lambda y: apply_fn(params, t, x, y, z)[1], (y,), (vec_y,))[1]
        uy_z = jvp(lambda z: apply_fn(params, t, x, y, z)[1], (z,), (vec_z,))[1]

        loss_y = jnp.mean((wy_t + ux*wy_x + uy*wy_y + uz*wy_z - \
             (wx*uy_x + wy*uy_y + wz*uy_z) - \
                nu*(wy_xx + wy_yy + wy_zz) - \
                    f[1])**2)

        # z-component
        wz_t = jvp(lambda t: vorz(apply_fn, params, t, x, y, z), (t,), (vec_t,))[1]
        wz_x, wz_xx = hvp_fwdfwd(lambda x: vorz(apply_fn, params, t, x, y, z), (x,), (vec_x,), True)
        wz_y, wz_yy = hvp_fwdfwd(lambda y: vorz(apply_fn, params, t, x, y, z), (y,), (vec_y,), True)
        wz_z, wz_zz = hvp_fwdfwd(lambda z: vorz(apply_fn, params, t, x, y, z), (z,), (vec_z,), True)
        
        uz_x = jvp(lambda x: apply_fn(params, t, x, y, z)[2], (x,), (vec_x,))[1]
        uz_y = jvp(lambda y: apply_fn(params, t, x, y, z)[2], (y,), (vec_y,))[1]
        uz_z = jvp(lambda z: apply_fn(params, t, x, y, z)[2], (z,), (vec_z,))[1]

        loss_z = jnp.mean((wz_t + ux*wz_x + uy*wz_y + uz*wz_z - \
             (wx*uz_x + wy*uz_y + wz*uz_z) - \
                nu*(wz_xx + wz_yy + wz_zz) - \
                    f[2])**2)

        loss_c = jnp.mean((ux_x + uy_y + uz_z)**2)

        return loss_x + loss_y + loss_z + lbda_c*loss_c

    def initial_loss(params, t, x, y, z, w, u):
        ux, uy, uz = apply_fn(params, t, x, y, z)
        wx = vorx(apply_fn, params, t, x, y, z)
        wy = vory(apply_fn, params, t, x, y, z)
        wz = vorz(apply_fn, params, t, x, y, z)
        loss = jnp.mean((wx - w[0])**2) + jnp.mean((wy - w[1])**2) + jnp.mean((wz - w[2])**2)
        loss += jnp.mean((ux - u[0])**2) + jnp.mean((uy - u[1])**2) + jnp.mean((uz - u[2])**2)
        return loss

    def boundary_loss(params, t, x, y, z, w):
        loss = 0.
        for i in range(6):
            wx = vorx(apply_fn, params, t[i], x[i], y[i], z[i])
            wy = vory(apply_fn, params, t[i], x[i], y[i], z[i])
            wz = vorz(apply_fn, params, t[i], x[i], y[i], z[i])
            loss += (1/6.) * jnp.mean((wx - w[i][0])**2) + jnp.mean((wy - w[i][1])**2) + jnp.mean((wz - w[i][2])**2)
        return loss

    # unpack data
    tc, xc, yc, zc, fc, ti, xi, yi, zi, wi, ui, tb, xb, yb, zb, wb = train_data

    # isolate loss func from redundant arguments
    loss_fn = lambda params: residual_loss(params, tc, xc, yc, zc, fc) + \
                        lbda_ic*initial_loss(params, ti, xi, yi, zi, wi, ui) + \
                        boundary_loss(params, tb, xb, yb, zb, wb)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # model and equation
    parser.add_argument('--model', type=str, default='spinn', choices=['spinn', 'pinn'], help='model name (pinn; spinn)')
    parser.add_argument('--debug', type=str, default='false', help='debugging purpose')
    parser.add_argument('--equation', type=str, default='navier_stokes4d', help='equation to solve')
    
    # pde settings
    parser.add_argument('--nc', type=int, default=32, help='the number of collocation points')
    parser.add_argument('--nc_test', type=int, default=20, help = 'the number of collocation points')

    # training settings
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')
    parser.add_argument('--mlp', type=str, default='modified_mlp', help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=5, help='the number of layer')
    parser.add_argument('--features', type=int, default=64, help='feature size of each layer')
    parser.add_argument('--r', type=int, default=128, help='rank of a approximated tensor')
    parser.add_argument('--out_dim', type=int, default=3, help='size of model output')
    parser.add_argument('--nu', type=float, default=0.05, help='viscosity')
    parser.add_argument('--lbda_c', type=int, default=100, help='None')
    parser.add_argument('--lbda_ic', type=int, default=10, help='None')

    # log settings
    parser.add_argument('--log_iter', type=int, default=1000, help='print log every...')
    parser.add_argument('--plot_iter', type=int, default=10000, help='plot result every...')

    args = parser.parse_args()

    # random key
    key = jax.random.PRNGKey(args.seed)

    # make & init model forward function
    key, subkey = jax.random.split(key, 2)
    apply_fn, params = setup_networks(args, subkey)

    # count total params
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # name model
    name = name_model(args)

    # result dir
    root_dir = os.path.join(os.getcwd(), 'results', args.equation, args.model)
    result_dir = os.path.join(root_dir, name)

    # make dir
    os.makedirs(result_dir, exist_ok=True)

    # optimizer
    optim = optax.adam(learning_rate=args.lr)
    state = optim.init(params)

    # dataset
    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey)
    test_data = generate_test_data(args, result_dir)

    # evaluation function
    eval_fn = setup_eval_function(args.model, args.equation)

    # save training configuration
    save_config(args, result_dir)

    # log
    logs = []
    if os.path.exists(os.path.join(result_dir, 'log (loss, error).csv')):
        os.remove(os.path.join(result_dir, 'log (loss, error).csv'))
    if os.path.exists(os.path.join(result_dir, 'best_error.csv')):
        os.remove(os.path.join(result_dir, 'best_error.csv'))
    best = 100000.

    print("compiling...")

    # start training
    for e in trange(1, args.epochs + 1):
        if e == 2:
            # exclude compiling time
            start = time.time()
        if e % 100 == 0:
            # sample new input data
            key, subkey = jax.random.split(key, 2)
            train_data = generate_train_data(args, subkey)

        loss, gradient = apply_model_spinn(apply_fn, params, args.nu, args.lbda_c, args.lbda_ic, *train_data)
        params, state = update_model(optim, gradient, params, state)

        if e % 10 == 0:
            if loss < best:
                best = loss
                best_error = eval_fn(apply_fn, params, *test_data)

        # log
        if e % args.log_iter == 0:
            error = eval_fn(apply_fn, params, *test_data)
            print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {error:.8f}, best error {best_error:.8f}')
            with open(os.path.join(result_dir, 'log (loss, error).csv'), 'a') as f:
                f.write(f'{loss}, {error}, {best_error}\n')

        # visualization
        if e % args.plot_iter == 0:
            show_solution(args, apply_fn, params, test_data, result_dir, e)

    # training done
    runtime = time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)')
    jnp.save(os.path.join(result_dir, 'params.npy'), params)
        
    # save runtime
    runtime = np.array([runtime])
    np.savetxt(os.path.join(result_dir, 'total runtime (sec).csv'), runtime, delimiter=',')

    # save total error
    with open(os.path.join(result_dir, 'best_error.csv'), 'a') as f:
        f.write(f'best error: {best_error}\n')