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


@partial(jax.jit, static_argnums=(0,))
def apply_model_spinn(apply_fn, params, *train_data):
    def residual_loss(params, t, x, y, a, b):
        # tangent vector dx/dx
        v_t = jnp.ones(t.shape)
        v_x = jnp.ones(x.shape)
        v_y = jnp.ones(y.shape)
        # 1st derivatives of u
        ut = jvp(lambda t: apply_fn(params, t, x, y), (t,), (v_t,))[1]
        ux = jvp(lambda x: apply_fn(params, t, x, y), (x,), (v_x,))[1]
        uy = jvp(lambda y: apply_fn(params, t, x, y), (y,), (v_y,))[1]
        return jnp.mean((ut + a*ux + b*uy)**2)

    def initial_loss(params, t, x, y, u):
        return jnp.mean((apply_fn(params, t, x, y) - u)**2)

    def boundary_loss(params, t, x, y, u):
        loss = 0.
        for i in range(4):
            loss += jnp.mean((apply_fn(params, t[i], x[i], y[i]) - u[i])**2)
        return loss

    # unpack data
    tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b = train_data

    # isolate loss func from redundant arguments
    loss_fn = lambda params: 10*residual_loss(params, tc, xc, yc, a, b) + \
                        initial_loss(params, ti, xi, yi, ui) + \
                        boundary_loss(params, tb, xb, yb, ub)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient


@partial(jax.jit, static_argnums=(0,))
def apply_model_pinn(apply_fn, params, *train_data):
    def residual_loss(params, t, x, y, a, b):
        # compute u
        # u = apply_fn(params, t, x, y)
        # tangent vector du/du
        v = jnp.ones(t.shape)
        # 1st derivatives of u
        ut = vjp(lambda t: apply_fn(params, t, x, y), t)[1](v)[0]
        ux = vjp(lambda x: apply_fn(params, t, x, y), x)[1](v)[0]
        uy = vjp(lambda y: apply_fn(params, t, x, y), y)[1](v)[0]
        return jnp.mean((ut + a*ux + b*uy)**2)

    def initial_boundary_loss(params, t, x, y, u):
        return jnp.mean((apply_fn(params, t, x, y) - u)**2)

    # unpack data
    tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b = train_data

    # isolate loss function from redundant arguments
    loss_fn = lambda params: 10*residual_loss(params, tc, xc, yc, a, b) + \
                        initial_boundary_loss(params, ti, xi, yi, ui) + \
                        initial_boundary_loss(params, tb, xb, yb, ub)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient


@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # model and equation
    parser.add_argument('--model', type=str, default='spinn', choices=['spinn', 'pinn'], help='model name (pinn; spinn)')
    parser.add_argument('--equation', type=str, default='flow_mixing3d', help='equation to solve')

    # input data settings
    parser.add_argument('--nc', type=int, default=64, help='the number of input points for each axis')
    parser.add_argument('--nc_test', type=int, default=100, help='the number of test points for each axis')

    # training settings
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    # model settings
    parser.add_argument('--mlp', type=str, default='modified_mlp', choices=['mlp', 'modified_mlp'], help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=3, help='the number of layer')
    parser.add_argument('--features', type=int, default=64, help='feature size of each layer')
    parser.add_argument('--r', type=int, default=128, help='rank of a approximated tensor')
    parser.add_argument('--out_dim', type=int, default=1, help='size of model output')
    parser.add_argument('--pos_enc', type=int, default=0, help='size of the positional encoding (zero if no encoding)')

    # PDE settings
    parser.add_argument('--vmax', type=float, default=0.385, help='maximum tangential velocity')

    # log settings
    parser.add_argument('--log_iter', type=int, default=5000, help='print log every...')
    parser.add_argument('--plot_iter', type=int, default=50000, help='plot result every...')

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

    # start training
    for e in trange(1, args.epochs + 1):
        if e == 2:
            # exclude compiling time
            start = time.time()

        if e % 100 == 0:
            # sample new input data
            key, subkey = jax.random.split(key, 2)
            train_data = generate_train_data(args, subkey)

        # single run
        if args.model == 'spinn':
            loss, gradient = apply_model_spinn(apply_fn, params, *train_data)
        elif args.model == 'pinn':
            loss, gradient = apply_model_pinn(apply_fn, params, *train_data)
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

