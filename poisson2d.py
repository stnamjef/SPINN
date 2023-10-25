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
from utils.visualizer import show_solution
import matplotlib.pyplot as plt
import pdb


@partial(jax.jit, static_argnums=(0,))
def apply_model_spinn(apply_fn, params, xc1, yc1, xc2, yc2, xb, yb):
    def residual_loss(params, x, y):
        # compute u
        u = apply_fn(params, x, y)
        # tangent vector dx/dx
        v_x = jnp.ones(x.shape)
        v_y = jnp.ones(y.shape)
        # 1st, 2nd derivatives of u
        _, uxx = hvp_fwdfwd(lambda x: apply_fn(params, x, y), (x,), (v_x,), True)
        _, uyy = hvp_fwdfwd(lambda y: apply_fn(params, x, y), (y,), (v_y,), True)
        return jnp.mean((1 + uxx + uyy)**2)

    def boundary_loss(params, x, y):
        loss = 0.
        for i in range(6):
            loss += jnp.mean(apply_fn(params, x[i], y[i])**2)
        return loss

    # isolate loss func from redundant arguments
    loss_fn = lambda params: residual_loss(params, xc1, yc1) + \
                        residual_loss(params, xc2, yc2) + \
                        1000*boundary_loss(params, xb, yb)
    # loss_fn = lambda params: boundary_loss(params, xb, yb)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # data directory
    parser.add_argument('--data_dir', type=str, default='./data/diffusion3d', help='a directory to gt data')

    # model and equation
    parser.add_argument('--model', type=str, default='spinn', choices=['spinn', 'pinn'], help='model name (pinn; spinn)')
    parser.add_argument('--equation', type=str, default='diffusion3d', help='equation to solve')
    
    # input data settings
    parser.add_argument('--nc', type=int, default=64, help='the number of input points for each axis')

    # training settings
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    # model settings
    parser.add_argument('--mlp', type=str, default='modified_mlp', choices=['mlp', 'modified_mlp'], help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=3, help='the number of layer')
    parser.add_argument('--features', type=int, default=128, help='feature size of each layer')
    parser.add_argument('--r', type=int, default=128, help='rank of the approximated tensor')
    parser.add_argument('--out_dim', type=int, default=1, help='size of model output')
    parser.add_argument('--pos_enc', type=int, default=0, help='size of the positional encoding (zero if no encoding)')

    # log settings
    parser.add_argument('--log_iter', type=int, default=10000, help='print log every...')
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
    train_data = generate_train_data(args, subkey, result_dir=result_dir)
    test_data = generate_test_data(args, result_dir)
    xt, yt, u_gt_ = test_data
    mask = jnp.ones((81, 161))
    temp_arr = jnp.expand_dims(jnp.concatenate((jnp.ones(81), jnp.zeros(161 - 81))), 0)
    for i in range(80):
        mask = jnp.concatenate((mask, temp_arr))

    if args.model == 'spinn':
        xc1_mult, yc1_mult, xc2_mult, yc2_mult, xb, yb = train_data
        xc1 = xc1_mult[0]
        yc1 = yc1_mult[0]
        xc2 = xc2_mult[0]
        yc2 = yc2_mult[0]

    # loss & evaluation function
    eval_fn = setup_eval_function(args.model, args.equation)

    # save training configuration
    save_config(args, result_dir)

    # log
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

        if e % 100 == 0 and args.model == 'pinn':
            # sample new input data
            key, subkey = jax.random.split(key, 2)
            train_data = generate_train_data(args, subkey)

        if e % 100 == 0 and args.model == 'spinn':
            # change input
            offset_idx = (e // 100) % 8
            xc1, yc1, xc2, yc2 = xc1_mult[offset_idx], yc1_mult[offset_idx], xc2_mult[offset_idx], yc2_mult[offset_idx]

        loss, gradient = apply_model_spinn(apply_fn, params, xc1, yc1, xc2, yc2, xb, yb)
        params, state = update_model(optim, gradient, params, state)

        if e % 100 == 0 and loss < best:
            # save the best error when the loss value is lowest
            best = loss
            best_error = eval_fn(apply_fn, mask, params, *test_data)

        # log
        if e % args.log_iter == 0:
            rl2 = eval_fn(apply_fn, mask, params, *test_data)
            print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {rl2:.8f}, best error {best_error:.8f}')
            with open(os.path.join(result_dir, 'log (loss, error).csv'), 'a') as f:
                f.write(f'{loss}, {rl2}, {best_error}\n')

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