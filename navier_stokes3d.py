import argparse
import csv
import os
import time

import jax
import numpy as np
import optax
from jax import jvp
from networks.hessian_vector_products import *
from tqdm import trange
from utils.data_generators import generate_test_data, generate_train_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import *
from utils.visualizer import show_solution
from utils.vorticity import velocity_to_vorticity_fwd


# loss function for navier stokes (SPINN)
@partial(jax.jit, static_argnums=(0,))
def apply_model_spinn(apply_fn, params, tc, xc, yc, ti, xi, yi, w0_gt, u0_gt, v0_gt, lbda_c, lbda_ic):
    def residual_loss(params, t, x, y):
        # compute [u, v]
        uv = apply_fn(params, t, x, y)
        # tangent vector dx/dx
        vec_t = jnp.ones(t.shape)
        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)
        w_t = jvp(
            lambda t: velocity_to_vorticity_fwd(apply_fn, params, t, x, y), 
            (t,), 
            (vec_t,)
        )[1]
        w_x = jvp(
            lambda x: velocity_to_vorticity_fwd(apply_fn, params, t, x, y), 
            (x,), 
            (vec_x,)
        )[1]
        w_y = jvp(
            lambda y: velocity_to_vorticity_fwd(apply_fn, params, t, x, y), 
            (y,), 
            (vec_y,)
        )[1]
        w_xx = hvp_fwdfwd(
            lambda x: velocity_to_vorticity_fwd(apply_fn, params, t, x, y), 
            (x,), 
            (vec_x,)
        )
        w_yy = hvp_fwdfwd(
            lambda y: velocity_to_vorticity_fwd(apply_fn, params, t, x, y), 
            (y,), 
            (vec_y,)
        )
        # PDE constraint
        R_w = w_t + uv[0]*w_x + uv[1]*w_y - 0.01*(w_xx + w_yy)

        # incompressible fluid constraint
        u_x = jvp(lambda x: apply_fn(params, t, x, y)[0], (x,), (vec_x,))[1]
        v_y = jvp(lambda y: apply_fn(params, t, x, y)[1], (y,), (vec_y,))[1]
        R_c = u_x + v_y

        return jnp.mean(R_w**2) + lbda_c*jnp.mean(R_c**2)

    def initial_loss(params, ti, xi, yi, w0_gt, u0_gt, v0_gt):
        # use initial vorticity and velocity
        w0 = velocity_to_vorticity_fwd(apply_fn, params, ti, xi, yi)
        R_ic_w = jnp.squeeze(w0) - w0_gt 
        u0, v0 = apply_fn(params, ti, xi, yi)
        R_ic_u = jnp.squeeze(u0) - u0_gt
        R_ic_v = jnp.squeeze(v0) - v0_gt
        loss = jnp.mean(jnp.square(R_ic_w)) + jnp.mean(jnp.square(R_ic_u)) + jnp.mean(jnp.square(R_ic_v))

        return loss

    # loss function w.r.t learnable parameters
    # no boundary loss since we're using exact periodic b.c
    loss_fn = lambda params: residual_loss(params, tc, xc, yc) + lbda_ic*initial_loss(params, ti, jnp.transpose(xi), jnp.transpose(yi), w0_gt, u0_gt, v0_gt)
    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # data directory
    parser.add_argument('--data_dir', type=str, default='./data/navier_stokes', help='a directory to reference solution')

    # model and equation
    parser.add_argument('--model', type=str, default='spinn', choices=['spinn', 'pinn'], help='model name (pinn; spinn)')
    parser.add_argument('--equation', type=str, default='navier_stokes3d', help='equation to solve')
    
    # input data settings
    parser.add_argument('--nt', type=int, default=None, help='the number of time points for time axis')
    parser.add_argument('--nxy', type=int, default=None, help='the number of points for each spatial axis')

    # training settings
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100000, help='training epochs')
    parser.add_argument('--offset_num', type=int, default=8, help='the number of offsets in training data')
    parser.add_argument('--offset_iter', type=int, default=100, help='change offset every...')
    parser.add_argument('--lbda_c', type=int, default=5000, help='weighting factor for incompressible condition')
    parser.add_argument('--lbda_ic', type=int, default=10000, help='weighting factor for initial condition')

    # model settings
    parser.add_argument('--mlp', type=str, default='modified_mlp', choices=['mlp', 'modified_mlp'], help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=3, help='the number of layer')
    parser.add_argument('--features', type=int, default=128, help='feature size of each layer')
    parser.add_argument('--r', type=int, default=128, help='rank of the approximated tensor')
    parser.add_argument('--out_dim', type=int, default=2, help='size of model output')
    parser.add_argument('--pos_enc', type=int, default=5, help='size of the positional encoding (zero if no encoding)')

    # time marching
    parser.add_argument('--marching_steps', type=int, default=10, help='step size for time marching')
    parser.add_argument('--step_idx', type=int, default=0, help='step index for time marching')

    # log settings
    parser.add_argument('--log_iter', type=int, default=1000, help='print log every...')
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
    result_dir = os.path.join(root_dir, name, f'{args.step_idx}')

    # make dir
    os.makedirs(result_dir, exist_ok=True)

    # optimizer
    optim = optax.adam(learning_rate=args.lr)
    state = optim.init(params)

    # dataset
    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey, result_dir=result_dir)
    test_data = generate_test_data(args, result_dir)

    # loss & evaluation function
    eval_fn = setup_eval_function(args.model, args.equation)

    # save training configuration
    save_config(args, result_dir)

    # log
    logs = []
    if os.path.exists(os.path.join(result_dir, 'log (loss, error).csv')):
        os.remove(os.path.join(result_dir, 'log (loss, error).csv'))
    if os.path.exists(os.path.join(result_dir, '..', 'bset_error.csv')):
        os.remove(os.path.join(result_dir, '..', 'bset_error.csv'))
    best = 10000000.
    
    # get data
    tc_mult, xc_mult, yc_mult, ti, xi, yi, w0, u0, v0 = train_data
    tc, xc, yc = tc_mult[0], xc_mult[0], yc_mult[0]

    # start training
    for e in trange(1, args.epochs + 1):
        if e == 2:
            # exclude compiling time
            start = time.time()

        if e % args.offset_iter == 0:
            # change input
            offset_idx = (e // args.offset_iter) % args.offset_num
            tc, xc, yc = tc_mult[offset_idx], xc_mult[offset_idx], yc_mult[offset_idx]

        loss, gradient = apply_model_spinn(apply_fn, params, tc, xc, yc, ti, xi, yi, w0, u0, v0, args.lbda_c, args.lbda_ic)
        params, state = update_model(optim, gradient, params, state)

        if e % 100 == 0 and e > args.epochs*0.7:
            if loss < best:
                best = loss
                best_error = eval_fn(apply_fn, params, *test_data)
                # save next IC prediction for time marching
                save_next_IC(root_dir, name, apply_fn,params, test_data, args.step_idx, e)

        # log
        if e % args.log_iter == 0:
            error = eval_fn(apply_fn, params, *test_data)
            if e == args.log_iter:
                best_error = error
            if e <= args.epochs*0.7:
                print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {error:.8f}, step_idx: {args.step_idx}')
                with open(os.path.join(result_dir, 'log (loss, error).csv'), 'a') as f:
                    f.write(f'{loss}, {error}\n')
            else:
                print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {error:.8f}, best error {best_error:.8f}, step_idx: {args.step_idx}')
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
    error_list = [0]*args.marching_steps
    if args.step_idx == args.marching_steps-1:
        for i in range(args.marching_steps):
            with open(os.path.join(root_dir, name, f'{i}', 'log (loss, error).csv'), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    error_list[i] = float(row[-1])

        final_error = sum(error_list)/len(error_list)
        with open(os.path.join(result_dir, '..', 'best_error.csv'), 'a') as f:
            f.write(f'test error for each time window: {error_list}\n')
            f.write(f'total error: {final_error}\n')