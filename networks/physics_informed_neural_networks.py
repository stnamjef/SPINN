import pdb
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

def _navier_stokes4d_exact_w(t, x, y, z, nu):
    # analytic form of vortcity
    w_x = -3*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.cos(z)
    w_y = 6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.cos(z)
    w_z = -6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.cos(2*y)*jnp.sin(z)
    return w_x, w_y, w_z

def _navier_stokes4d_exact_u(t, x, y, z, nu=0.05):
    # analytic form of velocity
    u_x = 2*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.sin(z)
    u_y = -1*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.sin(z)
    u_z = -2*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.sin(2*y)*jnp.cos(z)
    return u_x, u_y, u_z


class NS_exact(nn.Module):
    @nn.compact
    def __call__(self, t, x, y, z):
        # pdb.set_trace()
        if jnp.ndim(t) > 1:
            t = jnp.squeeze(t, axis=1)
        if jnp.ndim(x) > 1:
            x = jnp.squeeze(x, axis=1)
        if jnp.ndim(y) > 1:
            y = jnp.squeeze(y, axis=1)
        if jnp.ndim(z) > 1:
            z = jnp.squeeze(z, axis=1)
        t, x, y, z = jnp.meshgrid(t, x, y, z, indexing='ij')
        u_x, u_y, u_z = _navier_stokes4d_exact_u(t, x, y, z)
        # pdb.set_trace()
        return u_x, u_y, u_z



class PINN2d(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, y):
        X = jnp.concatenate([x, y], axis=1)
        init = nn.initializers.glorot_normal()
        for fs in self.features[:-1]:
            X = nn.Dense(fs, kernel_init=init)(X)
            X = nn.activation.tanh(X)
        X = nn.Dense(self.features[-1], kernel_init=init)(X)
        return X
    

class PINN3d(nn.Module):
    features: Sequence[int]
    out_dim: int
    pos_enc: int

    @nn.compact
    def __call__(self, x, y, z):
        if self.pos_enc != 0:
            # freq = jnp.array([[2**k for k in range(int(-(self.pos_enc-1)/2), int((self.pos_enc+1)/2))]]) * jnp.pi
            freq = jnp.array([[2**k for k in range(int(-(self.pos_enc-1)/2), int((self.pos_enc+1)/2))]])
            x = jnp.concatenate((jnp.sin(x@freq), jnp.cos(x@freq)), 1)
            y = jnp.concatenate((jnp.sin(y@freq), jnp.cos(y@freq)), 1)
            z = jnp.concatenate((jnp.sin(z@freq), jnp.cos(z@freq)), 1)
        X = jnp.concatenate([x, y, z], axis=1)
        
        init = nn.initializers.glorot_normal()
        for fs in self.features[:-1]:
            X = nn.Dense(fs, kernel_init=init)(X)
            X = nn.activation.tanh(X)
        X = nn.Dense(self.features[-1], kernel_init=init)(X)

        return X


class PINN4d(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, t, x, y, z):
        X = jnp.concatenate([t, x, y, z], axis=1)
        init = nn.initializers.glorot_normal()
        for fs in self.features[:-1]:
            X = nn.Dense(fs, kernel_init=init)(X)
            X = nn.activation.tanh(X)
        X = nn.Dense(self.features[-1], kernel_init=init)(X)
        return X


class SPINN2d(nn.Module):
    features: Sequence[int]
    r: int
    mlp: str

    @nn.compact
    def __call__(self, x, y):
        inputs, outputs = [x, y], []
        init = nn.initializers.glorot_normal()
        if self.mlp == 'mlp':
            for X in inputs:
                for fs in self.features[:-1]:
                    X = nn.Dense(fs, kernel_init=init)(X)
                    X = nn.activation.tanh(X)
                X = nn.Dense(self.r, kernel_init=init)(X)
                outputs += [X]
        else:
            for X in inputs:
                U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                for fs in self.features[:-1]:
                    Z = nn.Dense(fs, kernel_init=init)(H)
                    Z = nn.activation.tanh(Z)
                    H = (jnp.ones_like(Z)-Z)*U + Z*V
                H = nn.Dense(self.r, kernel_init=init)(H)
                outputs += [H]

        return jnp.dot(outputs[0], outputs[-1].T)


class SPINN3d(nn.Module):
    features: Sequence[int]
    r: int
    out_dim: int
    pos_enc: int
    mlp: str

    @nn.compact
    def __call__(self, x, y, z):
        '''
        inputs: input factorized coordinates
        outputs: feature output of each body network
        xy: intermediate tensor for feature merge btw. x and y axis
        pred: final model prediction (e.g. for 2d output, pred=[u, v])
        '''
        if self.pos_enc != 0:
            # positional encoding only to spatial coordinates
            freq = jnp.expand_dims(jnp.arange(1, self.pos_enc+1, 1), 0)
            y = jnp.concatenate((jnp.ones((y.shape[0], 1)), jnp.sin(y@freq), jnp.cos(y@freq)), 1)
            z = jnp.concatenate((jnp.ones((z.shape[0], 1)), jnp.sin(z@freq), jnp.cos(z@freq)), 1)

            # causal PINN version (also on time axis)
            #  freq_x = jnp.expand_dims(jnp.power(10.0, jnp.arange(0, 3)), 0)
            # x = x@freq_x
            
        inputs, outputs, xy, pred = [x, y, z], [], [], []
        init = nn.initializers.glorot_normal()

        if self.mlp == 'mlp':
            for X in inputs:
                for fs in self.features[:-1]:
                    X = nn.Dense(fs, kernel_init=init)(X)
                    X = nn.activation.tanh(X)
                X = nn.Dense(self.r*self.out_dim, kernel_init=init)(X)
                outputs += [jnp.transpose(X, (1, 0))]

        elif self.mlp == 'modified_mlp':
            for X in inputs:
                U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=init)(X))
                for fs in self.features[:-1]:
                    Z = nn.Dense(fs, kernel_init=init)(H)
                    Z = nn.activation.tanh(Z)
                    H = (jnp.ones_like(Z)-Z)*U + Z*V
                H = nn.Dense(self.r*self.out_dim, kernel_init=init)(H)
                outputs += [jnp.transpose(H, (1, 0))]
        
        for i in range(self.out_dim):
            xy += [jnp.einsum('fx, fy->fxy', outputs[0][self.r*i:self.r*(i+1)], outputs[1][self.r*i:self.r*(i+1)])]
            pred += [jnp.einsum('fxy, fz->xyz', xy[i], outputs[-1][self.r*i:self.r*(i+1)])]

        if len(pred) == 1:
            # 1-dimensional output
            return pred[0]
        else:
            # n-dimensional output
            return pred


class SPINN4d(nn.Module):
    features: Sequence[int]
    r: int
    out_dim: int
    mlp: str

    @nn.compact
    def __call__(self, t, x, y, z):
        inputs, outputs, tx, txy, pred = [t, x, y, z], [], [], [], []
        # inputs, outputs = [t, x, y, z], []
        init = nn.initializers.glorot_normal()
        for X in inputs:
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.r*self.out_dim, kernel_init=init)(X)
            outputs += [jnp.transpose(X, (1, 0))]

        for i in range(self.out_dim):
            tx += [jnp.einsum('ft, fx->ftx', 
            outputs[0][self.r*i:self.r*(i+1)], 
            outputs[1][self.r*i:self.r*(i+1)])]

            txy += [jnp.einsum('ftx, fy->ftxy', 
            tx[i], 
            outputs[2][self.r*i:self.r*(i+1)])]

            pred += [jnp.einsum('ftxy, fz->txyz', 
            txy[i], 
            outputs[3][self.r*i:self.r*(i+1)])]


        if len(pred) == 1:
            # 1-dimensional output
            return pred[0]
        else:
            # n-dimensional output
            return pred

class SPINNnd(nn.Module):
    features: Sequence[int]
    r: int

    @nn.compact
    def __call__(self, t, *x):
        inputs = [t, *x]
        dim = len(inputs)
        # inputs, outputs, tx, txy, pred = [t, x, y, z], [], [], [], []
        # inputs, outputs = [t, x, y, z], []
        outputs = []
        init = nn.initializers.glorot_normal()
        for X in inputs:
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.r, kernel_init=init)(X)
            outputs += [jnp.transpose(X, (1, 0))]

        # einsum(a,b->c)
        a = 'za'
        b = 'zb'
        c = 'zab'
        pred = jnp.einsum(f'{a}, {b}->{c}', outputs[0], outputs[1])
        for i in range(dim-2):
            a = c
            b = f'z{chr(97+i+2)}'
            c = c+chr(97+i+2)
            if i == dim-3:
                c = c[1:]
            pred = jnp.einsum(f'{a}, {b}->{c}', pred, outputs[i+2])
            # pred = jnp.einsum('fab, fc->fabc', pred, outputs[i+2])

        return pred