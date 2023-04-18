import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import flax.training.checkpoints as checkpoints
import optax


class FourierEncoding(nn.Module):
    dimensions: int
    n_basis: int
    freq_sigma: float

    def setup(self):
        def freq_init(k):
            weights = jax.random.normal(k, (self.dimensions, self.n_basis))
            return 2 * jnp.pi * self.freq_sigma * weights

        self.freq = self.param("freq", freq_init)
        self.magnitude = jnp.linalg.norm(self.freq, axis=0, keepdims=True)

    def __call__(self, position):
        projection = position @ self.freq
        sin_enc = jnp.sin(projection) / (self.magnitude * 1000.0)
        cos_enc = jnp.cos(projection) / (self.magnitude * 1000.0)

        return jnp.concatenate([position, sin_enc, cos_enc], axis=-1)


class ShapeModel(nn.Module):
    dimensions: int
    n_common_layers: int
    n_per_head_layers: int
    common_layer_units: int
    per_head_units: int
    n_fourier_basis: int
    fourier_freq_sigma: float

    def setup(self):
        self.position_encoding = FourierEncoding(self.dimensions,
                                                 self.n_fourier_basis,
                                                 self.fourier_freq_sigma)

        def activation(x): return nn.softplus(100 * x) / 100

        # Spherical Init
        def ki(k, s, d): return jax.random.normal(k, s) * jnp.sqrt(2 / s[-1])
        def bi(k, s, d): return jnp.zeros(s)

        common_layers = []
        for _ in range(self.n_common_layers):
            common_layers += [
                nn.Dense(self.common_layer_units, kernel_init=ki,
                         bias_init=bi),
                activation,
            ]
        self.common_layers = common_layers

        sdf_head = []
        for _ in range(self.n_per_head_layers):
            sdf_head += [
                nn.Dense(self.per_head_units, kernel_init=ki, bias_init=bi),
                activation,
            ]
        sdf_head.append(nn.Dense(1, kernel_init=ki, bias_init=bi))
        self.sdf_head = sdf_head

        grad_head = []
        for _ in range(self.n_per_head_layers):
            grad_head += [
                nn.Dense(self.per_head_units),
                activation,
            ]
        grad_head.append(nn.Dense(self.dimensions))
        self.grad_head = grad_head

        inner_mf_head = []
        for _ in range(self.n_per_head_layers):
            inner_mf_head += [
                nn.Dense(self.per_head_units),
                activation,
            ]
        inner_mf_head.append(nn.Dense(1))
        self.inner_mf_head = inner_mf_head

        outer_mf_head = []
        for _ in range(self.n_per_head_layers):
            outer_mf_head += [
                nn.Dense(self.per_head_units),
                activation,
            ]
        outer_mf_head.append(nn.Dense(1))
        self.outer_mf_head = outer_mf_head

    def __call__(self, position):
        net = self.position_encoding(position)
        for layer in self.common_layers:
            net = layer(net)
        common_feature = net

        net = common_feature
        for layer in self.sdf_head:
            net = layer(net)
        sdf = net

        net = common_feature
        for layer in self.grad_head:
            net = layer(net)
        grad = net

        net = common_feature
        for layer in self.inner_mf_head:
            net = layer(net)
        inner_mf = net

        net = common_feature
        for layer in self.outer_mf_head:
            net = layer(net)
        outer_mf = net

        mf = jnp.where(sdf < 0, inner_mf, outer_mf)

        return sdf, grad, mf

    def with_sdf_grad(self, position):
        def invocation(position):
            sdf, grad, mf = self(position)
            return jnp.sum(sdf), (sdf, grad, mf)

        agrad, aux = jax.grad(invocation, has_aux=True)(position)
        sdf, pgrad, mf = aux

        return sdf, agrad, pgrad, mf

    def with_mf_grad(self, position):
        def invocation(position):
            sdf, agrad, pgrad, mf = self.with_sdf_grad(position)
            return jnp.sum(mf), (sdf, agrad, pgrad, mf)

        mf_grad, aux = jax.grad(invocation, has_aux=True)(position)
        sdf, agrad, pgrad, mf = aux

        return sdf, agrad, pgrad, mf, mf_grad


def build_and_restore_model(dimensions, args, checkpoint_path, seed):
    model = ShapeModel(dimensions=dimensions,
                       n_common_layers=args.common_layers,
                       n_per_head_layers=args.per_head_layers,
                       common_layer_units=args.common_layer_units,
                       per_head_units=args.per_head_units,
                       n_fourier_basis=args.n_fourier_basis,
                       fourier_freq_sigma=args.fourier_freq_sigma)

    prototype_input = jnp.zeros((1, dimensions))
    params = model.init(jax.random.PRNGKey(seed), prototype_input)

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)
    params, opt_state, step = checkpoints.restore_checkpoint(checkpoint_path,
                                                             (params, opt_state, 0))

    return model, params, opt_state, step
