# from typing import Callable, Tuple
# import jax
# from jax import Array


# def transform_uniform_to_normal(
#     lower: Array, upper: Array
# ) -> Tuple[Callable, Callable]:
#     def transform(params: Array) -> Array:
#         p = (params - lower) / (upper - lower)
#         eps = jax.scipy.stats.norm.ppf(p)
#         return eps

#     def inv_transform(params: Array) -> Array:
#         u = jax.scipy.stats.norm.cdf(params)
#         return u * (upper - lower) + lower

#     return transform, inv_transform
