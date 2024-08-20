import jax.numpy as jnp
from typing import Dict, List
from jaxley.solver_gate import save_exp

def softplus(x: jnp.ndarray,b:float) -> jnp.ndarray:
    """Softplus."""
    return (1/b)*jnp.log(1 + save_exp(x*b))

def inv_softplus(x: jnp.ndarray,b:float) -> jnp.ndarray:
    """Inverse softplus."""
    return jnp.log(save_exp(b*x) - 1)/b

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid."""
    return 1 / (1 + save_exp(-x))


def expit(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse sigmoid (expit)"""
    return -jnp.log(1 / x - 1)


class ParamTransform:
    """Parameter transformation utility.

    Updated version that allows scaling the softplus with an additional parameter b

    This class is used to transform parameters from an unconstrained space to a constrained space
    and back. If the range is bounded both from above and below, we use the sigmoid function to
    transform the parameters. If the range is only bounded from below or above, we use softplus.

    Attributes:
        lowers: A dictionary of lower bounds for each parameter (None for no bound).
        uppers: A dictionary of upper bounds for each parameter (None for no bound).

    """

    def __init__(self, lowers: Dict[str, float], uppers: Dict[str, float],bs: Dict[str,float]) -> None:
        """Initialize the bounds.

        Args:
            lowers: A dictionary of lower bounds for each parameter (None for no bound).
            uppers: A dictionary of upper bounds for each parameter (None for no bound).
        """

        self.lowers = lowers
        self.uppers = uppers
        self.bs = bs

    def forward(self, params: List[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
        """Pushes unconstrained parameters through a tf such that they fit the interval.

        Args:
            params: A list of dictionaries with unconstrained parameters.

        Returns:
            A list of dictionaries with transformed parameters.

        """

        tf_params = []
        for param in params:
            key = list(param.keys())[0]

            # If constrained from below and above, use sigmoid
            if self.lowers[key] is not None and self.uppers[key] is not None:
                tf = (
                    sigmoid(param[key]) * (self.uppers[key] - self.lowers[key])
                    + self.lowers[key]
                )
                tf_params.append({key: tf})

            # If constrained from below, use softplus
            elif self.lowers[key] is not None:
                b = self.bs[key]
                tf = softplus(param[key],b) + self.lowers[key]
                tf_params.append({key: tf})

            # If constrained from above, use negative softplus
            elif self.uppers[key] is not None:
                b = self.bs[key]
                tf = -softplus(-param[key],b) + self.uppers[key]
                tf_params.append({key: tf})
            # Else just pass through
            else:
                tf_params.append({key: param[key]})
                
        return tf_params

    def inverse(self, params: jnp.ndarray) -> jnp.ndarray:
        """Takes parameters from within the interval and makes them unconstrained.

        Args:
            params: A list of dictionaries with transformed parameters.

        Returns:
            A list of dictionaries with unconstrained parameters.
        """

        tf_params = []
        for param in params:
            key = list(param.keys())[0]

            # If constrained from below and above, use expit
            if self.lowers[key] is not None and self.uppers[key] is not None:
                tf = expit(
                    (param[key] - self.lowers[key])
                    / (self.uppers[key] - self.lowers[key])
                )
                tf_params.append({key: tf})

            # If constrained from below, use inv_softplus
            elif self.lowers[key] is not None:
                b = self.bs[key]
                tf = inv_softplus(param[key] - self.lowers[key],b)
                tf_params.append({key: tf})

            # If constrained from above, use negative inv_softplus
            elif self.uppers[key] is not None:
                b = self.bs[key]
                tf = -inv_softplus(-(param[key] - self.uppers[key]),b)
                tf_params.append({key: tf})

            # else just pass through
            else:
                tf_params.append({key: param[key]})

        return tf_params
