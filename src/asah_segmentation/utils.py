""" utils.py

Generic utilities used across the whole project.

Exported Functions:
* safe_tuple: conditionally expands an element to a tuple.
* check_params: validates parameters of a function taking an arbitrary number
of positional and named arguments against specified constraints.

"""

from typing import Iterable


def safe_tuple[T](condition: bool, el: T) -> tuple[T] | tuple[()]:
    """
    Casts an element to a singleton or empty tuple based on a condition.

    This function is useful to conditionally add elements to a tuple or list:
    using `(..., *safe_tuple(cond, el), ...)` adds `el` to the relevant
    collection only if `cond` is met, otherwise expands empty tuple (i.e. does
    nothing).

    Parameters
    ----------
    condition : bool
        Condition to be met
    
    el : any
        The relevant element

    Returns
    -------
    tuple[any] or tuple[()]
        Singleton tuple if condition is verified, empty tuple otherwise
    """
    return ((el,) if condition else ())


def check_params(n_args: int | Iterable[int],
                 n_kwargs: int | Iterable[int],
                 args_len: int,
                 kwargs_len: int) -> None:
    """
    Checks that the number of positional and keyword arguments respects
    given constraints.

    Parameters
    ----------
    n_args : int or Iterable[int]
        The required positional arguments cardinality or a collection
        of possible cardinalities

    n_kwargs : int or Iterable[int]
        The required keyword arguments cardinality or a collection of
        possible cardinalities
        
    args_len : int
        Actual positional arguments cardinality

    kwargs_len : int
        Actual keyword arguments cardinality

    Returns
    -------
    check_params : None
    
    Raises
    ------
    TypeError
        If either positional arguments or keyword arguments do not
        respect given constraints
    """
    
    n_args = (n_args,) if isinstance(n_args, int) else tuple(n_args)
    n_kwargs = (n_kwargs,) if isinstance(n_kwargs, int) else tuple(n_kwargs)

    f_n_args = str(n_args[0]) if len(n_args) == 1 else \
        ", ".join(map(str, n_args[:-1])) + " or " + str(n_args[-1])
    f_n_kwargs = str(n_kwargs[0]) if len(n_kwargs) == 1 else \
        ", ".join(map(str, n_kwargs[:-1])) + " or " + str(n_kwargs[-1])

    if args_len not in n_args:
        raise TypeError(f"Wrong number of positional arguments:" \
                        f"{f_n_args} expected, {args_len} found.")

    if kwargs_len not in n_kwargs:
        raise TypeError(f"Wrong number of keyword arguments:" \
                        f"{f_n_kwargs} expected, {kwargs_len} found.")
