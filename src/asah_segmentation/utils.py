""" utils.py

Generic utilities used across the whole project.

Exported Functions:
* safe_tuple: conditionally expands an element to a tuple.

"""


def safe_tuple[T](condition: bool, el: T) -> tuple[T] | tuple[()]:
    """
    Casts an element to a singleton or empty tuple based on a condition.

    This function is useful to conditionally add elements to a tuple or list
    (see Examples for further reference).

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

    Examples
    --------
    >>> (1, *safe_tuple(True, 2), 3)
    (1, 2, 3)

    >>> (4, *safe_tuple(False, 5), 6)
    (4, 6)
    """
    return (el,) if condition else ()
