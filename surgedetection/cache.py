"""Functions to handle result caching."""
import hashlib
import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent.parent.joinpath(".cache").resolve()


def get_cache_name(
    function_name: str,
    args: Sequence[Any] | None = None,
    kwargs: dict[Any, Any] | None = None,
    extension: str = "pkl",
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """
    Retrieve a suitable and repeatable cache filename.

    Arguments
    ---------
    function_name: The name of the function, making the start of the filename.
    args: Arguments to derive the sha1 checksum of
    kwargs: Keyword arguments to derive the sha1 checksum of
    extension: The extension of the cached filepath (e.g. ".pkl" / ".txt")
    cache_dir: The base cache directory to use.

    Returns
    -------
    The full path to the cache filepath

    Examples
    --------
    >>> get_cache_name("my-func", cache_dir=Path("/tmp/"))
    PosixPath('/tmp/my-func.pkl')
    >>> get_cache_name("my-func", args=[1, "b"], cache_dir=Path("/tmp/"))
    PosixPath('/tmp/my-func_2d91a20a20fcaeb0ae60b5189b810bdf8481b1d7.pkl')
    """
    # Make sure that the path will be usable
    os.makedirs(cache_dir, exist_ok=True)

    arg_strs = ""
    if args is not None:
        arg_strs += "".join([str(a) for a in args])
    if kwargs is not None:
        arg_strs += "".join([str(k) + str(v) for k, v in kwargs.items()])

    args_hash = "" if len(arg_strs) == 0 else "_" + hashlib.sha1(arg_strs.encode("utf-8")).hexdigest()

    if not extension.startswith("."):
        extension = "." + extension

    return cache_dir.joinpath(function_name + args_hash).with_suffix(extension)


def cache(func, cache_dir: Path = CACHE_DIR):  # type: ignore
    """
    Cache a given function.

    Arguments
    ---------
    func: The function to cache
    """
    if not CACHE_DIR.is_dir():
        os.mkdir(CACHE_DIR)

    def wrapped(*args, **kwargs):  # type: ignore
        cache_filename = get_cache_name(func.__name__, args, kwargs)
        if cache_filename.is_file():
            with open(cache_filename, "rb") as infile:
                return pickle.load(infile)

        result = func(*args, **kwargs)

        with open(cache_filename, "wb") as outfile:
            pickle.dump(result, outfile)

        return result

    return wrapped


def test_cache() -> None:
    @cache
    def hello(_a: int) -> None:
        print("there")

    hello(_a=1)
