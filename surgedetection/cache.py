import hashlib
import os
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(__file__).joinpath("../../.cache").resolve()


def get_cache_name(function_name: str, args: Sequence[Any] | None = None, kwargs: dict[Any, Any] | None = None) -> Path:
    # Make sure that the path will be usable
    os.makedirs(CACHE_DIR, exist_ok=True)

    arg_strs = ""
    if args is not None:
        arg_strs += "".join([str(a) for a in args])
    if kwargs is not None:
        arg_strs += "".join([str(k) + str(v) for k, v in kwargs.items()])

    args_hash = "" if len(arg_strs) == 0 else "_" + hashlib.sha1(arg_strs.encode("utf-8")).hexdigest()

    return CACHE_DIR.joinpath(function_name + args_hash).with_suffix(".pkl")


def cache(func, cache_dir: Path = CACHE_DIR):  # type: ignore
    """
    Cache a given function.

    :param func: The function to cache
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
