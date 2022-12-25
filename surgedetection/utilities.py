import datetime
import os
import shutil
import urllib
from pathlib import Path
from typing import Any


class ConstantType:
    """Generic readonly document constants class."""

    def __getitem__(self, key: str) -> Any:
        """
        Get an item like a dict.

        param: key: The attribute name.

        return: attribute: The value of the attribute.
        """
        attribute = self.__getattribute__(key)
        return attribute

    @staticmethod
    def raise_readonly_error(key: Any, value: Any) -> None:
        """Raise a readonly error if a value is trying to be set."""
        raise ValueError(f"Trying to change a constant. Key: {key}, value: {value}")

    def __setattr__(self, key: Any, value: Any) -> None:
        """Override the Constants.key = value action."""
        self.raise_readonly_error(key, value)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Override the Constants['key'] = value action."""
        self.raise_readonly_error(key, value)


def now_str() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def download_file(url: str, filepath: Path, progress: bool = True) -> Path:
    import requests

    # If it doesn't have a suffix, assume it's a directory
    if len(filepath.suffix) == 0:
        filename = urllib.parse.urlparse(url)[2].split("/")[-1]
        filepath = filepath.joinpath(filename)

    os.makedirs(filepath.parent, exist_ok=True)

    with requests.get(url, stream=True) as request:
        with open(filepath, "wb") as outfile:
            shutil.copyfileobj(request.raw, outfile)

    return filepath
