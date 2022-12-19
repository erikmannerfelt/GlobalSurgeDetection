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
