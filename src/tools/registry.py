"""File for managing registering modules."""

from typing import Callable


def _register_generic(module_dict: dict, module_name: str, module) -> None:
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules.

    The class extends a dictionary and provides a register functions.
    Eg. creating a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name) -> Callable:
        """Get a register_fn decorator."""
        # used as function call
        # (disabled here, need to add argument `module=None` and `Optional[Callable]`)
        # if module is not None:
        #     _register_generic(self, module_name, module)
        #     return None

        # used as decorator
        def register_fn(function):
            _register_generic(self, module_name, function)
            return function

        return register_fn
