from __future__ import annotations

import inspect
import warnings
from abc import abstractmethod
from typing import Any, Callable, Optional

from rapidpipe.layer import Layer


class NumbaLayer(Layer):
    """
    Base class for layers whose core computation is a pure numeric kernel.
    The kernel() static method must be Numba-compatible:
      - No Python objects as arguments or return values
      - Only numpy arrays, scalars, and numba-supported types
    All Python-level marshalling happens in process() before/after the kernel call.
    """

    @staticmethod
    @abstractmethod
    def kernel(*args) -> Any:
        """
        Numba-compilable numeric function.

        This method must only contain Python/NumPy code supported by Numba's nopython mode.
        It should not reference `self` or any Python object.

        Args:
            *args: Numeric arrays or scalars passed from the `process` method.

        Returns:
            Computed numeric arrays or scalars.
        """
        ...

    # Class-level compiled function (shared across all instances of a subclass)
    _compiled_kernel: Optional[Callable] = None

    def __init_subclass__(cls, nopython: bool = True, cache: bool = True,
                          parallel: bool = False, nogil: bool = False, **kwargs):
        """
        Hook into subclass creation to Ahead-Of-Time (AOT) compile the kernel method.

        Args:
            nopython: Ensure the compiled function does not fall back to Python object mode.
            cache: Cache the compiled function across sessions to disk.
            parallel: Enable automatic parallelization.
            nogil: Release the Global Interpreter Lock (GIL) during execution.
            **kwargs: Other kwargs passed to the Layer base class.
        """
        super().__init_subclass__(**kwargs)

        kernel_attr = cls.__dict__.get("kernel")
        if kernel_attr is not None and not isinstance(kernel_attr, staticmethod):
            raise TypeError(
                "The 'kernel' method must be declared as a @staticmethod.")

        try:
            import numba
            cls._compiled_kernel = staticmethod(numba.jit(
                nopython=nopython, cache=cache, parallel=parallel, nogil=nogil
            )(cls.kernel))
        except ImportError:
            # Numba not installed — fall back to pure Python with a warning
            warnings.warn(
                "Numba is not installed; falling back to pure Python kernel execution.", UserWarning)
            cls._compiled_kernel = staticmethod(cls.kernel)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Missing Support for Multiple Outputs validation
        outputs = self._output_names
        if len(outputs) > 1:
            sig = inspect.signature(self.kernel)
            ret_annotation = sig.return_annotation
            if ret_annotation is not inspect._empty:
                origin = getattr(ret_annotation, "__origin__", None)
                if origin is tuple or origin is tuple:
                    args = getattr(ret_annotation, "__args__", tuple())
                    if args and len(args) != len(outputs) and args[-1] != Ellipsis:
                        warnings.warn(
                            f"Layer '{self.name}' declares {len(outputs)} outputs but kernel return type hint is {ret_annotation}",
                            UserWarning
                        )
                else:
                    warnings.warn(
                        f"Layer '{self.name}' declares {len(outputs)} outputs but kernel return type hint is not a tuple ({ret_annotation})",
                        UserWarning
                    )

    def process(self, **inputs) -> Any:
        """
        Process the incoming data by unpacking it into the compiled kernel.

        Override in your subclass if you need to:
          1. Extract or reshape numpy arrays from inputs
          2. Call `self._compiled_kernel(*positional_args)`
          3. Wrap the raw numeric result back into Python objects

        By default, it extracts values from `inputs` in the exact order
        of the kernel's parameters using `inspect.signature`, and directly
        executes the compiled kernel.

        Args:
            **inputs: Arbitrary keyword arguments mapping to the layer's declared dependencies.

        Returns:
            The raw return value of the underlying compiled kernel.
        """
        sig = inspect.signature(self.kernel)
        try:
            ordered_args = [inputs[name] for name in sig.parameters]
        except KeyError as e:
            raise KeyError(f"Missing required input for Numba kernel: {e}")

        return self._compiled_kernel(*ordered_args)

    @classmethod
    def warm_up(cls, *sample_inputs) -> None:
        """
        Trigger AOT compilation by evaluating the kernel with sample inputs.

        Numba lazily compiles functions upon their first invocation based on the
        inferred types of the arguments. Call this before `pipeline.run()` to avoid
        the first-call compilation stall during real-time execution.

        Args:
            *sample_inputs: Dummy numeric arrays or scalars matching the exact types
                and shapes expected during production.
        """
        cls._compiled_kernel(*sample_inputs)
