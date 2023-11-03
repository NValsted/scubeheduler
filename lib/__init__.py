import importlib.util
import re
import sys
from pathlib import Path


class _ADCSSim:
    def __init__(self):
        base_path = Path(__file__).parent / "adcs-simulation" / "adcs-simulation"

        specs = {}
        modules = {}
        for file in base_path.glob("**/*.py"):
            spec = importlib.util.spec_from_file_location(file.stem, file)
            specs[file.stem] = spec

            modules[file.stem] = importlib.util.module_from_spec(spec)  # type: ignore
            sys.modules[file.stem] = modules[file.stem]

        k = None
        while len(modules) > 0:
            try:
                if k is None:
                    k = next(iter(modules))
                specs[k].loader.exec_module(modules[k])
                self.__setattr__(k, modules[k])
                del modules[k]
                k = None
            except ModuleNotFoundError as e:
                print(e)
                k = None
            except ImportError as e:
                k = re.search(r"cannot import name '(.*)' from '(.*)'", e.msg).group(2)  # type: ignore # NOQA: E501


adcssim = _ADCSSim()
