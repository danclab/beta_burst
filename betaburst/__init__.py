import lazy_loader as lazy

(__getattr__, __dir__, __all__) = lazy.attach_stub(__name__, __file__)

# initialize logging
from .utils import set_log_level, set_log_file

set_log_level(None, False)
set_log_file()