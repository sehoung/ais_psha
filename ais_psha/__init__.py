# Import sub-packages so they are accessible when importing my_package
from . import SHpmc
from . import SHvegas
from . import SHpsha

# Optional: Define a convenient API by exposing selected modules or functions
__all__ = ["SHpmc", "SHvegas", "SHpmc"]
