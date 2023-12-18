# flake8: noqa
__version__ = '0.1.2'
"""
mkinit ~/code/graphid/graphid --noattrs --lazy_loader_typed
"""
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'api',
        'core',
        'demo',
        'ibeis',
        'util',
    },
    submod_attrs={},
)

__all__ = ['api', 'core', 'demo', 'ibeis', 'util']
