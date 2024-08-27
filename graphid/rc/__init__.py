__mkinit__ = """
mkinit ~/code/graphid/graphid/rc/__init__.py --lazy-loader

python -c "from geowatch import rc"
EAGER_IMPORT_MODULES=geowatch python -c "from graphid import rc"
"""
