# -*- mode: python ; coding: utf-8 -*-
# OBSCURA3D PyInstaller spec file - Windows (one-directory build for the MSI)
# Usage: pyinstaller --noconfirm --clean OBSCURA3D_windows.spec
#
# Warp and Numba are shipped as real .py files (datas, excluded from the PYZ)
# because their JIT compilation reads function sources via inspect.getsource,
# which does not work on frozen modules. Same for obscura_kernels.py.

import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules)

block_cipher = None
project_dir = SPECPATH

datas = [
    (os.path.join(project_dir, 'obscura_kernels.py'), '.'),
    (os.path.join(project_dir, 'OBSCURA3D.ico'), '.'),
    (os.path.join(project_dir, 'OBSCURA3D.png'), '.'),
]
binaries = []
hiddenimports = []
excludes = ['tkinter', 'obscura_kernels']


def _safe_submodules(pkg_name):
    """collect_submodules skipping packages that crash on headless builds
    (Qt init in pyqtgraph.examples, jupyter deps, test suites)."""
    return collect_submodules(
        pkg_name,
        filter=lambda n: not any(
            p in n.split('.') for p in ('tests', 'test', 'examples', 'jupyter')))


def _collect_ext_libs(pkg):
    """Collect compiled extensions (.pyd/.dll/.so/.dylib) of a package shipped
    as real source files - collect_dynamic_libs misses .pyd files."""
    pkg_dir = os.path.dirname(pkg.__file__)
    site_pkgs = os.path.dirname(pkg_dir)
    out = []
    for root, _dirs, files in os.walk(pkg_dir):
        for f in files:
            if f.endswith(('.pyd', '.dll', '.so', '.dylib')):
                rel = os.path.relpath(root, site_pkgs)
                out.append((os.path.join(root, f), rel))
    return out

# Warp: sources on disk for JIT kernel codegen + CUDA/CPU backends
try:
    import warp
    datas += collect_data_files('warp', include_py_files=True)
    binaries += collect_dynamic_libs('warp') + _collect_ext_libs(warp)
    excludes.append('warp')
    print("warp: bundled as data files (source required for JIT)")
except ImportError:
    print("warp not installed - Warp backends disabled in the build")

# Numba: same source-on-disk requirement (@njit rasterizer)
try:
    import numba
    datas += collect_data_files('numba', include_py_files=True)
    binaries += _collect_ext_libs(numba)
    excludes.append('numba')
    print("numba: bundled as data files")
except ImportError:
    print("numba not installed - texture baking will use the numpy fallback")

# llvmlite locates its .dll via importlib.resources and os.add_dll_directory,
# both of which misbehave on frozen PYZ modules - ship it as real files too
try:
    import llvmlite
    datas += collect_data_files('llvmlite', include_py_files=True)
    binaries += _collect_ext_libs(llvmlite)
    excludes.append('llvmlite')
    # delvewheel/auditwheel vendored runtimes (e.g. msvcp140-<hash>.dll)
    _ll_libs = os.path.join(
        os.path.dirname(os.path.dirname(llvmlite.__file__)), 'llvmlite.libs')
    if os.path.isdir(_ll_libs):
        for _dll in os.listdir(_ll_libs):
            binaries.append((os.path.join(_ll_libs, _dll), 'llvmlite.libs'))
except ImportError:
    pass

# Open3D: binary-heavy package, optional raycasting backend
try:
    import open3d
    datas += collect_data_files('open3d')
    binaries += collect_dynamic_libs('open3d')
    hiddenimports += _safe_submodules('open3d')
    print("open3d: bundled")
except ImportError:
    print("open3d not installed - Open3D backend disabled in the build")

# pyqtgraph + PyOpenGL: optional 3D viewer
try:
    import pyqtgraph
    datas += collect_data_files('pyqtgraph')
    hiddenimports += _safe_submodules('pyqtgraph')
except ImportError:
    print("pyqtgraph not installed - 3D viewer disabled in the build")

try:
    import OpenGL
    hiddenimports += _safe_submodules('OpenGL')
except ImportError:
    pass

# rtree is imported lazily by trimesh.bounds for proximity queries - force it
try:
    import rtree
    hiddenimports.append('rtree')
    binaries += _collect_ext_libs(rtree)  # rtree/lib/spatialindex* (pip wheels)
    # conda-forge ships spatialindex in $PREFIX/lib, outside the package
    for _ld in (os.path.join(sys.prefix, 'lib'),
                os.path.join(sys.prefix, 'Library', 'lib'),
                os.path.join(sys.prefix, 'Library', 'bin')):
        if os.path.isdir(_ld):
            binaries += [(os.path.join(_ld, f), 'rtree/lib')
                         for f in os.listdir(_ld) if 'spatialindex' in f]
except ImportError:
    print("rtree not installed - trimesh proximity queries disabled in the build")

a = Analysis(
    [os.path.join(project_dir, 'OBSCURA3D.py')],
    pathex=[project_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out plain Windows CRT/VC runtime DLLs shipped by conda - the system
# versions (Win10+) must be used; bundled conda copies cause 0xC0000005 crashes.
# Delvewheel-renamed copies ("msvcp140-<hash>.dll" etc.) are kept on purpose:
# pip wheels link against those private names and MUST ship them.
EXCLUDE_DLL_NAMES = {
    'ucrtbase.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
    'msvcp140.dll', 'concrt140.dll', 'vcamp140.dll',
    'vccorlib140.dll', 'vcomp140.dll',
}
EXCLUDE_DLL_PREFIXES = ('api-ms-win-',)
a.binaries = [b for b in a.binaries if not (
    os.path.basename(b[0]).lower() in EXCLUDE_DLL_NAMES or
    any(os.path.basename(b[0]).lower().startswith(p) for p in EXCLUDE_DLL_PREFIXES)
)]
print(f"Binaries after filtering: {len(a.binaries)}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OBSCURA3D',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=os.path.join(project_dir, 'OBSCURA3D.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='OBSCURA3D',
)
