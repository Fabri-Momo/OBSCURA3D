# -*- mode: python ; coding: utf-8 -*-
# OBSCURA3D PyInstaller spec file - Linux (one-directory build, .tar.gz)
# Usage (on Linux): pyinstaller --noconfirm --clean OBSCURA3D_linux.spec
#
# Output: dist/OBSCURA3D/ (executable folder, shipped as a tar.gz archive)
# Warp/Numba ship as real .py files (JIT needs inspect.getsource).

import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules)

block_cipher = None
project_dir = SPECPATH

datas = [
    (os.path.join(project_dir, 'obscura_kernels.py'), '.'),
    (os.path.join(project_dir, 'OBSCURA3D_512x512.png'), '.'),
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
    as real source files - collect_dynamic_libs misses some of them."""
    pkg_dir = os.path.dirname(pkg.__file__)
    site_pkgs = os.path.dirname(pkg_dir)
    out = []
    for root, _dirs, files in os.walk(pkg_dir):
        for f in files:
            if f.endswith(('.pyd', '.dll', '.so', '.dylib')):
                rel = os.path.relpath(root, site_pkgs)
                out.append((os.path.join(root, f), rel))
    return out


try:
    import warp
    datas += collect_data_files('warp', include_py_files=True)
    binaries += collect_dynamic_libs('warp') + _collect_ext_libs(warp)
    excludes.append('warp')
    print("warp: bundled as data files (source required for JIT)")
except ImportError:
    print("warp not installed - Warp backends disabled in the build")

try:
    import numba
    datas += collect_data_files('numba', include_py_files=True)
    binaries += _collect_ext_libs(numba)
    excludes.append('numba')
    print("numba: bundled as data files")
except ImportError:
    print("numba not installed - texture baking will use the numpy fallback")

try:
    import llvmlite
    datas += collect_data_files('llvmlite', include_py_files=True)
    binaries += _collect_ext_libs(llvmlite)
    excludes.append('llvmlite')
    _ll_libs = os.path.join(
        os.path.dirname(os.path.dirname(llvmlite.__file__)), 'llvmlite.libs')
    if os.path.isdir(_ll_libs):
        for _dll in os.listdir(_ll_libs):
            binaries.append((os.path.join(_ll_libs, _dll), 'llvmlite.libs'))
except ImportError:
    pass

try:
    import open3d
    datas += collect_data_files('open3d')
    binaries += collect_dynamic_libs('open3d')
    hiddenimports += _safe_submodules('open3d')
    print("open3d: bundled")
except ImportError:
    print("open3d not installed - Open3D backend disabled in the build")

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
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon_path = os.path.join(project_dir, 'OBSCURA3D_512x512.png')
if not os.path.isfile(icon_path):
    icon_path = None

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
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
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
