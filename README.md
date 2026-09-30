# OBSCURA3D

Cross-platform desktop application (PyQt5) for computing **volumetric
obscurance** on 3D meshes, with three modes:

- **VO** — full sphere (default)
- **VOP** — positive hemisphere along the surface normal
- **VON** — negative hemisphere

Results can be exported as per-vertex colors (PLY) or baked into a UV
texture (PNG). A shaded OpenGL viewer displays the result automatically
after each computation.

## Downloads

Ready-to-use installers are published on the
[GitHub releases page](https://github.com/Fabri-Momo/OBSCURA3D/releases):

| Platform | Package |
|----------|---------|
| Windows x64 | `OBSCURA3D-<version>-win64.msi` (installer with license page and install-path selection) |
| macOS Apple Silicon | `OBSCURA3D-macOS-Apple-Silicon.dmg` |
| Linux x86_64 | `OBSCURA3D-<version>-linux-x86_64.tar.gz` |

## Origin

OBSCURA3D derives from the following publication (a copy,
[`2021 jch.pdf`](2021%20jch.pdf), is included in this repository):

> Rolland, T., Monna, F., Magail, J., Esin, Y., Navarro, N., Wilczek, J.,
> Gantulga, J.-O., Chateau-Smith, C. (2021). *Documenting carved stones
> from 3D models. Part II – Ambient occlusion to reveal carved parts.*
> Journal of Cultural Heritage, 49, 28–37.
> <https://doi.org/10.1016/j.culher.2021.03.006>

The paper evaluates five algorithms for computing ambient occlusion and
sky visibility on 3D models of carved stones (Mongolian deer stones) and
shows that **volumetric obscurance gives the best results** for revealing
carved figures. If you use OBSCURA3D in your research, please cite it.

## Compute backends

Auto-detected in this order:

| Backend | Platform |
|---------|----------|
| Warp CUDA | Windows/Linux with NVIDIA GPU |
| Warp CPU | any OS without CUDA |
| Open3D | CPU fallback (`RaycastingScene`) |
| NumPy | last resort |
| metal_hybrid | reserved Apple Silicon slot (opt-in) |

Force a backend: `python OBSCURA3D.py --backend warp_cpu`
Headless sanity check: `python OBSCURA3D.py --smoke-test`

## Run from source

```bash
pip install -r requirements.txt
python OBSCURA3D.py
```

## Build distributables

All three builds use PyInstaller (`--noconfirm --clean <spec>`). Warp and
Numba are bundled as real `.py` source files because their JIT compilation
needs `inspect.getsource`, which does not work on frozen modules.

### Windows — single-file MSI

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean OBSCURA3D_windows.spec
dist\OBSCURA3D\OBSCURA3D.exe --smoke-test
python build_msi.py          # requires WiX v3 (candle/light) or WiX v4 (wix)
```

Produces `dist/OBSCURA3D-<version>-win64.msi` — a single installer with
license acceptance, install-directory selection, and desktop/start-menu
shortcuts.

### macOS — DMG (Apple Silicon)

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean OBSCURA3D_mac.spec
dist/OBSCURA3D.app/Contents/MacOS/OBSCURA3D --smoke-test
mkdir -p dist/dmg && cp -R dist/OBSCURA3D.app dist/dmg/
ln -s /Applications dist/dmg/Applications
hdiutil create -volname OBSCURA3D -srcfolder dist/dmg -ov -format UDZO dist/OBSCURA3D-macOS-Apple-Silicon.dmg
```

### Linux — tar.gz

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean OBSCURA3D_linux.spec
dist/OBSCURA3D/OBSCURA3D --smoke-test
cp OBSCURA3D.desktop dist/OBSCURA3D/
tar -czf OBSCURA3D-linux-x86_64.tar.gz -C dist OBSCURA3D
```

## CI

`.github/workflows/build.yml` builds all three artifacts on
`workflow_dispatch` or when pushing a `v*` tag, and publishes a GitHub
release for tagged builds.
