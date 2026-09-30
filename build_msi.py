"""
build_msi.py - Create a Windows MSI installer from the PyInstaller
dist/OBSCURA3D output using WiX Toolset (same approach as vSky2).

Prerequisites:
    - Run PyInstaller first:  pyinstaller --noconfirm --clean OBSCURA3D_windows.spec
    - WiX Toolset v3 installed (https://wixtoolset.org/releases/)
      or WiX v4+ via dotnet tool: dotnet tool install --global wix
    - On PATH: candle.exe and light.exe (WiX v3) or wix.exe (WiX v4+)

Usage:
    conda activate vo_sdf
    python build_msi.py
"""

import os
import re
import sys
import uuid
import shutil
import subprocess
import xml.etree.ElementTree as ET

# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Version read from OBSCURA3D.py (__version__ = "x.y.z") without importing it
with open(os.path.join(PROJECT_DIR, "OBSCURA3D.py"), encoding="utf-8") as _f:
    _m = re.search(r'__version__\s*=\s*["\']([^"\']+)', _f.read())
APP_VERSION = _m.group(1) if _m else "1.0.0"

APP_NAME = "OBSCURA3D"
APP_MANUFACTURER = "Universite de Bourgogne"
APP_DESCRIPTION = "OBSCURA3D - Volumetric Obscurance (VO / VOP / VON)"
APP_EXE = "OBSCURA3D.exe"
UPGRADE_CODE = "7a4b9c2d-3e5f-4a6b-8c1d-2e3f4a5b6c7d"  # Fixed GUID for upgrades

# dist location can be overridden (e.g. when building outside a synced folder)
DIST_DIR = os.environ.get(
    "OBSCURA3D_DIST", os.path.join(PROJECT_DIR, "dist", "OBSCURA3D"))
BUILD_DIR = os.path.join(PROJECT_DIR, "build_msi")
WXS_FILE = os.path.join(BUILD_DIR, "OBSCURA3D.wxs")
WIXOBJ_FILE = os.path.join(BUILD_DIR, "OBSCURA3D.wixobj")
MSI_FILE = os.path.join(
    os.path.dirname(DIST_DIR), f"OBSCURA3D-{APP_VERSION}-win64.msi")

ICON_FILE = os.path.join(PROJECT_DIR, "OBSCURA3D.ico")
LOGO_PNG = os.path.join(PROJECT_DIR, "OBSCURA3D_512x512.png")
BANNER_BMP = os.path.join(BUILD_DIR, "banner.bmp")
DIALOG_BMP = os.path.join(BUILD_DIR, "dialog.bmp")


# ============================================================
# WiX detection
# ============================================================

def find_wix():
    """Detect WiX toolset version and executables.

    WiX v3 is preferred: the generated WXS uses the v3 schema.  WiX v4+
    (dotnet tool) is supported too — the WXS is converted with 'wix convert'.
    """
    candle = shutil.which("candle")
    light = shutil.which("light")
    if candle and light:
        return "v3", (candle, light)

    for wix_dir in [
        os.path.join(os.environ.get("WIX", ""), "bin"),
        r"C:\Program Files (x86)\WiX Toolset v3.14\bin",
        r"C:\Program Files (x86)\WiX Toolset v3.11\bin",
        r"C:\Program Files\WiX Toolset v3.14\bin",
    ]:
        candle = os.path.join(wix_dir, "candle.exe")
        light = os.path.join(wix_dir, "light.exe")
        if os.path.isfile(candle) and os.path.isfile(light):
            return "v3", (candle, light)

    wix4 = shutil.which("wix")
    if wix4:
        return "v4", wix4

    return None, None


# ============================================================
# WXS generation (WiX XML source)
# ============================================================

def generate_unique_id(prefix, path):
    """Generate a deterministic WiX-safe ID from a file path."""
    path_uuid = uuid.uuid5(uuid.NAMESPACE_URL, path).hex[:16]
    return prefix + "_" + path_uuid


def collect_files(base_dir):
    """Walk the dist directory and collect all files grouped by directory."""
    dir_map = {}  # relative_dir -> list of (full_path, filename)
    for root, dirs, files in os.walk(base_dir):
        rel_dir = os.path.relpath(root, base_dir)
        if rel_dir == ".":
            rel_dir = ""
        dir_map[rel_dir] = []
        for f in files:
            full_path = os.path.join(root, f)
            dir_map[rel_dir].append((full_path, f))
    return dir_map


def build_wxs(dist_dir, wxs_path):
    """Generate the WiX XML source file from the PyInstaller output."""

    NS = "http://schemas.microsoft.com/wix/2006/wi"
    ET.register_namespace("", NS)

    def ns(tag):
        return f"{{{NS}}}{tag}"

    # Root
    wix = ET.Element(ns("Wix"))

    # Product
    product = ET.SubElement(wix, ns("Product"), {
        "Id": "*",
        "Name": APP_NAME,
        "Language": "1033",
        "Version": APP_VERSION,
        "Manufacturer": APP_MANUFACTURER,
        "UpgradeCode": UPGRADE_CODE,
    })

    # Package
    ET.SubElement(product, ns("Package"), {
        "InstallerVersion": "500",
        "Compressed": "yes",
        "InstallScope": "perMachine",
        "Description": APP_DESCRIPTION,
        "Manufacturer": APP_MANUFACTURER,
    })

    # Media
    ET.SubElement(product, ns("MediaTemplate"), {
        "EmbedCab": "yes",
    })

    # MajorUpgrade
    ET.SubElement(product, ns("MajorUpgrade"), {
        "DowngradeErrorMessage": "A newer version of [ProductName] is already installed.",
    })

    # Icon for Add/Remove Programs
    if os.path.isfile(ICON_FILE):
        ET.SubElement(product, ns("Icon"), {
            "Id": "AppIcon.ico",
            "SourceFile": ICON_FILE,
        })
        ET.SubElement(product, ns("Property"), {
            "Id": "ARPPRODUCTICON",
            "Value": "AppIcon.ico",
        })

    # UI: license acceptance + installation directory selection
    ET.SubElement(product, ns("UIRef"), {"Id": "WixUI_InstallDir"})
    ET.SubElement(product, ns("Property"), {
        "Id": "WIXUI_INSTALLDIR",
        "Value": "INSTALLDIR",
    })

    # License (generated RTF)
    license_rtf = os.path.join(BUILD_DIR, "license.rtf")
    _create_license_rtf(license_rtf)
    ET.SubElement(product, ns("WixVariable"), {
        "Id": "WixUILicenseRtf",
        "Value": license_rtf,
    })

    # Installer branding images (banner + dialog with logo)
    if os.path.isfile(BANNER_BMP):
        ET.SubElement(product, ns("WixVariable"), {
            "Id": "WixUIBannerBmp",
            "Value": BANNER_BMP,
        })
    if os.path.isfile(DIALOG_BMP):
        ET.SubElement(product, ns("WixVariable"), {
            "Id": "WixUIDialogBmp",
            "Value": DIALOG_BMP,
        })

    # Directory structure
    directory = ET.SubElement(product, ns("Directory"), {
        "Id": "TARGETDIR",
        "Name": "SourceDir",
    })
    pf_dir = ET.SubElement(directory, ns("Directory"), {
        "Id": "ProgramFiles64Folder",
    })
    install_dir = ET.SubElement(pf_dir, ns("Directory"), {
        "Id": "INSTALLDIR",
        "Name": APP_NAME,
    })

    # Collect files
    dir_map = collect_files(dist_dir)
    component_ids = []
    dir_id_map = {"": "INSTALLDIR"}

    # First pass: create all directories
    for rel_dir in sorted(dir_map.keys()):
        if rel_dir == "":
            continue
        parts = rel_dir.split(os.sep)
        current = install_dir
        current_path = ""
        for part in parts:
            current_path = os.path.join(current_path, part) if current_path else part
            dir_id = generate_unique_id("Dir", current_path)
            if current_path not in dir_id_map:
                current = ET.SubElement(current, ns("Directory"), {
                    "Id": dir_id,
                    "Name": part,
                })
                dir_id_map[current_path] = dir_id
            else:
                for child in current:
                    if child.get("Id") == dir_id_map[current_path]:
                        current = child
                        break

    # Second pass: create components with files using DirectoryRef
    for rel_dir, files in dir_map.items():
        if not files:
            continue

        parent_dir_id = dir_id_map.get(rel_dir, "INSTALLDIR")
        dir_ref = ET.SubElement(product, ns("DirectoryRef"), {
            "Id": parent_dir_id,
        })

        for full_path, filename in files:
            rel_file = os.path.join(rel_dir, filename) if rel_dir else filename
            comp_id = generate_unique_id("Cmp", rel_file)
            file_id = generate_unique_id("Fil", rel_file)

            component = ET.SubElement(dir_ref, ns("Component"), {
                "Id": comp_id,
                "Guid": str(uuid.uuid5(uuid.NAMESPACE_URL, rel_file)),
                "Win64": "yes",
            })
            component_ids.append(comp_id)

            file_attrs = {
                "Id": file_id,
                "Source": full_path,
                "KeyPath": "yes",
            }

            # Main exe gets shortcuts
            if filename == APP_EXE and rel_dir == "":
                file_attrs["Name"] = APP_EXE
                ET.SubElement(component, ns("File"), file_attrs)

                # Desktop shortcut
                ET.SubElement(component, ns("Shortcut"), {
                    "Id": "DesktopShortcut",
                    "Name": APP_NAME,
                    "Description": APP_DESCRIPTION,
                    "Directory": "DesktopFolder",
                    "WorkingDirectory": "INSTALLDIR",
                    "Advertise": "yes",
                    **({"Icon": "AppIcon.ico"} if os.path.isfile(ICON_FILE) else {}),
                })

                # Start menu shortcut
                ET.SubElement(component, ns("Shortcut"), {
                    "Id": "StartMenuShortcut",
                    "Name": APP_NAME,
                    "Description": APP_DESCRIPTION,
                    "Directory": "ProgramMenuFolder",
                    "WorkingDirectory": "INSTALLDIR",
                    "Advertise": "yes",
                    **({"Icon": "AppIcon.ico"} if os.path.isfile(ICON_FILE) else {}),
                })
            else:
                ET.SubElement(component, ns("File"), file_attrs)

    # Shortcut directories
    ET.SubElement(directory, ns("Directory"), {"Id": "DesktopFolder", "Name": "Desktop"})
    ET.SubElement(directory, ns("Directory"), {"Id": "ProgramMenuFolder", "Name": "Programs"})

    # Feature
    feature = ET.SubElement(product, ns("Feature"), {
        "Id": "Complete",
        "Title": APP_NAME,
        "Level": "1",
    })
    for comp_id in component_ids:
        ET.SubElement(feature, ns("ComponentRef"), {"Id": comp_id})

    # Write WXS
    tree = ET.ElementTree(wix)
    ET.indent(tree, space="  ")
    tree.write(wxs_path, encoding="utf-8", xml_declaration=True)
    print(f"Generated WXS: {wxs_path}")
    print(f"  Components: {len(component_ids)}")


def _create_installer_images():
    """Generate banner (493x58) and dialog (493x312) BMP images with the app logo."""
    try:
        from PIL import Image as PILImage

        os.makedirs(BUILD_DIR, exist_ok=True)

        if os.path.isfile(LOGO_PNG):
            logo = PILImage.open(LOGO_PNG).convert("RGBA")
        elif os.path.isfile(ICON_FILE):
            logo = PILImage.open(ICON_FILE).convert("RGBA")
        else:
            print("WARNING: No logo found, skipping installer images")
            return

        # Banner: 493x58, logo on the right
        banner = PILImage.new("RGB", (493, 58), (255, 255, 255))
        logo_h = 50
        logo_resized = logo.resize((logo_h, logo_h), PILImage.LANCZOS)
        banner.paste(logo_resized, (493 - logo_h - 4, 4), logo_resized)
        banner.save(BANNER_BMP)
        print(f"Created banner: {BANNER_BMP}")

        # Dialog: 493x312, logo centered on left panel
        dialog = PILImage.new("RGB", (493, 312), (255, 255, 255))
        for x in range(164):
            for y in range(312):
                dialog.putpixel((x, y), (41, 65, 122))  # dark blue
        logo_d = 120
        logo_dialog = logo.resize((logo_d, logo_d), PILImage.LANCZOS)
        paste_x = (164 - logo_d) // 2
        paste_y = (312 - logo_d) // 2
        dialog.paste(logo_dialog, (paste_x, paste_y), logo_dialog)
        dialog.save(DIALOG_BMP)
        print(f"Created dialog: {DIALOG_BMP}")

    except ImportError:
        print("WARNING: Pillow not available, skipping installer images")
    except Exception as e:
        print(f"WARNING: Could not create installer images: {e}")


def _create_license_rtf(rtf_path):
    """Create the license RTF shown in the MSI license-acceptance page."""
    rtf_content = r"""{\rtf1\ansi\deff0
{\fonttbl{\f0\fswiss Helvetica;}}
\f0\fs20
\b OBSCURA3D - Volumetric Obscurance\b0\par
\par
Copyright (c) 2026 Universite de Bourgogne\par
Fabrice Monna\par
\par
This software is distributed under the GNU General Public License v3.0.\par
\par
You are free to use, modify, and redistribute this software under the terms of the GPL v3.0 license.\par
\par
For more information, see https://www.gnu.org/licenses/gpl-3.0.html\par
}"""
    with open(rtf_path, 'w', encoding='utf-8') as f:
        f.write(rtf_content)


# ============================================================
# Build MSI
# ============================================================

def build_msi_v3(candle_exe, light_exe):
    """Build MSI using WiX v3 (candle + light)."""
    print("\n--- Compiling WXS (candle) ---")
    result = subprocess.run([
        candle_exe,
        "-arch", "x64",
        "-ext", "WixUIExtension",
        "-o", WIXOBJ_FILE,
        WXS_FILE,
    ], cwd=BUILD_DIR)
    if result.returncode != 0:
        print("ERROR: candle failed!")
        sys.exit(result.returncode)

    print("\n--- Linking MSI (light) ---")
    result = subprocess.run([
        light_exe,
        "-ext", "WixUIExtension",
        "-o", MSI_FILE,
        WIXOBJ_FILE,
    ], cwd=BUILD_DIR)
    if result.returncode != 0:
        print("ERROR: light failed!")
        sys.exit(result.returncode)


def build_msi_v4(wix_exe):
    """Build MSI using WiX v4+ (wix build)."""
    # The WXS is written in the v3 schema — convert it for WiX v4+
    print("\n--- Converting WXS to WiX v4 schema ---")
    result = subprocess.run(
        [wix_exe, "convert", WXS_FILE], cwd=BUILD_DIR)
    if result.returncode != 0:
        print("ERROR: wix convert failed!")
        sys.exit(result.returncode)

    print("\n--- Building MSI (wix v4+) ---")
    result = subprocess.run([
        wix_exe, "build",
        "-arch", "x64",
        "-ext", "WixToolset.UI.wixext",
        "-o", MSI_FILE,
        WXS_FILE,
    ], cwd=BUILD_DIR)
    if result.returncode != 0:
        print("ERROR: wix build failed! (if the UI extension is missing: "
              "'wix extension add -g WixToolset.UI.wixext')")
        sys.exit(result.returncode)


# ============================================================
# Main
# ============================================================

def main():
    if not os.path.isdir(DIST_DIR):
        print(f"ERROR: PyInstaller output not found at {DIST_DIR}")
        print("Run 'pyinstaller --noconfirm --clean OBSCURA3D_windows.spec' first.")
        sys.exit(1)

    wix_version, wix_tools = find_wix()
    if wix_version is None:
        print("ERROR: WiX Toolset not found!")
        print("Install WiX v3: https://wixtoolset.org/releases/")
        print("Or WiX v4+: dotnet tool install --global wix")
        sys.exit(1)

    print(f"WiX version: {wix_version}")
    if wix_version == "v3":
        print(f"  candle: {wix_tools[0]}")
        print(f"  light:  {wix_tools[1]}")
    else:
        print(f"  wix: {wix_tools}")

    os.makedirs(BUILD_DIR, exist_ok=True)

    total_files = sum(len(files) for _, _, files in os.walk(DIST_DIR))
    print(f"\nDist directory: {DIST_DIR}")
    print(f"Total files to package: {total_files}")

    print("\n--- Creating installer images ---")
    _create_installer_images()

    print("\n--- Generating WXS ---")
    build_wxs(DIST_DIR, WXS_FILE)

    if wix_version == "v3":
        build_msi_v3(wix_tools[0], wix_tools[1])
    else:
        build_msi_v4(wix_tools)

    if os.path.isfile(MSI_FILE):
        size_mb = os.path.getsize(MSI_FILE) / (1024 * 1024)
        print("\n" + "=" * 60)
        print("MSI BUILD SUCCESSFUL!")
        print(f"  Output: {MSI_FILE}")
        print(f"  Size:   {size_mb:.1f} MB")
        print("=" * 60)
    else:
        print("\nERROR: MSI file was not created!")
        sys.exit(1)


if __name__ == '__main__':
    main()
