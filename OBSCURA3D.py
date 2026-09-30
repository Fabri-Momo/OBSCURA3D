"""
OBSCURA3D — Volumetric Obscurance, cross-platform.

Modes: VO (full sphere), VOP (positive hemisphere), VON (negative hemisphere).

Auto-detected backend:
  1. Warp CUDA  — Windows/Linux NVIDIA  (best)
  2. Warp CPU   — Mac Apple Silicon / any OS without CUDA
  3. Open3D     — fallback if Warp is missing (RaycastingScene, multithreaded CPU)
  4. NumPy      — last resort if neither Warp nor Open3D
  5. metal_hybrid — reserved slot for Apple Silicon GPU (opt-in: --backend metal_hybrid)

Mac install :  pip install warp-lang open3d
PC install  :  pip install warp-lang   (CUDA auto-detected)
"""

import sys
import os
import platform
import traceback
import numpy as np
import trimesh
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox,
    QCheckBox, QTextEdit, QFileDialog, QGroupBox, QAction,
    QStatusBar, QProgressBar, QSizePolicy, QFrame,
    QComboBox, QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QTextCursor, QIcon

__version__ = "1.0.0"

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _PYQTGRAPH_OK = True
except ImportError:
    _PYQTGRAPH_OK = False
    print("WARNING: pyqtgraph not available -> 3D visualization disabled")


# ---------------------------------------------------------------------------
# Available backend detection
# ---------------------------------------------------------------------------

_BACKEND   = "numpy"   # updated below
_WP_DEVICE = "cpu"

# --backend warp_cuda|warp_cpu|open3d|numpy  to force a backend
_FORCE_BACKEND = None
for _i, _arg in enumerate(sys.argv[1:]):
    if _arg == "--backend" and _i + 1 < len(sys.argv) - 1:
        _FORCE_BACKEND = sys.argv[_i + 2]

try:
    import warp as wp
    wp.init()
    if wp.is_device_available("cuda"):
        _WP_DEVICE = "cuda"
        _BACKEND   = "warp_cuda"
    else:
        _WP_DEVICE = "cpu"
        _BACKEND   = "warp_cpu"
except Exception:
    wp = None

try:
    import open3d as o3d
    _ = o3d.t.geometry.RaycastingScene()
    _O3D_OK = True
except Exception:
    o3d = None
    _O3D_OK = False

if _BACKEND == "numpy" and _O3D_OK:
    _BACKEND = "open3d"

# metal_hybrid is a reserved backend: only relevant on Apple Silicon macOS
# and only "available" when the metal_hybrid module is actually importable.
_METAL_OK = False
if platform.system() == "Darwin" and platform.machine() == "arm64":
    try:
        import metal_hybrid as _mh  # noqa: F401
        _METAL_OK = True
    except Exception:
        _METAL_OK = False

# Force the backend if requested on the command line
if _FORCE_BACKEND in ("warp_cuda", "warp_cpu", "open3d", "numpy", "metal_hybrid"):
    if _FORCE_BACKEND in ("warp_cuda", "warp_cpu") and wp is None:
        print(f"WARNING: Warp not available, cannot force {_FORCE_BACKEND}")
    elif _FORCE_BACKEND == "open3d" and not _O3D_OK:
        print("WARNING: Open3D not available, cannot force open3d")
    elif _FORCE_BACKEND == "metal_hybrid" and not _METAL_OK:
        print("WARNING: metal_hybrid is reserved for Apple Silicon macOS with "
              "the metal-hybrid package installed — cannot force it")
    else:
        _BACKEND = _FORCE_BACKEND
        if _FORCE_BACKEND == "warp_cuda":   _WP_DEVICE = "cuda"
        elif _FORCE_BACKEND == "warp_cpu":  _WP_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Warp kernels — live in obscura_kernels.py, a real .py file shipped as a data
# file in frozen builds so inspect.getsource works (Warp/Numba need source).
# ---------------------------------------------------------------------------

if wp is not None:
    try:
        from obscura_kernels import (
            _wp_build_tangent_frames, _wp_vo_kernel, _wp_vo_finalize)
    except Exception:
        print("WARNING: obscura_kernels.py not found — Warp backend disabled")
        wp = None
        if _BACKEND in ("warp_cuda", "warp_cpu"):
            _BACKEND = "open3d" if _O3D_OK else "numpy"


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def fibonacci_disk(n, radius):
    k = np.arange(n)
    golden_angle = np.pi * (3 - np.sqrt(5))
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * golden_angle
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)]), r


def precompute_local_offsets(disk_xy, disk_r, radius, voxel_step):
    z_values = np.arange(-radius, radius + voxel_step * 0.5, voxel_step)
    offsets = []
    for (x, y), rho in zip(disk_xy, disk_r):
        h = np.sqrt(max(radius**2 - rho**2, 0.0))
        z = z_values[(z_values >= -h) & (z_values <= h)]
        if len(z) == 0:
            continue
        offsets.append(np.column_stack([
            np.full(len(z), x), np.full(len(z), y), z,
        ]))
    return np.vstack(offsets).astype(np.float32)


def build_tangent_frames_np(normals):
    N     = len(normals)
    n_len = np.linalg.norm(normals, axis=1)
    valid = (n_len > 1e-12).astype(np.int32)
    n = normals.copy()
    n[valid == 1] /= n_len[valid == 1, None]
    ref = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(np.float32)
    ref[np.abs(np.einsum('ij,ij->i', n, ref)) > 0.9] = [0.0, 1.0, 0.0]
    t1 = np.cross(n, ref)
    t1_len = np.linalg.norm(t1, axis=1, keepdims=True)
    t1 /= np.where(t1_len < 1e-12, 1.0, t1_len)
    t2 = np.cross(n, t1)
    t2_len = np.linalg.norm(t2, axis=1, keepdims=True)
    t2 /= np.where(t2_len < 1e-12, 1.0, t2_len)
    return t1.astype(np.float32), t2.astype(np.float32), valid, n.astype(np.float32)


def _fractional_inside(signed_distance, voxel_step, valid, invert=False):
    inside = np.clip(0.5 - np.asarray(signed_distance) / voxel_step, 0.0, 1.0)
    if invert:
        inside = 1.0 - inside
    return inside.astype(np.float32) * np.asarray(valid, dtype=np.float32)


def _finalize_vo(lin, lout, lin_pos, lout_pos, lin_neg, lout_neg, valid, mode):
    """Convert inside/outside accumulators into a per-vertex VO scalar.

    mode: "vo" (full sphere), "vop" (positive hemisphere) or "von" (negative).
    """
    mode = str(mode).lower()
    with np.errstate(invalid='ignore', divide='ignore'):
        if mode == "vo":
            vo = np.where((valid != 0) & (lin > 0.0), lout / lin, np.nan)
        elif mode == "vop":
            total_pos = lin_pos + lout_pos
            vo = np.where((valid != 0) & (total_pos > 0.0), lout_pos / total_pos, np.nan)
        elif mode == "von":
            total_neg = lin_neg + lout_neg
            vo = np.where((valid != 0) & (total_neg > 0.0), lin_neg / total_neg, np.nan)
        else:
            raise ValueError(f"Unknown vo_mode: {mode!r}")
    return vo.astype(np.float32)


# Numba rasterizer — also in obscura_kernels.py (same inspect.getsource reason)
try:
    import obscura_kernels as _ok_mod
    _rasterize_numba = getattr(_ok_mod, '_rasterize_numba', None)
    _NUMBA_OK = getattr(_ok_mod, '_NUMBA_OK', False) and _rasterize_numba is not None
    _NUMBA_ERR = getattr(_ok_mod, '_NUMBA_ERR', '')
except Exception as _e:
    _rasterize_numba = None
    _NUMBA_OK = False
    _NUMBA_ERR = str(_e)


def rasterize_vo_to_texture(uv, faces, vo_values_gray, texture_size=2048, log_fn=None):
    """Barycentric rasterization — Numba JIT parallel if available, else numpy chunks.
    No matplotlib/scipy. Works with all UV layouts.
    """
    if isinstance(texture_size, tuple):
        tex_w, tex_h = int(texture_size[0]), int(texture_size[1])
    else:
        tex_w = tex_h = int(texture_size)
    if log_fn:
        log_fn(f"Bake texture {tex_w}x{tex_h} ({len(uv)} UV pts, {len(faces)} tri)…")

    uv0 = uv[faces[:,0]]; uv1 = uv[faces[:,1]]; uv2 = uv[faces[:,2]]
    cross = ((uv1[:,0]-uv0[:,0])*(uv2[:,1]-uv0[:,1]) -
             (uv1[:,1]-uv0[:,1])*(uv2[:,0]-uv0[:,0]))
    tri_idx = np.where(np.abs(cross) > 1e-10)[0]
    if log_fn:
        log_fn(f"Valid UV triangles: {len(tri_idx)}/{len(faces)}")

    # UV → pixels  (V flipped)
    au = (uv0[tri_idx, 0] * (tex_w - 1)).astype(np.float32)
    av = ((1.0 - uv0[tri_idx, 1]) * (tex_h - 1)).astype(np.float32)
    bu = (uv1[tri_idx, 0] * (tex_w - 1)).astype(np.float32)
    bv = ((1.0 - uv1[tri_idx, 1]) * (tex_h - 1)).astype(np.float32)
    cu = (uv2[tri_idx, 0] * (tex_w - 1)).astype(np.float32)
    cv = ((1.0 - uv2[tri_idx, 1]) * (tex_h - 1)).astype(np.float32)

    v0 = vo_values_gray[faces[tri_idx, 0]].astype(np.float32)
    v1 = vo_values_gray[faces[tri_idx, 1]].astype(np.float32)
    v2 = vo_values_gray[faces[tri_idx, 2]].astype(np.float32)

    # Barycentric denominator (F,)
    denom = (bv - cv)*(au - cu) + (cu - bu)*(av - cv)
    ok = np.abs(denom) > 1e-8
    tri_idx = tri_idx[ok]
    au=au[ok]; av=av[ok]; bu=bu[ok]; bv=bv[ok]; cu=cu[ok]; cv=cv[ok]
    v0=v0[ok]; v1=v1[ok]; v2=v2[ok]; denom=denom[ok]

    # Integer bounding boxes of each triangle (F,)
    c0 = np.maximum(0,       np.floor(np.minimum(au, np.minimum(bu, cu))).astype(np.int32))
    c1 = np.minimum(tex_w-1, np.ceil( np.maximum(au, np.maximum(bu, cu))).astype(np.int32))
    r0 = np.maximum(0,       np.floor(np.minimum(av, np.minimum(bv, cv))).astype(np.int32))
    r1 = np.minimum(tex_h-1, np.ceil( np.maximum(av, np.maximum(bv, cv))).astype(np.int32))

    result  = np.full((tex_h, tex_w), 128.0, dtype=np.float32)
    covered = np.zeros((tex_h, tex_w), dtype=np.uint8)

    if not _NUMBA_OK and log_fn:
        log_fn(f"⚠ Numba unavailable ({_NUMBA_ERR}) → numpy fallback")

    if _NUMBA_OK:
        import numba as _nb
        nthreads = _nb.get_num_threads()
        pixel_areas = (c1 - c0 + 1) * (r1 - r0 + 1)
        if log_fn: log_fn(f"Method: Numba JIT parallel ({nthreads} threads) — "
                          f"{len(denom):,} triangles, avg bbox={pixel_areas.mean():.1f}px², "
                          f"max={pixel_areas.max()}px²")
        # Force memory contiguity — required by Numba parallel
        au=np.ascontiguousarray(au); av=np.ascontiguousarray(av)
        bu=np.ascontiguousarray(bu); bv=np.ascontiguousarray(bv)
        cu=np.ascontiguousarray(cu); cv=np.ascontiguousarray(cv)
        v0=np.ascontiguousarray(v0); v1=np.ascontiguousarray(v1)
        v2=np.ascontiguousarray(v2); denom=np.ascontiguousarray(denom)
        c0=np.ascontiguousarray(c0); c1=np.ascontiguousarray(c1)
        r0=np.ascontiguousarray(r0); r1=np.ascontiguousarray(r1)
        _rasterize_numba(au, av, bu, bv, cu, cv, v0, v1, v2, denom,
                         c0, c1, r0, r1, result, covered, tex_w, tex_h)
        result = np.clip(result, 0, 255).astype(np.uint8)
        fill_ratio = covered.astype(bool).mean()
        if log_fn:
            log_fn(f"Texture generated ({fill_ratio*100:.1f}% covered)")
        return np.stack([result, result, result], axis=-1)

    # Fallback numpy chunks
    CHUNK = 512
    F = len(denom)
    for start in range(0, F, CHUNK):
        end = min(start + CHUNK, F)
        sl  = slice(start, end)
        C   = end - start   # triangles in this chunk

        # Bbox width/height of each triangle
        bw = (c1[sl] - c0[sl] + 1)   # (C,)
        bh = (r1[sl] - r0[sl] + 1)

        # Enumerate all pixels of each bbox: cartesian product via repeat/tile
        # pixel_count per triangle = bw * bh
        counts = bw * bh                          # (C,)
        total  = int(counts.sum())
        if total == 0:
            continue

        # Triangle index for each pixel
        tri_rep = np.repeat(np.arange(C), counts)  # (total,)

        # Local coordinates (dc, dr) inside each bbox
        dc_all = np.concatenate([np.tile(np.arange(bw[i]), bh[i]) for i in range(C)])
        dr_all = np.concatenate([np.repeat(np.arange(bh[i]), bw[i]) for i in range(C)])

        # Absolute pixel coordinates
        px = (c0[sl][tri_rep] + dc_all).astype(np.float32)
        py = (r0[sl][tri_rep] + dr_all).astype(np.float32)

        # Barycentrics
        _au=au[sl][tri_rep]; _av=av[sl][tri_rep]
        _bu=bu[sl][tri_rep]; _bv=bv[sl][tri_rep]
        _cu=cu[sl][tri_rep]; _cv=cv[sl][tri_rep]
        _d =denom[sl][tri_rep]

        w0 = ((_bv-_cv)*(px-_cu) + (_cu-_bu)*(py-_cv)) / _d
        w1 = ((_cv-_av)*(px-_cu) + (_au-_cu)*(py-_cv)) / _d
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        ri = py[inside].astype(np.int32)
        ci = px[inside].astype(np.int32)
        ti = tri_rep[inside]
        vals = w0[inside]*v0[sl][ti] + w1[inside]*v1[sl][ti] + w2[inside]*v2[sl][ti]
        result [ri, ci] = vals
        covered[ri, ci] = np.uint8(1)

    result = np.clip(result, 0, 255).astype(np.uint8)
    fill_ratio = covered.astype(bool).mean()
    if log_fn:
        log_fn(f"Method: numpy chunks (barycentric)")
        log_fn(f"Texture generated ({fill_ratio*100:.1f}% covered)")
    return np.stack([result, result, result], axis=-1)


def _get_uv(mesh):
    v = mesh.visual
    if hasattr(v, 'uv') and v.uv is not None:
        return np.asarray(v.uv, dtype=np.float64)
    if hasattr(v, 'to_texture') and callable(v.to_texture):
        try:
            tv = v.to_texture()
            if hasattr(tv, 'uv') and tv.uv is not None:
                return np.asarray(tv.uv, dtype=np.float64)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# VO computation — Warp backend
# ---------------------------------------------------------------------------

def _compute_vo_warp(mesh, vertices, normals, local_offsets, radius,
                     voxel_step, invert, log_fn, progress_fn, vo_mode="vo",
                     cancel_fn=None):
    N = len(vertices)
    M = len(local_offsets)
    max_dist = radius * 3.0
    faces = np.asarray(mesh.faces).flatten().astype(np.int32)
    mode_int = {"vo": 0, "vop": 1, "von": 2}.get(str(vo_mode).lower(), 0)

    wp_mesh = wp.Mesh(
        points=wp.array(vertices,  dtype=wp.vec3,  device=_WP_DEVICE),
        indices=wp.array(faces,    dtype=wp.int32, device=_WP_DEVICE),
    )
    wp_verts    = wp.array(vertices,      dtype=wp.vec3,    device=_WP_DEVICE)
    wp_norms    = wp.array(normals,       dtype=wp.vec3,    device=_WP_DEVICE)
    wp_offsets  = wp.array(local_offsets, dtype=wp.vec3,    device=_WP_DEVICE)
    wp_t1       = wp.zeros(N, dtype=wp.vec3,    device=_WP_DEVICE)
    wp_t2       = wp.zeros(N, dtype=wp.vec3,    device=_WP_DEVICE)
    wp_valid    = wp.zeros(N, dtype=wp.int32,   device=_WP_DEVICE)
    wp_lin      = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_lout     = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_lin_pos  = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_lout_pos = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_lin_neg  = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_lout_neg = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_vo       = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)

    progress_fn(35)
    wp.launch(kernel=_wp_build_tangent_frames, dim=N,
              inputs=[wp_norms, wp_t1, wp_t2, wp_valid], device=_WP_DEVICE)

    if cancel_fn is not None and cancel_fn():
        raise _Cancelled()
    log_fn(f"Launching Warp {'CUDA GPU' if _WP_DEVICE == 'cuda' else 'CPU'} kernel "
           f"({N} × {M} = {N*M:,} threads)…")
    wp.launch(kernel=_wp_vo_kernel, dim=(N, M),
              inputs=[
                  wp_mesh.id, wp_verts, wp_norms, wp_t1, wp_t2, wp_valid,
                  wp_offsets, wp_lin, wp_lout,
                  wp_lin_pos, wp_lout_pos, wp_lin_neg, wp_lout_neg,
                  wp.float32(max_dist), wp.float32(voxel_step),
                  wp.int32(1 if invert else 0),
              ], device=_WP_DEVICE)
    wp.launch(kernel=_wp_vo_finalize, dim=N,
              inputs=[wp_lin, wp_lout,
                      wp_lin_pos, wp_lout_pos, wp_lin_neg, wp_lout_neg,
                      wp_valid, wp_vo, wp.int32(mode_int)], device=_WP_DEVICE)
    wp.synchronize()
    if cancel_fn is not None and cancel_fn():
        raise _Cancelled()
    log_fn("Warp kernel done.")
    progress_fn(80)

    vo = wp_vo.numpy()
    vo[vo < 0.0] = np.nan
    return vo


# ---------------------------------------------------------------------------
# VO computation — Open3D backend (RaycastingScene, multithreaded CPU)
# ---------------------------------------------------------------------------

def _compute_vo_open3d(mesh, vertices, normals, local_offsets, radius,
                       voxel_step, invert, log_fn, progress_fn, vo_mode="vo",
                       cancel_fn=None):
    import open3d as o3d
    N = len(vertices)
    M = len(local_offsets)

    log_fn("Backend: Open3D RaycastingScene (exact BVH, multithreaded CPU)")

    # Build the Open3D scene
    verts_o3d = o3d.core.Tensor(vertices,             dtype=o3d.core.float32)
    faces_o3d = o3d.core.Tensor(
        np.asarray(mesh.faces, dtype=np.uint32),       dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(verts_o3d, faces_o3d)
    log_fn(f"Open3D BVH built ({len(mesh.faces):,} triangles).")
    progress_fn(35)

    # NumPy tangent frames
    t1, t2, valid, normals_n = build_tangent_frames_np(normals)

    lin      = np.zeros(N, dtype=np.float32)
    lout     = np.zeros(N, dtype=np.float32)
    lin_pos  = np.zeros(N, dtype=np.float32)
    lout_pos = np.zeros(N, dtype=np.float32)
    lin_neg  = np.zeros(N, dtype=np.float32)
    lout_neg = np.zeros(N, dtype=np.float32)

    log_fn(f"Open3D VO computation ({N} × {M} = {N*M:,} queries)…")
    chunk = 500_000   # queries per batch to avoid OOM

    all_q = []
    vid_list = []
    oz_all = []
    for oid in range(M):
        ox, oy, oz = local_offsets[oid]
        qx = vertices[:,0] + ox*t1[:,0] + oy*t2[:,0] + oz*normals_n[:,0]
        qy = vertices[:,1] + ox*t1[:,1] + oy*t2[:,1] + oz*normals_n[:,1]
        qz = vertices[:,2] + ox*t1[:,2] + oy*t2[:,2] + oz*normals_n[:,2]
        all_q.append(np.column_stack([qx, qy, qz]))
        vid_list.append(np.arange(N, dtype=np.int32))
        oz_all.append(np.full(N, oz, dtype=np.float32))

    all_q   = np.concatenate(all_q,    axis=0).astype(np.float32)  # (N*M, 3)
    vid_all = np.concatenate(vid_list, axis=0)                     # (N*M,)
    oz_all  = np.concatenate(oz_all,   axis=0)                     # (N*M,)
    total   = len(all_q)

    processed = 0
    while processed < total:
        if cancel_fn is not None and cancel_fn():
            raise _Cancelled()
        end = min(processed + chunk, total)
        q_batch = all_q[processed:end]
        v_batch = vid_all[processed:end]
        oz      = oz_all[processed:end]

        q_t   = o3d.core.Tensor(q_batch, dtype=o3d.core.float32)
        # compute_signed_distance uses winding number → works on open meshes
        sd = scene.compute_signed_distance(q_t).numpy()   # (batch,)

        valid_b = valid[v_batch].astype(np.float32)
        inside  = _fractional_inside(sd, voxel_step, valid_b, invert)
        outside = (1.0 - inside) * valid_b
        pos = ((oz >= 0.0) & (valid_b > 0.0)).astype(np.float32)
        neg = ((oz <= 0.0) & (valid_b > 0.0)).astype(np.float32)

        np.add.at(lin,      v_batch, inside  * voxel_step)
        np.add.at(lout,     v_batch, outside * voxel_step)
        np.add.at(lin_pos,  v_batch, inside  * voxel_step * pos)
        np.add.at(lout_pos, v_batch, outside * voxel_step * pos)
        np.add.at(lin_neg,  v_batch, inside  * voxel_step * neg)
        np.add.at(lout_neg, v_batch, outside * voxel_step * neg)

        processed = end
        pct = 35 + int(45 * processed / total)
        progress_fn(pct)

    log_fn("Open3D computation done.")
    progress_fn(80)

    return _finalize_vo(lin, lout, lin_pos, lout_pos, lin_neg, lout_neg, valid, vo_mode)


# ---------------------------------------------------------------------------
# VO computation — pure numpy fallback
# ---------------------------------------------------------------------------

def _compute_vo_numpy(mesh, vertices, normals, local_offsets, radius,
                      voxel_step, invert, log_fn, progress_fn, vo_mode="vo",
                      cancel_fn=None):
    N = len(vertices)
    M = len(local_offsets)
    log_fn("Backend: NumPy CPU (fallback — slow)")
    t1, t2, valid, normals_n = build_tangent_frames_np(normals)
    prox = trimesh.proximity.ProximityQuery(mesh)
    lin      = np.zeros(N, dtype=np.float32)
    lout     = np.zeros(N, dtype=np.float32)
    lin_pos  = np.zeros(N, dtype=np.float32)
    lout_pos = np.zeros(N, dtype=np.float32)
    lin_neg  = np.zeros(N, dtype=np.float32)
    lout_neg = np.zeros(N, dtype=np.float32)
    for oid in range(M):
        if cancel_fn is not None and cancel_fn():
            raise _Cancelled()
        ox, oy, oz = local_offsets[oid]
        qx = vertices[:,0] + ox*t1[:,0] + oy*t2[:,0] + oz*normals_n[:,0]
        qy = vertices[:,1] + ox*t1[:,1] + oy*t2[:,1] + oz*normals_n[:,1]
        qz = vertices[:,2] + ox*t1[:,2] + oy*t2[:,2] + oz*normals_n[:,2]
        q_pts = np.column_stack([qx, qy, qz])
        closest, distance, face_idx = prox.on_surface(q_pts)
        face_normals = np.asarray(mesh.face_normals)[face_idx]
        diff = q_pts - closest
        sign = np.sign(np.einsum('ij,ij->i', diff, face_normals))
        inside  = _fractional_inside(sign * distance, voxel_step, valid, invert)
        outside = (1.0 - inside) * valid
        pos = (oz >= 0.0).astype(np.float32) * valid
        neg = (oz <= 0.0).astype(np.float32) * valid
        lin      += inside  * voxel_step
        lout     += outside * voxel_step
        lin_pos  += inside  * voxel_step * pos
        lout_pos += outside * voxel_step * pos
        lin_neg  += inside  * voxel_step * neg
        lout_neg += outside * voxel_step * neg
        progress_fn(35 + int(45 * oid / M))
    progress_fn(80)
    return _finalize_vo(lin, lout, lin_pos, lout_pos, lin_neg, lout_neg, valid, vo_mode)


# ---------------------------------------------------------------------------
# 3D visualization window
# ---------------------------------------------------------------------------

if _PYQTGRAPH_OK:
    def _reset_gl_shader_cache():
        """Drop cached GL shader programs.

        pyqtgraph caches compiled program handles globally (ShaderProgram.prog),
        but those handles become invalid once a GLViewWidget's context is
        destroyed (e.g. after the viewer window is closed).  Clearing the cache
        forces recompilation in the next live context.
        """
        try:
            from pyqtgraph.opengl import shaders as _pg_shaders
            for _sp in list(_pg_shaders.ShaderProgram.names.values()):
                _sp.prog = None
        except Exception:
            pass

    def _make_headlight_shader():
        """Brighter variant of pyqtgraph's 'shaded' program.

        Key light comes from the camera (headlight) with a fill on back
        faces, so the model stays readable from any angle.
        """
        import textwrap
        from pyqtgraph.opengl.shaders import (
            ShaderProgram, VertexShader, FragmentShader)
        return ShaderProgram('obscura_headlight', [
            VertexShader(textwrap.dedent("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """)),
            FragmentShader(textwrap.dedent("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 n = normalize(v_normal);
                    float d = dot(n, normalize(vec3(0.4, 0.4, -1.0)));
                    float p = d > 0.0 ? d : -d * 0.45;
                    vec3 rgb = v_color.rgb * (0.35 + 0.65 * p);
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """)),
        ])

    try:
        _HEADLIGHT_SHADER = _make_headlight_shader()
    except Exception:
        _HEADLIGHT_SHADER = 'shaded'

    class MeshViewer3D(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            _reset_gl_shader_cache()
            self.setWindowTitle("3D Mesh Viewer")
            self.setGeometry(100, 100, 800, 600)
            
            # Central OpenGL widget
            self.gl_widget = gl.GLViewWidget()
            self.setCentralWidget(self.gl_widget)
            
            # Camera setup
            self.gl_widget.setCameraPosition(distance=5, elevation=30, azimuth=45)
            
            # Reference axes
            self.gl_axis = gl.GLAxisItem()
            self.gl_axis.setSize(1, 1, 1)
            self.gl_widget.addItem(self.gl_axis)
            
            # Ground grid
            self.gl_grid = gl.GLGridItem()
            self.gl_grid.scale(0.1, 0.1, 0.1)
            self.gl_widget.addItem(self.gl_grid)
            
            self.mesh_item = None
            self.current_mesh = None
            
            # Instructions
            self.statusBar().showMessage("Navigation: left click = rotate | right click = pan | wheel = zoom")
        
        def load_mesh(self, mesh):
            """Load a trimesh mesh into the 3D view."""
            self.current_mesh = mesh
            
            # Debug info
            print(f"[DEBUG] Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            print(f"[DEBUG] Bounds: min={mesh.vertices.min(axis=0)}, max={mesh.vertices.max(axis=0)}")
            
            # Remove the previous mesh if any
            if self.mesh_item is not None:
                self.gl_widget.removeItem(self.mesh_item)
            
            # Prepare vertices and faces for pyqtgraph
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)

            # Use baked vertex colors when present (e.g. exported VO result)
            vcolors = None
            vis = getattr(mesh, 'visual', None)
            if hasattr(vis, 'vertex_colors') and vis.vertex_colors is not None:
                vc = np.asarray(vis.vertex_colors)
                if vc.shape[0] == len(vertices) and vc.shape[1] >= 3:
                    vcolors = vc.astype(np.float32) / 255.0

            # Create the mesh item — shaded surface (point cloud fallback)
            if len(faces) > 0:
                mesh_kwargs = dict(
                    vertexes=vertices,
                    faces=faces,
                    smooth=True,
                    computeNormals=True,
                    shader=_HEADLIGHT_SHADER,
                    drawEdges=False,
                )
                if vcolors is not None:
                    mesh_kwargs['vertexColors'] = vcolors
                else:
                    mesh_kwargs['color'] = (0.75, 0.75, 0.78, 1.0)  # Light gray
                self.mesh_item = gl.GLMeshItem(**mesh_kwargs)
            else:
                self.mesh_item = gl.GLScatterPlotItem(
                    pos=vertices,
                    size=2.0,
                    color=vcolors if vcolors is not None else (0.75, 0.75, 0.78, 1.0),  # Light gray
                    pxMode=True,
                )
            
            self.gl_widget.addItem(self.mesh_item)
            
            # Center the view on the mesh
            self.center_view()
            
            # Debug camera
            bbox = mesh.bounding_box
            print(f"[DEBUG] Center: {bbox.centroid}, Extents: {bbox.extents}")
        
        def center_view(self):
            """Center the camera on the mesh."""
            if self.current_mesh is None:
                return
            
            # Compute mesh center and size
            center = self.current_mesh.bounding_box.centroid
            extents = self.current_mesh.bounding_box.extents
            
            # Camera distance based on mesh size
            max_extent = np.max(extents)
            if max_extent < 1e-6:
                max_extent = 1.0  # Avoid distance = 0 for a tiny mesh
            distance = max_extent * 3.0
            
            # Position the camera (center is not supported in this version)
            self.gl_widget.setCameraPosition(
                distance=distance,
                elevation=30,
                azimuth=45
            )
            # Alternative: move the mesh to the origin for visualization
            if self.mesh_item is not None:
                self.mesh_item.translate(-center[0], -center[1], -center[2])
            
            # Also scale the grid and axes
            scale = max_extent * 0.5
            self.gl_axis.setSize(scale, scale, scale)
            self.gl_grid.scale(scale/10, scale/10, scale/10)
        
        def keyPressEvent(self, event):
            """Keyboard shortcuts."""
            if event.key() == Qt.Key_R:
                self.center_view()
            elif event.key() == Qt.Key_Escape:
                self.close()

        def closeEvent(self, event):
            # Hide instead of destroying: keeps the GL context alive so the
            # cached shader programs stay valid when the window is reopened.
            event.ignore()
            self.hide()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def clean_mesh(mesh, log_fn=None, progress_fn=None):
    """Robust mesh cleanup: normals, duplicates, degenerate faces."""
    changes = []
    original_v = len(mesh.vertices)
    original_f = len(mesh.faces)
    
    if progress_fn:
        progress_fn(2)
    
    # 1. Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) != original_v:
        changes.append(f"vertices {original_v}→{len(mesh.vertices)}")
        if log_fn:
            log_fn(f"  • Unreferenced vertices removed ({original_v}→{len(mesh.vertices)})")
    
    if progress_fn:
        progress_fn(4)
    
    # 2. Remove degenerate faces (zero area)
    non_degenerate = mesh.area_faces > 1e-12
    if not non_degenerate.all():
        mesh.update_faces(non_degenerate)
        changes.append(f"degenerate faces removed")
        if log_fn:
            log_fn(f"  • Degenerate faces removed ({(~non_degenerate).sum()} faces)")
    
    if progress_fn:
        progress_fn(6)
    
    # 3. Merge duplicate vertices (tolerance 1e-8)
    # mesh.merge_vertices() handles detection automatically
    mesh.merge_vertices()
    changes.append("duplicate vertices merged")
    if log_fn:
        log_fn("  • Duplicate vertices merged")
    
    if progress_fn:
        progress_fn(8)
    
    # 4. Recompute normals (more reliable than imported normals)
    mesh.fix_normals()
    # Force recomputation by clearing the caches
    mesh.face_normals = None
    mesh.vertex_normals = None
    changes.append("normals recomputed")
    if log_fn:
        log_fn("  • Normals recomputed")
    
    if progress_fn:
        progress_fn(9)
    
    # 5. Detect and invert if needed (consistent orientation)
    # Negative signed volume on a watertight mesh = inward-pointing normals.
    try:
        if mesh.is_watertight and mesh.volume < 0:
            mesh.invert()
            changes.append("mesh inverted (negative volume)")
            if log_fn:
                log_fn("  • Mesh inverted (negative volume)")
    except Exception:
        if log_fn:
            log_fn("  • Orientation detection skipped (volume failed)")
    
    # 6. Final consistency check
    if not mesh.is_watertight:
        changes.append("⚠ mesh non-watertight")
        if log_fn:
            log_fn("  • ⚠ Non-watertight mesh (OK for VO)")
    # mesh.fix_inconsistent_faces() doesn't exist in trimesh, skip this step
    
    if log_fn and changes:
        log_fn(f"Cleanup done: {len(changes)} fixes → {', '.join(changes)}")
    
    return mesh


def compute_vo_sdf(input_mesh, output_mesh, radius, voxel_step, n_disk, invert,
                   export_mode, texture_size, texture_format, log_fn, progress_fn,
                   vo_mode="vo", cancel_fn=None):

    def _check_cancel():
        if cancel_fn is not None and cancel_fn():
            raise _Cancelled()

    vo_mode = str(vo_mode).lower()
    if vo_mode not in ("vo", "vop", "von"):
        raise ValueError(f"vo_mode must be 'vo', 'vop' or 'von', got {vo_mode!r}")
    progress_fn(0)
    mesh = trimesh.load_mesh(input_mesh, process=False)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The file is not a simple triangular mesh.")
    
    # Systematic cleanup
    mesh = clean_mesh(mesh, log_fn, progress_fn)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The file is not a simple triangular mesh.")

    if export_mode == "texture_uv":
        uv_raw = _get_uv(mesh)
    else:
        uv_raw = None
        mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight:
        log_fn("⚠ Mesh not closed (OK, winding number still works).")

    progress_fn(10)
    vertices = np.asarray(mesh.vertices,      dtype=np.float32)
    normals  = np.asarray(mesh.vertex_normals, dtype=np.float32)
    N = len(vertices)

    disk_xy, disk_r = fibonacci_disk(n_disk, radius)
    local_offsets   = precompute_local_offsets(disk_xy, disk_r, radius, voxel_step)
    M = len(local_offsets)

    progress_fn(20)
    log_fn(f"Vertices: {N}  |  Offsets/vertex: {M}  |  Total: {N*M:,}")
    log_fn(f"Backend: {_BACKEND}  |  Mode: {vo_mode}")

    # --- Dispatch by backend ---
    # Ordered fallback chain: a backend can fail on the target machine
    # (missing CUDA toolkit, driver, or JIT toolchain) — degrade gracefully.
    _check_cancel()
    if _BACKEND == "metal_hybrid":
        log_fn("metal_hybrid is a reserved backend — routed to NumPy "
               "(no Metal kernels in this build)")
    if _BACKEND in ("warp_cuda", "warp_cpu"):
        candidates = ["warp", "open3d", "numpy"]
    elif _BACKEND == "open3d":
        candidates = ["open3d", "numpy"]
    else:
        candidates = ["numpy"]

    vo = None
    last_err = None
    for cand in candidates:
        if cand == "warp" and wp is None:
            continue
        if cand == "open3d" and not _O3D_OK:
            continue
        try:
            if cand == "warp":
                vo = _compute_vo_warp(mesh, vertices, normals, local_offsets,
                                      radius, voxel_step, invert, log_fn,
                                      progress_fn, vo_mode, cancel_fn)
            elif cand == "open3d":
                vo = _compute_vo_open3d(mesh, vertices, normals, local_offsets,
                                        radius, voxel_step, invert, log_fn,
                                        progress_fn, vo_mode, cancel_fn)
            else:
                vo = _compute_vo_numpy(mesh, vertices, normals, local_offsets,
                                       radius, voxel_step, invert, log_fn,
                                       progress_fn, vo_mode, cancel_fn)
            break
        except _Cancelled:
            raise
        except Exception as e:
            last_err = e
            log_fn(f"⚠ {cand} backend failed: {e} → falling back")
    if vo is None:
        raise RuntimeError(f"All compute backends failed (last: {last_err})")

    # --- Common post-processing ---
    _check_cancel()
    finite = np.isfinite(vo)
    if finite.sum() == 0:
        raise RuntimeError("No valid VO value computed.")

    lo, hi = np.nanpercentile(vo[finite], [2, 98])
    log_fn(f"Raw VO   : min={vo[finite].min():.4f}  max={vo[finite].max():.4f}  "
           f"std={vo[finite].std():.4f}  valid={finite.sum()}/{N}")
    log_fn(f"Normalization: p2={lo:.4f}  p98={hi:.4f}")

    gray = np.zeros(N, dtype=np.uint8)
    if hi - lo < 1e-6:
        log_fn("⚠ VO range nearly zero!")
        gray[finite] = 128
    else:
        gray[finite] = np.clip(255*(vo[finite]-lo)/(hi-lo), 0, 255).astype(np.uint8)
    log_fn(f"Gray : min={gray.min()}  max={gray.max()}  unique={len(np.unique(gray))}")
    progress_fn(85)

    if export_mode == "vertex_colors":
        colors = np.column_stack([gray, gray, gray, np.full(N, 255, dtype=np.uint8)])
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        mesh.export(output_mesh)
        progress_fn(100)
        log_fn(f"✔ Mesh exported (vertex colors): {output_mesh}")
    else:
        texture_path = output_mesh
        if uv_raw is None:
            log_fn("⚠ No UVs — cylindrical projection")
            centered = vertices - vertices.mean(axis=0)
            theta    = np.arctan2(centered[:,2], centered[:,0])
            u_c = theta/(2*np.pi) + 0.5
            v_c = (centered[:,1]-centered[:,1].min()) / (centered[:,1].max()-centered[:,1].min()+1e-8)
            uv = np.column_stack([u_c, v_c])
        else:
            uv = uv_raw
            log_fn(f"UVs: {uv.shape[0]} pts")
        if uv.shape[0] != N:
            raise RuntimeError(f"UV mismatch: {uv.shape[0]} vs {N} vertices.")
        uv = uv.copy()
        log_fn(f"Raw UVs: U=[{uv[:,0].min():.4f}, {uv[:,0].max():.4f}]  "
               f"V=[{uv[:,1].min():.4f}, {uv[:,1].max():.4f}]")
        # Only normalize if UVs are outside [0,1] — otherwise keep the original scale
        for i in range(2):
            mn, mx = uv[:,i].min(), uv[:,i].max()
            r = mx - mn
            if r > 1e-6 and (mn < -0.01 or mx > 1.01):
                uv[:,i] = (uv[:,i] - mn) / r
                log_fn(f"UV axis {i} renormalized [{mn:.3f},{mx:.3f}] → [0,1]")
        progress_fn(90)
        texture_img = rasterize_vo_to_texture(uv, mesh.faces, gray, texture_size, log_fn)
        pil_img = Image.fromarray(texture_img)
        if texture_path.lower().endswith(".png"):
            pil_img.save(texture_path, "PNG")
        else:
            pil_img.save(texture_path, "JPEG", quality=95)
        _sz = (f"{texture_size[0]}x{texture_size[1]}" if isinstance(texture_size, tuple)
               else f"{texture_size}x{texture_size}")
        log_fn(f"✔ VO texture exported ({_sz}): {os.path.basename(texture_path)}")
        log_fn("ℹ Apply this texture to your original model in Blender/your 3D software.")
        progress_fn(100)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _Cancelled(Exception):
    """Raised inside compute_vo_sdf when the user requests cancellation."""
    pass


class Worker(QObject):
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error    = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, input_mesh, output_mesh, radius, voxel_step, n_disk, invert,
                 export_mode, texture_size, texture_format, vo_mode="vo"):
        super().__init__()
        self._cancelled    = False
        self.input_mesh    = input_mesh
        self.output_mesh   = output_mesh
        self.radius        = radius
        self.voxel_step    = voxel_step
        self.n_disk        = n_disk
        self.invert        = invert
        self.export_mode   = export_mode
        self.texture_size  = texture_size
        self.texture_format = texture_format
        self.vo_mode       = vo_mode

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            compute_vo_sdf(
                self.input_mesh, self.output_mesh,
                self.radius, self.voxel_step, self.n_disk, self.invert,
                self.export_mode, self.texture_size, self.texture_format,
                log_fn=self.log.emit, progress_fn=self.progress.emit,
                vo_mode=self.vo_mode,
                cancel_fn=lambda: self._cancelled,
            )
            self.finished.emit()
        except _Cancelled:
            self.cancelled.emit()
        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OBSCURA3D")
        self.setMinimumSize(680, 580)
        self._thread = None
        self._worker = None
        self.current_mesh_path = None
        self._build_ui()
        self._build_menu()

    def _build_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        open_act = QAction("&Open mesh…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._browse_input)
        file_menu.addAction(open_act)
        save_act = QAction("Set &output…", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._browse_output)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)
        help_menu = mb.addMenu("&Help")
        about_act = QAction("&About…", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # Backend info banner
        backend_labels = {
            "warp_cuda": "🟢 Backend: Warp CUDA GPU",
            "warp_cpu":  "🟡 Backend: Warp CPU (no CUDA GPU detected)",
            "open3d":    "🟡 Backend: Open3D CPU (Warp missing)",
            "metal_hybrid": "🟡 Backend: metal_hybrid (reserved — routed to NumPy)",
            "numpy":     "🔴 Backend: NumPy CPU (slow — install warp-lang or open3d)",
        }
        banner = QLabel(backend_labels.get(_BACKEND, _BACKEND))
        banner.setStyleSheet("padding: 4px 8px; border-radius: 4px; "
                             "background: #2a2a2a; color: #ddd;")
        root.addWidget(banner)

        # Files group
        files_box = QGroupBox("Files")
        files_layout = QVBoxLayout(files_box)
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input:"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select a mesh (.ply, .obj, .stl…)")
        in_row.addWidget(self.input_edit)
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(btn_in)
        files_layout.addLayout(in_row)
        out_row = QHBoxLayout()
        self.output_label = QLabel("Output: ")
        out_row.addWidget(self.output_label)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output file (.ply)")
        out_row.addWidget(self.output_edit)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        files_layout.addLayout(out_row)
        
        # 3D Viewer button
        viewer_row = QHBoxLayout()
        self.btn_viewer = QPushButton("View 3D")
        self.btn_viewer.setEnabled(False)
        self.btn_viewer.clicked.connect(self._open_3d_viewer)
        if not _PYQTGRAPH_OK:
            self.btn_viewer.setText("View 3D (pyqtgraph required)")
            self.btn_viewer.setEnabled(False)
        viewer_row.addWidget(self.btn_viewer)
        viewer_row.addStretch()
        files_layout.addLayout(viewer_row)
        
        root.addWidget(files_box)

        # Settings group
        settings_box = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_box)
        settings_layout.setSpacing(20)
        v1 = QVBoxLayout()
        v1.addWidget(QLabel("Radius (m)"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.0001, 100.0)
        self.spin_radius.setDecimals(4)
        self.spin_radius.setSingleStep(0.005)
        self.spin_radius.setValue(0.01)
        self.spin_radius.setToolTip("Local radius of the obscurance sphere.")
        v1.addWidget(self.spin_radius)
        settings_layout.addLayout(v1)
        v2 = QVBoxLayout()
        v2.addWidget(QLabel("Step (m)"))
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.00001, 10.0)
        self.spin_step.setDecimals(5)
        self.spin_step.setSingleStep(0.0005)
        self.spin_step.setValue(0.001)
        v2.addWidget(self.spin_step)
        settings_layout.addLayout(v2)
        v3 = QVBoxLayout()
        v3.addWidget(QLabel("Disk samples"))
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(4, 256)
        self.spin_samples.setSingleStep(4)
        self.spin_samples.setValue(16)
        v3.addWidget(self.spin_samples)
        settings_layout.addLayout(v3)
        v4 = QVBoxLayout()
        v4.addWidget(QLabel("Invert sign"))
        self.chk_invert = QCheckBox("Invert")
        v4.addWidget(self.chk_invert)
        v4.addStretch()
        settings_layout.addLayout(v4)
        v_mode = QVBoxLayout()
        v_mode.addWidget(QLabel("Mode"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("VO — full sphere", "vo")
        self.combo_mode.addItem("VOP — positive hemisphere", "vop")
        self.combo_mode.addItem("VON — negative hemisphere", "von")
        self.combo_mode.setToolTip("Openness measure: VO (full sphere), "
                                   "VOP (positive hemisphere) or VON (negative hemisphere)")
        self.combo_mode.currentIndexChanged.connect(self._on_vo_mode_changed)
        v_mode.addWidget(self.combo_mode)
        settings_layout.addLayout(v_mode)
        settings_layout.addStretch()
        root.addWidget(settings_box)

        # Export group
        export_box = QGroupBox("Export options")
        export_layout = QHBoxLayout(export_box)
        export_layout.setSpacing(20)
        v_export = QVBoxLayout()
        v_export.addWidget(QLabel("Export mode"))
        self.combo_export = QComboBox()
        self.combo_export.addItem("Vertex colors", "vertex_colors")
        self.combo_export.addItem("Texture UV", "texture_uv")
        self.combo_export.currentIndexChanged.connect(self._on_export_mode_changed)
        v_export.addWidget(self.combo_export)
        export_layout.addLayout(v_export)
        v_tex_size = QVBoxLayout()
        v_tex_size.addWidget(QLabel("Texture size"))
        self.combo_tex_size = QComboBox()
        self.combo_tex_size.addItem("1024x1024", 1024)
        self.combo_tex_size.addItem("2048x2048", 2048)
        self.combo_tex_size.addItem("4096x4096", 4096)
        self.combo_tex_size.setCurrentIndex(1)
        self.combo_tex_size.setEnabled(False)
        v_tex_size.addWidget(self.combo_tex_size)
        export_layout.addLayout(v_tex_size)
        v_tex_fmt = QVBoxLayout()
        v_tex_fmt.addWidget(QLabel("Texture format"))
        self.combo_tex_fmt = QComboBox()
        self.combo_tex_fmt.addItem("PNG", ".png")
        self.combo_tex_fmt.addItem("JPEG", ".jpg")
        self.combo_tex_fmt.setEnabled(False)
        v_tex_fmt.addWidget(self.combo_tex_fmt)
        export_layout.addLayout(v_tex_fmt)
        export_layout.addStretch()
        root.addWidget(export_box)

        # Run / Cancel
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run computation")
        self.btn_run.setFixedHeight(38)
        font = self.btn_run.font(); font.setPointSize(11); font.setBold(True)
        self.btn_run.setFont(font)
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(38)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        run_row.addWidget(self.btn_cancel, stretch=0)
        root.addLayout(run_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # Log
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_edit)
        root.addWidget(log_box, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(f"Ready — {_BACKEND}")

    def _detect_texture_size(self, path):
        try:
            m = trimesh.load_mesh(path, process=False)
            if not isinstance(m, trimesh.Trimesh): return None
            vis = m.visual
            img = None
            if hasattr(vis, 'material'):
                mat = vis.material
                if hasattr(mat, 'image') and mat.image is not None:
                    img = mat.image
                elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    img = mat.baseColorTexture
            if img is not None: return img.size
        except Exception: pass
        return None

    def _apply_texture_size(self, path):
        size = self._detect_texture_size(path)
        if size is None: return
        w, h = size
        for i in range(self.combo_tex_size.count()):
            d = self.combo_tex_size.itemData(i)
            if d == (w, h) or d == w == h:
                self.combo_tex_size.setCurrentIndex(i); return
        self.combo_tex_size.insertItem(0, f"{w}x{h} (original)", (w, h))
        self.combo_tex_size.setCurrentIndex(0)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open mesh", self.input_edit.text() or "",
            "3D Meshes (*.ply *.obj *.stl *.off *.glb *.gltf);;All (*.*)")
        if path:
            self.input_edit.setText(path)
            if self.combo_export.currentData() == "texture_uv":
                self._apply_texture_size(path)
            if not self.output_edit.text():
                base = os.path.splitext(path)[0]
                is_tex = self.combo_export.currentData() == "texture_uv"
                suffix = "_" + self.combo_mode.currentData().upper()
                self.output_edit.setText(base + suffix + (self.combo_tex_fmt.currentData() if is_tex else ".ply"))
            self.btn_viewer.setEnabled(_PYQTGRAPH_OK)
            self.current_mesh_path = path

    def _browse_output(self):
        is_texture = self.combo_export.currentData() == "texture_uv"
        if is_texture:
            fmt = self.combo_tex_fmt.currentData()
            filt = "PNG (*.png);;JPEG (*.jpg);;All (*.*)" if fmt == ".png" else "JPEG (*.jpg);;PNG (*.png);;All (*.*)"
            path, _ = QFileDialog.getSaveFileName(self, "Output texture", self.output_edit.text() or "", filt)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Output mesh", self.output_edit.text() or "",
                "PLY (*.ply);;GLB (*.glb);;All (*.*)", "PLY (*.ply)")
        if path:
            self.output_edit.setText(path)

    def _open_3d_viewer(self, path=None):
        """Open the 3D visualization window."""
        if not _PYQTGRAPH_OK:
            self._log("⚠ pyqtgraph required for 3D visualization")
            return

        path = path or self.input_edit.text().strip() or self.current_mesh_path
        if not path:
            self._log("⚠ Please load a mesh file first")
            return

        try:
            # Load the mesh
            mesh = trimesh.load_mesh(path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            if isinstance(mesh, trimesh.PointCloud):
                mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                self._log("⚠ The file is not a simple triangular mesh")
                return
                
            # Create and show the 3D window (single instance — the widget is
            # hidden on close so its GL context, and the compiled shader
            # programs bound to it, survive across reopens).
            if not hasattr(self, 'viewer_3d'):
                self.viewer_3d = MeshViewer3D(self)

            _reset_gl_shader_cache()
            self.viewer_3d.load_mesh(mesh)
            self.viewer_3d.show()
            self.viewer_3d.raise_()
            self.viewer_3d.activateWindow()
            self._log(f"🔍 3D view: {os.path.basename(path)} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
            
        except Exception as e:
            self._log(f"⚠ Error loading mesh: {e}")

    def _run(self):
        input_path  = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path:
            self._log("⚠ Please select an input file."); return
        if not os.path.isfile(input_path):
            self._log(f"⚠ File not found: {input_path}"); return
        if not output_path:
            self._log("⚠ Please set an output file."); return

        self.log_edit.clear()
        export_mode = self.combo_export.currentData()
        self._log(f"Input  : {input_path}")
        self._log(f"Output : {output_path}")
        self._log(f"Radius : {self.spin_radius.value()}  |  Step : {self.spin_step.value()}  "
                  f"|  Samples : {self.spin_samples.value()}  "
                  f"|  Invert : {self.chk_invert.isChecked()}  "
                  f"|  Mode : {self.combo_mode.currentData()}")
        self._log(f"Backend: {_BACKEND}")
        self._log("-" * 60)
        self._set_running(True)

        texture_size = self.combo_tex_size.currentData() if export_mode == "texture_uv" else 2048
        self._worker = Worker(
            input_mesh=input_path, output_mesh=output_path,
            radius=self.spin_radius.value(), voxel_step=self.spin_step.value(),
            n_disk=self.spin_samples.value(), invert=self.chk_invert.isChecked(),
            export_mode=export_mode, texture_size=texture_size, texture_format="",
            vo_mode=self.combo_mode.currentData(),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._update_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
        self._log("⚠ Cancellation requested — aborting…")
        self.btn_cancel.setEnabled(False)

    def _on_finished(self):
        self._log("=" * 60)
        self._log("✔ Computation finished successfully.")
        self._set_running(False)
        self.status.showMessage("Computation finished.")
        # Show the result in the 3D viewer (exported mesh in vertex colors
        # mode, input mesh in texture mode — a texture has no 3D output).
        if _PYQTGRAPH_OK:
            if self.combo_export.currentData() == "texture_uv":
                view_path = self.input_edit.text().strip()
            else:
                view_path = self.output_edit.text().strip()
            self._open_3d_viewer(view_path)

    def _on_error(self, tb):
        self._log("=" * 60)
        self._log("✖ ERROR:\n" + tb)
        self._set_running(False)
        self.status.showMessage("Error — see the log.")

    def _on_cancelled(self):
        self._log("=" * 60)
        self._log("✖ Computation cancelled — nothing was exported.")
        self._set_running(False)
        self.status.showMessage("Cancelled.")

    def _on_vo_mode_changed(self, *args):
        """Keep the output filename suffix (_VO/_VOP/_VON) in sync with the mode."""
        out = self.output_edit.text().strip()
        if not out:
            return
        stem, ext = os.path.splitext(out)
        for s in ("_VO", "_VOP", "_VON"):
            if stem.upper().endswith(s):
                stem = stem[:-len(s)]
                break
        self.output_edit.setText(stem + "_" + self.combo_mode.currentData().upper() + ext)

    def _on_export_mode_changed(self, index):
        is_texture = self.combo_export.currentData() == "texture_uv"
        self.combo_tex_size.setEnabled(is_texture)
        self.combo_tex_fmt.setEnabled(is_texture)
        if is_texture:
            self.output_label.setText("Texture: ")
            self.output_edit.setPlaceholderText("Output texture file (.png, .jpg)")
            inp = self.input_edit.text().strip()
            if inp and os.path.isfile(inp):
                self._apply_texture_size(inp)
        else:
            self.output_label.setText("Output: ")
            self.output_edit.setPlaceholderText("Output file (.ply)")
        output_path = self.output_edit.text().strip()
        if output_path:
            base, ext = os.path.splitext(output_path)
            fmt = self.combo_tex_fmt.currentData() if is_texture else ".ply"
            if ext.lower() != fmt:
                self.output_edit.setText(base + fmt)

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.progress.setVisible(running)
        if not running:
            self.progress.setValue(0)

    def _update_progress(self, value: int):
        self.progress.setValue(value)

    def _log(self, text: str):
        self.log_edit.append(text)
        self.log_edit.moveCursor(QTextCursor.End)
        self.status.showMessage(text[:100])

    def _about(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "About",
            "<b>OBSCURA3D</b> — Cross-platform<br><br>"
            "<b>Modes:</b> VO · VOP · VON<br><br>"
            "<b>Backends (auto-detected):</b><br>"
            "• <b>Warp CUDA</b>: Windows/Linux NVIDIA GPU<br>"
            "• <b>Warp CPU</b>: Mac Apple Silicon / any OS<br>"
            "• <b>Open3D</b>: fallback if Warp is missing<br>"
            "• <b>NumPy</b>: last resort fallback<br>"
            "• <b>metal_hybrid</b>: reserved (Apple Silicon, opt-in)<br><br>"
            "<b>Mac install:</b><br>"
            "<tt>pip install warp-lang open3d</tt>"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _resource_path(relative):
    """Absolute path to a bundled resource (PyInstaller-aware)."""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


if __name__ == "__main__":
    # Headless sanity check for CI/builds: no QApplication, exits immediately
    if "--smoke-test" in sys.argv:
        print(f"OBSCURA3D {__version__} smoke test - backend={_BACKEND} "
              f"wp_device={_WP_DEVICE} open3d={_O3D_OK} numba={_NUMBA_OK} "
              f"pyqtgraph={_PYQTGRAPH_OK} metal={_METAL_OK}")
        if not _NUMBA_OK:
            print(f"  numba err: {_NUMBA_ERR}")
        # trimesh lazy deps used at runtime: scipy (merge_vertices),
        # rtree (ProximityQuery / winding_number) - easy to miss in bundles
        _tm = trimesh.creation.icosphere(subdivisions=1)
        _tm.merge_vertices()
        _sd = trimesh.proximity.signed_distance(
            _tm, [_tm.vertices.mean(axis=0)])[0]
        print(f"  trimesh proximity OK (signed_distance={_sd:.3f})")
        if wp is not None:
            # Actually launch a kernel: verifies inspect.getsource works
            # (JIT source access is broken in badly-packaged frozen builds)
            _n = wp.array(np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                          dtype=wp.vec3, device=_WP_DEVICE)
            _t1 = wp.zeros(1, dtype=wp.vec3, device=_WP_DEVICE)
            _t2 = wp.zeros(1, dtype=wp.vec3, device=_WP_DEVICE)
            _vv = wp.zeros(1, dtype=wp.int32, device=_WP_DEVICE)
            wp.launch(_wp_build_tangent_frames, dim=1,
                      inputs=[_n, _t1, _t2, _vv], device=_WP_DEVICE)
            wp.synchronize()
            if _vv.numpy()[0] != 1:
                raise RuntimeError("warp kernel smoke test failed")
            print("  warp kernel launch OK")
        print("OBSCURA3D smoke test PASSED")
        sys.exit(0)
    app = QApplication(sys.argv)
    app.setApplicationName("OBSCURA3D")
    app.setStyle("Fusion")
    icon_file = "OBSCURA3D.ico" if sys.platform == "win32" else "OBSCURA3D.png"
    icon_path = _resource_path(icon_file)
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
