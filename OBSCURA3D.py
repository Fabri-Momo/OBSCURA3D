"""
vo_sdf_gui4.py — Volumetric Obscurance, cross-platform.

Backend auto-détecté :
  1. Warp CUDA  — Windows/Linux NVIDIA  (meilleur, identique à vo_sdf_gui.py)
  2. Warp CPU   — Mac Apple Silicon / tout OS sans CUDA
  3. Open3D     — fallback si Warp absent (RaycastingScene, multithreaded CPU)
  4. NumPy      — fallback ultime si ni Warp ni Open3D

Installation Mac :  pip install warp-lang open3d
Installation PC  :  pip install warp-lang   (CUDA auto-détecté)
"""

import sys
import os
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

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _PYQTGRAPH_OK = True
except ImportError:
    _PYQTGRAPH_OK = False
    print("⚠ pyqtgraph non disponible → visualisation 3D désactivée")


# ---------------------------------------------------------------------------
# Détection du backend disponible
# ---------------------------------------------------------------------------

_BACKEND   = "numpy"   # sera mis à jour ci-dessous
_WP_DEVICE = "cpu"

# --backend warp_cuda|warp_cpu|open3d|numpy  pour forcer un backend
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

# Forcer le backend si demandé en ligne de commande
if _FORCE_BACKEND in ("warp_cuda", "warp_cpu", "open3d", "numpy"):
    if _FORCE_BACKEND in ("warp_cuda", "warp_cpu") and wp is None:
        print(f"⚠ Warp non disponible, impossible de forcer {_FORCE_BACKEND}")
    elif _FORCE_BACKEND == "open3d" and not _O3D_OK:
        print(f"⚠ Open3D non disponible, impossible de forcer open3d")
    else:
        _BACKEND = _FORCE_BACKEND
        if _FORCE_BACKEND == "warp_cuda":   _WP_DEVICE = "cuda"
        elif _FORCE_BACKEND == "warp_cpu":  _WP_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Kernels Warp (définis seulement si Warp est disponible)
# ---------------------------------------------------------------------------

if wp is not None:
    @wp.kernel
    def _wp_build_tangent_frames(
        normals:   wp.array(dtype=wp.vec3),
        t1_out:    wp.array(dtype=wp.vec3),
        t2_out:    wp.array(dtype=wp.vec3),
        valid_out: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        normal = normals[tid]
        n_len = wp.length(normal)
        if n_len < 1.0e-12:
            valid_out[tid] = wp.int32(0)
            t1_out[tid]    = wp.vec3(0.0, 0.0, 0.0)
            t2_out[tid]    = wp.vec3(0.0, 0.0, 0.0)
            return
        valid_out[tid] = wp.int32(1)
        n   = normal / n_len
        ref = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(wp.dot(n, ref)) > 0.9:
            ref = wp.vec3(0.0, 1.0, 0.0)
        t1 = wp.normalize(wp.cross(n, ref))
        t2 = wp.normalize(wp.cross(n, t1))
        t1_out[tid] = t1
        t2_out[tid] = t2

    @wp.kernel
    def _wp_vo_kernel(
        mesh_id:    wp.uint64,
        vertices:   wp.array(dtype=wp.vec3),
        normals:    wp.array(dtype=wp.vec3),
        t1_arr:     wp.array(dtype=wp.vec3),
        t2_arr:     wp.array(dtype=wp.vec3),
        valid_arr:  wp.array(dtype=wp.int32),
        offsets:    wp.array(dtype=wp.vec3),
        lin_out:    wp.array(dtype=wp.float32),
        lout_out:   wp.array(dtype=wp.float32),
        max_dist:   wp.float32,
        voxel_step: wp.float32,
        invert:     wp.int32,
    ):
        vid, oid = wp.tid()
        if valid_arr[vid] == wp.int32(0):
            return
        p  = vertices[vid]
        n  = wp.normalize(normals[vid])
        t1 = t1_arr[vid]
        t2 = t2_arr[vid]
        off = offsets[oid]
        q   = p + off[0] * t1 + off[1] * t2 + off[2] * n
        query = wp.mesh_query_point_sign_winding_number(mesh_id, q, max_dist)
        is_inside = wp.int32(0)
        if query.result:
            closest     = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            signed_dist = query.sign * wp.length(q - closest)
            if signed_dist < 0.0:
                is_inside = wp.int32(1)
        if invert != wp.int32(0):
            is_inside = wp.int32(1) - is_inside
        if is_inside == wp.int32(1):
            wp.atomic_add(lin_out,  vid, voxel_step)
        else:
            wp.atomic_add(lout_out, vid, voxel_step)

    @wp.kernel
    def _wp_vo_finalize(
        lin_arr:   wp.array(dtype=wp.float32),
        lout_arr:  wp.array(dtype=wp.float32),
        valid_arr: wp.array(dtype=wp.int32),
        vo_out:    wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        if valid_arr[tid] == wp.int32(0):
            vo_out[tid] = float(-1.0)
            return
        lin  = lin_arr[tid]
        lout = lout_arr[tid]
        vo_out[tid] = lout / lin if lin > 0.0 else float(-1.0)


# ---------------------------------------------------------------------------
# Fonctions utilitaires communes
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


try:
    import numba
    from numba import njit, prange

    @njit(parallel=True, cache=False, nogil=True)
    def _rasterize_numba(au, av, bu, bv, cu, cv, v0, v1, v2, denom,
                         c0, c1, r0, r1, result, covered, tex_w, tex_h):
        F = len(au)
        for i in prange(F):
            d = denom[i]
            for row in range(r0[i], r1[i] + 1):
                for col in range(c0[i], c1[i] + 1):
                    px = float(col); py = float(row)
                    w0 = ((bv[i]-cv[i])*(px-cu[i]) + (cu[i]-bu[i])*(py-cv[i])) / d
                    w1 = ((cv[i]-av[i])*(px-cu[i]) + (au[i]-cu[i])*(py-cv[i])) / d
                    w2 = 1.0 - w0 - w1
                    if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                        result[row, col]  = w0*v0[i] + w1*v1[i] + w2*v2[i]
                        covered[row, col] = np.uint8(1)

    _NUMBA_OK = True
    _NUMBA_ERR = ""
except Exception as _e:
    _NUMBA_OK = False
    _NUMBA_ERR = str(_e)


def rasterize_vo_to_texture(uv, faces, vo_values_gray, texture_size=2048, log_fn=None):
    """Rastérisation barycentrique — Numba JIT parallel si disponible, sinon numpy chunks.
    Sans matplotlib/scipy. Fonctionne avec tous les layouts UV.
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
        log_fn(f"Triangles UV valides : {len(tri_idx)}/{len(faces)}")

    # UV → pixels  (V inversé)
    au = (uv0[tri_idx, 0] * (tex_w - 1)).astype(np.float32)
    av = ((1.0 - uv0[tri_idx, 1]) * (tex_h - 1)).astype(np.float32)
    bu = (uv1[tri_idx, 0] * (tex_w - 1)).astype(np.float32)
    bv = ((1.0 - uv1[tri_idx, 1]) * (tex_h - 1)).astype(np.float32)
    cu = (uv2[tri_idx, 0] * (tex_w - 1)).astype(np.float32)
    cv = ((1.0 - uv2[tri_idx, 1]) * (tex_h - 1)).astype(np.float32)

    v0 = vo_values_gray[faces[tri_idx, 0]].astype(np.float32)
    v1 = vo_values_gray[faces[tri_idx, 1]].astype(np.float32)
    v2 = vo_values_gray[faces[tri_idx, 2]].astype(np.float32)

    # Dénominateur barycentrique (F,)
    denom = (bv - cv)*(au - cu) + (cu - bu)*(av - cv)
    ok = np.abs(denom) > 1e-8
    tri_idx = tri_idx[ok]
    au=au[ok]; av=av[ok]; bu=bu[ok]; bv=bv[ok]; cu=cu[ok]; cv=cv[ok]
    v0=v0[ok]; v1=v1[ok]; v2=v2[ok]; denom=denom[ok]

    # Bounding boxes entières de chaque triangle (F,)
    c0 = np.maximum(0,       np.floor(np.minimum(au, np.minimum(bu, cu))).astype(np.int32))
    c1 = np.minimum(tex_w-1, np.ceil( np.maximum(au, np.maximum(bu, cu))).astype(np.int32))
    r0 = np.maximum(0,       np.floor(np.minimum(av, np.minimum(bv, cv))).astype(np.int32))
    r1 = np.minimum(tex_h-1, np.ceil( np.maximum(av, np.maximum(bv, cv))).astype(np.int32))

    result  = np.full((tex_h, tex_w), 128.0, dtype=np.float32)
    covered = np.zeros((tex_h, tex_w), dtype=np.uint8)

    if not _NUMBA_OK and log_fn:
        log_fn(f"⚠ Numba indisponible ({_NUMBA_ERR}) → fallback numpy")

    if _NUMBA_OK:
        import numba as _nb
        nthreads = _nb.get_num_threads()
        pixel_areas = (c1 - c0 + 1) * (r1 - r0 + 1)
        if log_fn: log_fn(f"Méthode : Numba JIT parallel ({nthreads} threads) — "
                          f"{len(denom):,} triangles, bbox moy={pixel_areas.mean():.1f}px², "
                          f"max={pixel_areas.max()}px²")
        # Forcer la contiguïté mémoire — requis par Numba parallel
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
            log_fn(f"Texture générée ({fill_ratio*100:.1f}% couvert)")
        return np.stack([result, result, result], axis=-1)

    # Fallback numpy chunks
    CHUNK = 512
    F = len(denom)
    for start in range(0, F, CHUNK):
        end = min(start + CHUNK, F)
        sl  = slice(start, end)
        C   = end - start   # nb triangles dans ce chunk

        # Largeur/hauteur bbox de chaque triangle
        bw = (c1[sl] - c0[sl] + 1)   # (C,)
        bh = (r1[sl] - r0[sl] + 1)

        # Énumérer tous les pixels de chaque bbox : produit cartésien via repeat/tile
        # pixel_count par triangle = bw * bh
        counts = bw * bh                          # (C,)
        total  = int(counts.sum())
        if total == 0:
            continue

        # Indice triangle pour chaque pixel
        tri_rep = np.repeat(np.arange(C), counts)  # (total,)

        # Coordonnées locales (dc, dr) dans chaque bbox
        dc_all = np.concatenate([np.tile(np.arange(bw[i]), bh[i]) for i in range(C)])
        dr_all = np.concatenate([np.repeat(np.arange(bh[i]), bw[i]) for i in range(C)])

        # Coordonnées pixel absolues
        px = (c0[sl][tri_rep] + dc_all).astype(np.float32)
        py = (r0[sl][tri_rep] + dr_all).astype(np.float32)

        # Barycentriques
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
        log_fn(f"Méthode : numpy chunks (barycentrique)")
        log_fn(f"Texture générée ({fill_ratio*100:.1f}% couvert)")
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
# Calcul VO — backend Warp
# ---------------------------------------------------------------------------

def _compute_vo_warp(mesh, vertices, normals, local_offsets, radius,
                     voxel_step, invert, log_fn, progress_fn):
    N = len(vertices)
    M = len(local_offsets)
    max_dist = radius * 3.0
    faces = np.asarray(mesh.faces).flatten().astype(np.int32)

    wp_mesh = wp.Mesh(
        points=wp.array(vertices,  dtype=wp.vec3,  device=_WP_DEVICE),
        indices=wp.array(faces,    dtype=wp.int32, device=_WP_DEVICE),
    )
    wp_verts   = wp.array(vertices,      dtype=wp.vec3,    device=_WP_DEVICE)
    wp_norms   = wp.array(normals,       dtype=wp.vec3,    device=_WP_DEVICE)
    wp_offsets = wp.array(local_offsets, dtype=wp.vec3,    device=_WP_DEVICE)
    wp_t1      = wp.zeros(N, dtype=wp.vec3,    device=_WP_DEVICE)
    wp_t2      = wp.zeros(N, dtype=wp.vec3,    device=_WP_DEVICE)
    wp_valid   = wp.zeros(N, dtype=wp.int32,   device=_WP_DEVICE)
    wp_lin     = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_lout    = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)
    wp_vo      = wp.zeros(N, dtype=wp.float32, device=_WP_DEVICE)

    progress_fn(35)
    wp.launch(kernel=_wp_build_tangent_frames, dim=N,
              inputs=[wp_norms, wp_t1, wp_t2, wp_valid], device=_WP_DEVICE)

    log_fn(f"Lancement kernel Warp {'CUDA GPU' if _WP_DEVICE == 'cuda' else 'CPU'} "
           f"({N} × {M} = {N*M:,} threads)…")
    wp.launch(kernel=_wp_vo_kernel, dim=(N, M),
              inputs=[
                  wp_mesh.id, wp_verts, wp_norms, wp_t1, wp_t2, wp_valid,
                  wp_offsets, wp_lin, wp_lout,
                  wp.float32(max_dist), wp.float32(voxel_step),
                  wp.int32(1 if invert else 0),
              ], device=_WP_DEVICE)
    wp.launch(kernel=_wp_vo_finalize, dim=N,
              inputs=[wp_lin, wp_lout, wp_valid, wp_vo], device=_WP_DEVICE)
    wp.synchronize()
    log_fn("Kernel Warp terminé.")
    progress_fn(80)

    vo = wp_vo.numpy()
    vo[vo < 0.0] = np.nan
    return vo


# ---------------------------------------------------------------------------
# Calcul VO — backend Open3D (RaycastingScene, multithreaded CPU)
# ---------------------------------------------------------------------------

def _compute_vo_open3d(mesh, vertices, normals, local_offsets, radius,
                       voxel_step, invert, log_fn, progress_fn):
    import open3d as o3d
    N = len(vertices)
    M = len(local_offsets)

    log_fn("Backend : Open3D RaycastingScene (BVH exact, multithreaded CPU)")

    # Construire la scène Open3D
    verts_o3d = o3d.core.Tensor(vertices,             dtype=o3d.core.float32)
    faces_o3d = o3d.core.Tensor(
        np.asarray(mesh.faces, dtype=np.uint32),       dtype=o3d.core.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(verts_o3d, faces_o3d)
    log_fn(f"BVH Open3D construit ({len(mesh.faces):,} triangles).")
    progress_fn(35)

    # Repères tangents numpy
    t1, t2, valid, normals_n = build_tangent_frames_np(normals)

    lin  = np.zeros(N, dtype=np.float32)
    lout = np.zeros(N, dtype=np.float32)

    log_fn(f"Calcul VO Open3D ({N} × {M} = {N*M:,} requêtes)…")
    chunk = 500_000   # requêtes par batch pour éviter OOM

    all_q = []
    vid_list = []
    for oid in range(M):
        ox, oy, oz = local_offsets[oid]
        qx = vertices[:,0] + ox*t1[:,0] + oy*t2[:,0] + oz*normals_n[:,0]
        qy = vertices[:,1] + ox*t1[:,1] + oy*t2[:,1] + oz*normals_n[:,1]
        qz = vertices[:,2] + ox*t1[:,2] + oy*t2[:,2] + oz*normals_n[:,2]
        all_q.append(np.column_stack([qx, qy, qz]))
        vid_list.append(np.arange(N, dtype=np.int32))

    all_q   = np.concatenate(all_q,   axis=0).astype(np.float32)  # (N*M, 3)
    vid_all = np.concatenate(vid_list, axis=0)                     # (N*M,)
    total   = len(all_q)

    processed = 0
    while processed < total:
        end = min(processed + chunk, total)
        q_batch = all_q[processed:end]
        v_batch = vid_all[processed:end]

        q_t   = o3d.core.Tensor(q_batch, dtype=o3d.core.float32)
        # compute_signed_distance utilise winding number → fonctionne sur meshes ouverts
        sd = scene.compute_signed_distance(q_t).numpy()   # (batch,)

        inside = (sd < 0.0).astype(np.int32) * valid[v_batch]
        if invert:
            inside = (1 - inside) * valid[v_batch]
        np.add.at(lin,  v_batch, inside.astype(np.float32)   * voxel_step)
        np.add.at(lout, v_batch, (1-inside).astype(np.float32) * voxel_step * valid[v_batch])

        processed = end
        pct = 35 + int(45 * processed / total)
        progress_fn(pct)

    log_fn("Calcul Open3D terminé.")
    progress_fn(80)

    with np.errstate(invalid='ignore', divide='ignore'):
        vo = np.where(lin > 0, lout / lin, np.nan)
    return vo


# ---------------------------------------------------------------------------
# Calcul VO — fallback numpy pur
# ---------------------------------------------------------------------------

def _compute_vo_numpy(mesh, vertices, normals, local_offsets, radius,
                      voxel_step, invert, log_fn, progress_fn):
    N = len(vertices)
    M = len(local_offsets)
    log_fn("Backend : NumPy CPU (fallback — lent)")
    t1, t2, valid, normals_n = build_tangent_frames_np(normals)
    prox = trimesh.proximity.ProximityQuery(mesh)
    lin  = np.zeros(N, dtype=np.float32)
    lout = np.zeros(N, dtype=np.float32)
    for oid in range(M):
        ox, oy, oz = local_offsets[oid]
        qx = vertices[:,0] + ox*t1[:,0] + oy*t2[:,0] + oz*normals_n[:,0]
        qy = vertices[:,1] + ox*t1[:,1] + oy*t2[:,1] + oz*normals_n[:,1]
        qz = vertices[:,2] + ox*t1[:,2] + oy*t2[:,2] + oz*normals_n[:,2]
        q_pts = np.column_stack([qx, qy, qz])
        _, _, face_idx = prox.on_surface(q_pts)
        face_normals = np.asarray(mesh.face_normals)[face_idx]
        closest, _, _ = prox.on_surface(q_pts)
        diff = q_pts - closest
        sign = np.sign(np.einsum('ij,ij->i', diff, face_normals))
        inside = (sign < 0).astype(np.int32) * valid
        if invert:
            inside = (1 - inside) * valid
        lin  += inside.astype(np.float32)   * voxel_step
        lout += (1-inside).astype(np.float32) * voxel_step * valid
        progress_fn(35 + int(45 * oid / M))
    progress_fn(80)
    with np.errstate(invalid='ignore', divide='ignore'):
        vo = np.where(lin > 0, lout / lin, np.nan)
    return vo


# ---------------------------------------------------------------------------
# Fenêtre de visualisation 3D
# ---------------------------------------------------------------------------

if _PYQTGRAPH_OK:
    class MeshViewer3D(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Visualisation 3D du Mesh")
            self.setGeometry(100, 100, 800, 600)
            
            # Widget central OpenGL
            self.gl_widget = gl.GLViewWidget()
            self.setCentralWidget(self.gl_widget)
            
            # Configuration de la caméra
            self.gl_widget.setCameraPosition(distance=5, elevation=30, azimuth=45)
            
            # Axes pour référence
            self.gl_axis = gl.GLAxisItem()
            self.gl_axis.setSize(1, 1, 1)
            self.gl_widget.addItem(self.gl_axis)
            
            # Grille au sol
            self.gl_grid = gl.GLGridItem()
            self.gl_grid.scale(0.1, 0.1, 0.1)
            self.gl_widget.addItem(self.gl_grid)
            
            self.mesh_item = None
            self.current_mesh = None
            
            # Instructions
            self.statusBar().showMessage("Navigation : clic gauche = rotation | clic droit = pan | molette = zoom")
        
        def load_mesh(self, mesh):
            """Charger un mesh trimesh dans la vue 3D."""
            self.current_mesh = mesh
            
            # Debug info
            print(f"[DEBUG] Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            print(f"[DEBUG] Bounds: min={mesh.vertices.min(axis=0)}, max={mesh.vertices.max(axis=0)}")
            
            # Supprimer l'ancien mesh s'il existe
            if self.mesh_item is not None:
                self.gl_widget.removeItem(self.mesh_item)
            
            # Préparer les vertices et faces pour pyqtgraph
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)
            
            # Créer l'item mesh
            self.mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                color=(0.7, 0.7, 0.7, 1.0),  # Gris clair
                smooth=False,
                computeNormals=True,
                drawEdges=True,
                edgeColor=(0.3, 0.3, 0.3, 1.0)  # Bords sombres
            )
            
            self.gl_widget.addItem(self.mesh_item)
            
            # Centrer la vue sur le mesh
            self.center_view()
            
            # Debug caméra
            bbox = mesh.bounding_box
            print(f"[DEBUG] Center: {bbox.centroid}, Extents: {bbox.extents}")
        
        def center_view(self):
            """Centrer la caméra sur le mesh."""
            if self.current_mesh is None:
                return
            
            # Calculer le centre et la taille du mesh
            center = self.current_mesh.bounding_box.centroid
            extents = self.current_mesh.bounding_box.extents
            
            # Distance de caméra basée sur la taille du mesh
            max_extent = np.max(extents)
            if max_extent < 1e-6:
                max_extent = 1.0  # Éviter distance = 0 pour mesh tiny
            distance = max_extent * 3.0
            
            # Positionner la caméra (center n'est pas supporté dans cette version)
            self.gl_widget.setCameraPosition(
                distance=distance,
                elevation=30,
                azimuth=45
            )
            # Alternative: déplacer le mesh vers l'origine pour la visualisation
            if self.mesh_item is not None:
                self.mesh_item.translate(-center[0], -center[1], -center[2])
            
            # Ajuster aussi la grille et axes à l'échelle
            scale = max_extent * 0.5
            self.gl_axis.setSize(scale, scale, scale)
            self.gl_grid.scale(scale/10, scale/10, scale/10)
        
        def keyPressEvent(self, event):
            """Raccourcis clavier."""
            if event.key() == Qt.Key_R:
                self.center_view()
            elif event.key() == Qt.Key_Escape:
                self.close()


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def clean_mesh(mesh, log_fn=None, progress_fn=None):
    """Nettoyage robuste du mesh : normales, doublons, dégénérés."""
    changes = []
    original_v = len(mesh.vertices)
    original_f = len(mesh.faces)
    
    if progress_fn:
        progress_fn(2)
    
    # 1. Supprimer les vertices non référencés
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) != original_v:
        changes.append(f"vertices {original_v}→{len(mesh.vertices)}")
        if log_fn:
            log_fn(f"  • Vertices non référencés supprimés ({original_v}→{len(mesh.vertices)})")
    
    if progress_fn:
        progress_fn(4)
    
    # 2. Supprimer les faces dégénérées (surface nulle)
    non_degenerate = mesh.area_faces > 1e-12
    if not non_degenerate.all():
        mesh.update_faces(non_degenerate)
        changes.append(f"faces dégénérées supprimées")
        if log_fn:
            log_fn(f"  • Faces dégénérées supprimées ({(~non_degenerate).sum()} faces)")
    
    if progress_fn:
        progress_fn(6)
    
    # 3. Fusionner les vertices dupliqués (tolérance 1e-8)
    # mesh.merge_vertices() gère automatiquement la détection
    mesh.merge_vertices()
    changes.append("vertices dupliqués fusionnés")
    if log_fn:
        log_fn("  • Vertices dupliqués fusionnés")
    
    if progress_fn:
        progress_fn(8)
    
    # 4. Recalculer les normales (plus fiable que les normales importées)
    mesh.fix_normals()
    # Forcer le recalcul en vidant les caches
    mesh.face_normals = None
    mesh.vertex_normals = None
    changes.append("normales recalculées")
    if log_fn:
        log_fn("  • Normales recalculées")
    
    if progress_fn:
        progress_fn(9)
    
    # 5. Détecter et inverser si nécessaire (orientation cohérente)
    # Utiliser le winding number global comme indicateur
    try:
        wn = trimesh.proximity.winding_number(mesh, mesh.vertices.mean(axis=0))
        if wn < 0:
            mesh.invert()
            changes.append("mesh inversé (winding number négatif)")
            if log_fn:
                log_fn("  • Mesh inversé (winding number négatif)")
    except Exception:
        if log_fn:
            log_fn("  • Détection orientation ignorée (échec winding number)")
    
    # 6. Vérifier la cohérence finale
    if not mesh.is_watertight:
        changes.append("⚠ mesh non-watertight")
        if log_fn:
            log_fn("  • ⚠ Mesh non-watertight (OK pour VO)")
    # mesh.fix_inconsistent_faces() n'existe pas dans trimesh, on saute cette étape
    
    if log_fn and changes:
        log_fn(f"Nettoyage terminé : {len(changes)} corrections → {', '.join(changes)}")
    
    return mesh


def compute_vo_sdf(input_mesh, output_mesh, radius, voxel_step, n_disk, invert,
                   export_mode, texture_size, texture_format, log_fn, progress_fn):
    progress_fn(0)
    mesh = trimesh.load_mesh(input_mesh, process=False)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Le fichier n'est pas un maillage triangulaire simple.")
    
    # Nettoyage systématique
    mesh = clean_mesh(mesh, log_fn, progress_fn)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Le fichier n'est pas un maillage triangulaire simple.")

    if export_mode == "texture_uv":
        uv_raw = _get_uv(mesh)
    else:
        uv_raw = None
        mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight:
        log_fn("⚠ Maillage non fermé (OK, winding number fonctionne quand même).")

    progress_fn(10)
    vertices = np.asarray(mesh.vertices,      dtype=np.float32)
    normals  = np.asarray(mesh.vertex_normals, dtype=np.float32)
    N = len(vertices)

    disk_xy, disk_r = fibonacci_disk(n_disk, radius)
    local_offsets   = precompute_local_offsets(disk_xy, disk_r, radius, voxel_step)
    M = len(local_offsets)

    progress_fn(20)
    log_fn(f"Sommets : {N}  |  Offsets/sommet : {M}  |  Total : {N*M:,}")
    log_fn(f"Backend : {_BACKEND}")

    # --- Dispatch selon backend ---
    if _BACKEND in ("warp_cuda", "warp_cpu"):
        vo = _compute_vo_warp(mesh, vertices, normals, local_offsets, radius,
                              voxel_step, invert, log_fn, progress_fn)
    elif _BACKEND == "open3d":
        vo = _compute_vo_open3d(mesh, vertices, normals, local_offsets, radius,
                                voxel_step, invert, log_fn, progress_fn)
    else:
        vo = _compute_vo_numpy(mesh, vertices, normals, local_offsets, radius,
                               voxel_step, invert, log_fn, progress_fn)

    # --- Post-traitement commun ---
    finite = np.isfinite(vo)
    if finite.sum() == 0:
        raise RuntimeError("Aucune valeur VO valide calculée.")

    lo, hi = np.nanpercentile(vo[finite], [2, 98])
    log_fn(f"VO brut  : min={vo[finite].min():.4f}  max={vo[finite].max():.4f}  "
           f"std={vo[finite].std():.4f}  valid={finite.sum()}/{N}")
    log_fn(f"Normalisation : p2={lo:.4f}  p98={hi:.4f}")

    gray = np.zeros(N, dtype=np.uint8)
    if hi - lo < 1e-6:
        log_fn("⚠ Plage VO quasi nulle !")
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
        log_fn(f"✔ Maillage exporté (vertex colors) : {output_mesh}")
    else:
        texture_path = output_mesh
        if uv_raw is None:
            log_fn("⚠ Pas d'UVs — projection cylindrique")
            centered = vertices - vertices.mean(axis=0)
            theta    = np.arctan2(centered[:,2], centered[:,0])
            u_c = theta/(2*np.pi) + 0.5
            v_c = (centered[:,1]-centered[:,1].min()) / (centered[:,1].max()-centered[:,1].min()+1e-8)
            uv = np.column_stack([u_c, v_c])
        else:
            uv = uv_raw
            log_fn(f"UVs : {uv.shape[0]} pts")
        if uv.shape[0] != N:
            raise RuntimeError(f"Incohérence UVs : {uv.shape[0]} vs {N} vertices.")
        uv = uv.copy()
        log_fn(f"UV bruts : U=[{uv[:,0].min():.4f}, {uv[:,0].max():.4f}]  "
               f"V=[{uv[:,1].min():.4f}, {uv[:,1].max():.4f}]")
        # Ne normaliser que si les UVs sont hors [0,1] — sinon conserver l'échelle originale
        for i in range(2):
            mn, mx = uv[:,i].min(), uv[:,i].max()
            r = mx - mn
            if r > 1e-6 and (mn < -0.01 or mx > 1.01):
                uv[:,i] = (uv[:,i] - mn) / r
                log_fn(f"UV axe {i} renormalisé [{mn:.3f},{mx:.3f}] → [0,1]")
        progress_fn(90)
        texture_img = rasterize_vo_to_texture(uv, mesh.faces, gray, texture_size, log_fn)
        pil_img = Image.fromarray(texture_img)
        if texture_path.lower().endswith(".png"):
            pil_img.save(texture_path, "PNG")
        else:
            pil_img.save(texture_path, "JPEG", quality=95)
        _sz = (f"{texture_size[0]}x{texture_size[1]}" if isinstance(texture_size, tuple)
               else f"{texture_size}x{texture_size}")
        log_fn(f"✔ Texture VO exportée ({_sz}) : {os.path.basename(texture_path)}")
        log_fn("ℹ Appliquez cette texture à votre modèle original dans Blender/votre logiciel 3D.")
        progress_fn(100)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class Worker(QObject):
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, input_mesh, output_mesh, radius, voxel_step, n_disk, invert,
                 export_mode, texture_size, texture_format):
        super().__init__()
        self.input_mesh    = input_mesh
        self.output_mesh   = output_mesh
        self.radius        = radius
        self.voxel_step    = voxel_step
        self.n_disk        = n_disk
        self.invert        = invert
        self.export_mode   = export_mode
        self.texture_size  = texture_size
        self.texture_format = texture_format

    def run(self):
        try:
            compute_vo_sdf(
                self.input_mesh, self.output_mesh,
                self.radius, self.voxel_step, self.n_disk, self.invert,
                self.export_mode, self.texture_size, self.texture_format,
                log_fn=self.log.emit, progress_fn=self.progress.emit,
            )
            self.finished.emit()
        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window  (identique à vo_sdf_gui.py)
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VO-SDF v4 — Cross-platform")
        self.setMinimumSize(680, 580)
        self._thread = None
        self._worker = None
        self.current_mesh_path = None
        self._build_ui()
        self._build_menu()

    def _build_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&Fichier")
        open_act = QAction("&Ouvrir maillage…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._browse_input)
        file_menu.addAction(open_act)
        save_act = QAction("Définir &sortie…", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._browse_output)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quitter", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)
        help_menu = mb.addMenu("&Aide")
        about_act = QAction("À &propos…", self)
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
            "warp_cuda": "🟢 Backend : Warp CUDA GPU",
            "warp_cpu":  "🟡 Backend : Warp CPU (pas de GPU CUDA détecté)",
            "open3d":    "🟡 Backend : Open3D CPU (Warp absent)",
            "numpy":     "🔴 Backend : NumPy CPU (lent — installer warp-lang ou open3d)",
        }
        banner = QLabel(backend_labels.get(_BACKEND, _BACKEND))
        banner.setStyleSheet("padding: 4px 8px; border-radius: 4px; "
                             "background: #2a2a2a; color: #ddd;")
        root.addWidget(banner)

        # Files group
        files_box = QGroupBox("Fichiers")
        files_layout = QVBoxLayout(files_box)
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Entrée :"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Sélectionner un maillage (.ply, .obj, .stl…)")
        in_row.addWidget(self.input_edit)
        btn_in = QPushButton("Parcourir…")
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(btn_in)
        files_layout.addLayout(in_row)
        out_row = QHBoxLayout()
        self.output_label = QLabel("Sortie :  ")
        out_row.addWidget(self.output_label)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Fichier de sortie (.ply)")
        out_row.addWidget(self.output_edit)
        btn_out = QPushButton("Parcourir…")
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        files_layout.addLayout(out_row)
        
        # 3D Viewer button
        viewer_row = QHBoxLayout()
        self.btn_viewer = QPushButton("Visualiser 3D")
        self.btn_viewer.setEnabled(False)
        self.btn_viewer.clicked.connect(self._open_3d_viewer)
        if not _PYQTGRAPH_OK:
            self.btn_viewer.setText("Visualiser 3D (pyqtgraph requis)")
            self.btn_viewer.setEnabled(False)
        viewer_row.addWidget(self.btn_viewer)
        viewer_row.addStretch()
        files_layout.addLayout(viewer_row)
        
        root.addWidget(files_box)

        # Settings group
        settings_box = QGroupBox("Paramètres")
        settings_layout = QHBoxLayout(settings_box)
        settings_layout.setSpacing(20)
        v1 = QVBoxLayout()
        v1.addWidget(QLabel("Rayon (m)"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.0001, 100.0)
        self.spin_radius.setDecimals(4)
        self.spin_radius.setSingleStep(0.005)
        self.spin_radius.setValue(0.01)
        self.spin_radius.setToolTip("Rayon local de la sphère d'obscurance.")
        v1.addWidget(self.spin_radius)
        settings_layout.addLayout(v1)
        v2 = QVBoxLayout()
        v2.addWidget(QLabel("Pas (m)"))
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.00001, 10.0)
        self.spin_step.setDecimals(5)
        self.spin_step.setSingleStep(0.0005)
        self.spin_step.setValue(0.001)
        v2.addWidget(self.spin_step)
        settings_layout.addLayout(v2)
        v3 = QVBoxLayout()
        v3.addWidget(QLabel("Échantillons disque"))
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(4, 256)
        self.spin_samples.setSingleStep(4)
        self.spin_samples.setValue(16)
        v3.addWidget(self.spin_samples)
        settings_layout.addLayout(v3)
        v4 = QVBoxLayout()
        v4.addWidget(QLabel("Inverser signe"))
        self.chk_invert = QCheckBox("Invert")
        v4.addWidget(self.chk_invert)
        v4.addStretch()
        settings_layout.addLayout(v4)
        settings_layout.addStretch()
        root.addWidget(settings_box)

        # Export group
        export_box = QGroupBox("Options d'export")
        export_layout = QHBoxLayout(export_box)
        export_layout.setSpacing(20)
        v_export = QVBoxLayout()
        v_export.addWidget(QLabel("Mode d'export"))
        self.combo_export = QComboBox()
        self.combo_export.addItem("Couleurs de vertex", "vertex_colors")
        self.combo_export.addItem("Texture UV", "texture_uv")
        self.combo_export.currentIndexChanged.connect(self._on_export_mode_changed)
        v_export.addWidget(self.combo_export)
        export_layout.addLayout(v_export)
        v_tex_size = QVBoxLayout()
        v_tex_size.addWidget(QLabel("Taille texture"))
        self.combo_tex_size = QComboBox()
        self.combo_tex_size.addItem("1024x1024", 1024)
        self.combo_tex_size.addItem("2048x2048", 2048)
        self.combo_tex_size.addItem("4096x4096", 4096)
        self.combo_tex_size.setCurrentIndex(1)
        self.combo_tex_size.setEnabled(False)
        v_tex_size.addWidget(self.combo_tex_size)
        export_layout.addLayout(v_tex_size)
        v_tex_fmt = QVBoxLayout()
        v_tex_fmt.addWidget(QLabel("Format texture"))
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
        self.btn_run = QPushButton("▶  Lancer le calcul")
        self.btn_run.setFixedHeight(38)
        font = self.btn_run.font(); font.setPointSize(11); font.setBold(True)
        self.btn_run.setFont(font)
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Annuler")
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
        log_box = QGroupBox("Journal")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_edit)
        root.addWidget(log_box, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(f"Prêt — {_BACKEND}")

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
            self, "Ouvrir maillage", self.input_edit.text() or "",
            "Maillages 3D (*.ply *.obj *.stl *.off *.glb *.gltf);;Tous (*.*)")
        if path:
            self.input_edit.setText(path)
            if self.combo_export.currentData() == "texture_uv":
                self._apply_texture_size(path)
            if not self.output_edit.text():
                base = os.path.splitext(path)[0]
                is_tex = self.combo_export.currentData() == "texture_uv"
                self.output_edit.setText(base + "_VO" + (self.combo_tex_fmt.currentData() if is_tex else ".ply"))
            self.btn_viewer.setEnabled(_PYQTGRAPH_OK)
            self.current_mesh_path = path

    def _browse_output(self):
        is_texture = self.combo_export.currentData() == "texture_uv"
        if is_texture:
            fmt = self.combo_tex_fmt.currentData()
            filt = "PNG (*.png);;JPEG (*.jpg);;Tous (*.*)" if fmt == ".png" else "JPEG (*.jpg);;PNG (*.png);;Tous (*.*)"
            path, _ = QFileDialog.getSaveFileName(self, "Texture de sortie", self.output_edit.text() or "", filt)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Maillage de sortie", self.output_edit.text() or "",
                "PLY (*.ply);;GLB (*.glb);;Tous (*.*)", "PLY (*.ply)")
        if path:
            self.output_edit.setText(path)

    def _open_3d_viewer(self):
        """Ouvrir la fenêtre de visualisation 3D."""
        if not _PYQTGRAPH_OK:
            self._log("⚠ pyqtgraph requis pour la visualisation 3D")
            return
            
        if not hasattr(self, 'current_mesh_path') or not self.current_mesh_path:
            self._log("⚠ Veuillez d'abord charger un fichier mesh")
            return
            
        try:
            # Charger le mesh
            mesh = trimesh.load_mesh(self.current_mesh_path, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                self._log("⚠ Le fichier n'est pas un mesh triangulaire simple")
                return
                
            # Créer et afficher la fenêtre 3D
            if not hasattr(self, 'viewer_3d') or not self.viewer_3d.isVisible():
                self.viewer_3d = MeshViewer3D(self)
            
            self.viewer_3d.load_mesh(mesh)
            self.viewer_3d.show()
            self._log(f"🔍 Visualisation 3D : {os.path.basename(self.current_mesh_path)} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
            
        except Exception as e:
            self._log(f"⚠ Erreur lors du chargement du mesh : {e}")

    def _run(self):
        input_path  = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path:
            self._log("⚠ Veuillez sélectionner un fichier d'entrée."); return
        if not os.path.isfile(input_path):
            self._log(f"⚠ Fichier introuvable : {input_path}"); return
        if not output_path:
            self._log("⚠ Veuillez définir un fichier de sortie."); return

        self.log_edit.clear()
        export_mode = self.combo_export.currentData()
        self._log(f"Entrée  : {input_path}")
        self._log(f"Sortie  : {output_path}")
        self._log(f"Rayon   : {self.spin_radius.value()}  |  Pas : {self.spin_step.value()}  "
                  f"|  Échantillons : {self.spin_samples.value()}  "
                  f"|  Invert : {self.chk_invert.isChecked()}")
        self._log(f"Backend : {_BACKEND}")
        self._log("-" * 60)
        self._set_running(True)

        texture_size = self.combo_tex_size.currentData() if export_mode == "texture_uv" else 2048
        self._worker = Worker(
            input_mesh=input_path, output_mesh=output_path,
            radius=self.spin_radius.value(), voxel_step=self.spin_step.value(),
            n_disk=self.spin_samples.value(), invert=self.chk_invert.isChecked(),
            export_mode=export_mode, texture_size=texture_size, texture_format="",
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._update_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _cancel(self):
        self._log("⚠ Annulation demandée — le calcul en cours se terminera normalement.")
        self.btn_cancel.setEnabled(False)

    def _on_finished(self):
        self._log("=" * 60)
        self._log("✔ Calcul terminé avec succès.")
        self._set_running(False)
        self.status.showMessage("Calcul terminé.")

    def _on_error(self, tb):
        self._log("=" * 60)
        self._log("✖ ERREUR :\n" + tb)
        self._set_running(False)
        self.status.showMessage("Erreur — voir le journal.")

    def _on_export_mode_changed(self, index):
        is_texture = self.combo_export.currentData() == "texture_uv"
        self.combo_tex_size.setEnabled(is_texture)
        self.combo_tex_fmt.setEnabled(is_texture)
        if is_texture:
            self.output_label.setText("Texture : ")
            self.output_edit.setPlaceholderText("Fichier texture de sortie (.png, .jpg)")
            inp = self.input_edit.text().strip()
            if inp and os.path.isfile(inp):
                self._apply_texture_size(inp)
        else:
            self.output_label.setText("Sortie :  ")
            self.output_edit.setPlaceholderText("Fichier de sortie (.ply)")
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
            self, "À propos",
            "<b>VO-SDF GUI v4</b> — Cross-platform<br><br>"
            "<b>Backends (auto-détectés) :</b><br>"
            "• <b>Warp CUDA</b> : Windows/Linux NVIDIA GPU<br>"
            "• <b>Warp CPU</b> : Mac Apple Silicon / tout OS<br>"
            "• <b>Open3D</b> : fallback si Warp absent<br>"
            "• <b>NumPy</b> : fallback ultime<br><br>"
            "<b>Installation Mac :</b><br>"
            "<tt>pip install warp-lang open3d</tt>"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
