# -*- coding: utf-8 -*-
"""JIT kernels for OBSCURA3D.

Kept in a real ``.py`` file — shipped as a data file in PyInstaller builds —
so ``inspect.getsource`` works.  Warp kernels and Numba ``@njit`` functions
both need source access, which frozen PYZ modules do not provide.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

try:
    import warp as wp
    _WP_KERNELS_OK = True
except Exception:
    wp = None
    _WP_KERNELS_OK = False

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
        lin_pos_out:  wp.array(dtype=wp.float32),
        lout_pos_out: wp.array(dtype=wp.float32),
        lin_neg_out:  wp.array(dtype=wp.float32),
        lout_neg_out: wp.array(dtype=wp.float32),
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
        inside = float(0.0)
        if query.result:
            closest     = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            signed_dist = query.sign * wp.length(q - closest)
            inside = wp.clamp(0.5 - signed_dist / voxel_step, 0.0, 1.0)
        if invert != wp.int32(0):
            inside = 1.0 - inside
        outside = 1.0 - inside
        inside_volume  = inside  * voxel_step
        outside_volume = outside * voxel_step
        wp.atomic_add(lin_out,  vid, inside_volume)
        wp.atomic_add(lout_out, vid, outside_volume)
        if off[2] >= 0.0:
            wp.atomic_add(lin_pos_out,  vid, inside_volume)
            wp.atomic_add(lout_pos_out, vid, outside_volume)
        if off[2] <= 0.0:
            wp.atomic_add(lin_neg_out,  vid, inside_volume)
            wp.atomic_add(lout_neg_out, vid, outside_volume)

    @wp.kernel
    def _wp_vo_finalize(
        lin_arr:      wp.array(dtype=wp.float32),
        lout_arr:     wp.array(dtype=wp.float32),
        lin_pos_arr:  wp.array(dtype=wp.float32),
        lout_pos_arr: wp.array(dtype=wp.float32),
        lin_neg_arr:  wp.array(dtype=wp.float32),
        lout_neg_arr: wp.array(dtype=wp.float32),
        valid_arr:    wp.array(dtype=wp.int32),
        vo_out:       wp.array(dtype=wp.float32),
        mode:         wp.int32,
    ):
        tid = wp.tid()
        if valid_arr[tid] == wp.int32(0):
            vo_out[tid] = float(-1.0)
            return
        if mode == wp.int32(0):
            lin  = lin_arr[tid]
            lout = lout_arr[tid]
            vo_out[tid] = lout / lin if lin > 0.0 else float(-1.0)
        elif mode == wp.int32(1):
            lin   = lin_pos_arr[tid]
            lout  = lout_pos_arr[tid]
            total = lin + lout
            vo_out[tid] = lout / total if total > 0.0 else float(-1.0)
        else:
            lin   = lin_neg_arr[tid]
            lout  = lout_neg_arr[tid]
            total = lin + lout
            vo_out[tid] = lin / total if total > 0.0 else float(-1.0)


# ---------------------------------------------------------------------------
# Numba rasterizer
# ---------------------------------------------------------------------------

try:
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
