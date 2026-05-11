"""BAM Powder Bed Slicer — core processing functions.

Import this module from the notebook and call ``slice_stl()``.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None
    import warnings
    warnings.warn("trimesh not installed. Run: pip install trimesh", ImportWarning)

logger = logging.getLogger("slicer")


# ── STL loading ───────────────────────────────────────────────────────────────

def _load_stl_as_single_mesh(stl_path: Path):
    if trimesh is None:
        raise ImportError("trimesh is required. Run: pip install trimesh")
    loaded = trimesh.load(stl_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {stl_path}")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported STL content in {stl_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {stl_path}")
    return mesh


def _voxelize_stl(stl_path: Path, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        raise ValueError("Voxel size must be > 0")
    mesh = _load_stl_as_single_mesh(stl_path)
    if not mesh.is_watertight:
        logger.warning(
            "'%s' is not watertight; voxelisation may miss interior volume.",
            stl_path.name,
        )
    voxel_grid = mesh.voxelized(pitch=voxel_size).fill()
    centers = voxel_grid.points
    if centers.shape[0] == 0:
        raise ValueError(f"Voxelisation produced no voxels for {stl_path}")
    return centers


# ── DSMN file writing ─────────────────────────────────────────────────────────

def _write_dsmn_header(fh, stl_name: str, stl_ext: str, p: dict) -> None:
    """Write the fixed-format DSMN header (33 parameter lines + 66 blank lines)."""
    for v in [
        "DSMR-" + stl_name + stl_ext, p["PrinterName"],
        str(p["VoxelDimY"] * 1000), str(p["PrinterY"] * 1000),
        str(p["VoxelDimX"] * 1000), str(round(p["PrinterX"] * 1000, 1)),
        str(p["NozzleOpen"]), str(p["NozzleNum"]),
        str(p["VoxelDimZ"] * 1000), str(p["PrinterZ"] * 1000),
        str(p["PrintSpeedX"]), str(p["PrintSpeedY"]), str(p["PrintSpeedZ"]),
        str(p["LayingSpeedX"]),
        str(p["ManualSpeedX"]), str(p["ManualSpeedY"]), str(p["ManualSpeedZ"]),
        str(p["LinesNum"]), str(p["LayerNumMax"]),
        str(p["DeltaExtraPlaneEnd"]),
        str(p["VoxelRaiseBeforeLaying"]), str(p["VoxelRaiseBeforePrinting"]),
        str(p["DeltaExtraPlaneStart"]),
        str(p["RecoaterOpeningPositionLaying"]), str(p["RecoaterClosingPositionLaying"]),
        str(p["RecoaterOpeningPositionPrinting"]), str(p["RecoaterClosingPositionPrinting"]),
        str(p["RecoaterOpenCloseSpeed"]), str(p["RecoaterHoleOpening"]),
        str(p["LumpBreakerPowder"]),
        str(p["MinLiquid1LevelRange1To7"]), str(p["MaxLiquid1LevelRange1To7"]),
        str(p["MinLiquid2LevelRange1To7"]), str(p["MaxLiquid2LevelRange1To7"]),
    ]:
        fh.write(v + "\n")
    fh.write("\n" * 66)


def _encode_layer_to_hex(layer_2d: np.ndarray) -> list:
    """Encode a 2-D binary layer array as a list of hex strings."""
    hex_lines = []
    for row in layer_2d:
        chunks = [row[i:i + 4] for i in range(0, len(row), 4)]
        hex_lines.append("".join(
            hex(int("".join(map(str, c.astype(int))), 2))[2:].upper()
            for c in chunks
        ))
    return hex_lines


def _write_dsmn(
    path: Path,
    bed: np.ndarray,
    stl_name: str,
    stl_ext: str,
    p: dict,
    debug_mode: bool = False,
    pre: int = 0,
    post: int = 0,
) -> None:
    """Write a complete DSMN file (header + body) from a 3-D bed array."""
    zero_line = "0" * (p["NozzleNum"] // 4)
    with open(path, "w") as fh:
        _write_dsmn_header(fh, stl_name, stl_ext, p)
        if debug_mode and pre > 0:
            fh.write("--- Start of additional zero layers (pre) ---\n")
        for _ in range(pre):
            for __ in range(p["LinesNum"]):
                fh.write(zero_line + "\n")
            if debug_mode:
                fh.write("--- End of additional zero layer (pre) ---\n")
        if debug_mode:
            fh.write("--- Start of actual model layers ---\n")
        for li in range(bed.shape[2]):
            for line in _encode_layer_to_hex(np.transpose(bed[:, :, li])):
                fh.write(line + "\n")
            if debug_mode:
                fh.write("-" * (p["NozzleNum"] // 4) + "\n")
        if debug_mode and post > 0:
            fh.write("--- Start of additional zero layers (post) ---\n")
        for _ in range(post):
            for __ in range(p["LinesNum"]):
                fh.write(zero_line + "\n")
            if debug_mode:
                fh.write("--- End of additional zero layer (post) ---\n")
        fh.write("END\n")


# ── Log file ──────────────────────────────────────────────────────────────────

def _write_log(
    log_path: Path,
    p: dict,
    tot_voxels: int,
    tot_num_layers: int,
    sliced_dims: tuple,
    bbox: list,
    cog: list,
    fit_bool: bool,
) -> None:
    sdx, sdy, sdz = sliced_dims
    dev = 0.05
    warn = lambda a, r: "  <-- deviation > 5 %!" if abs(1 - a / r) > dev else ""
    with open(log_path, "w") as f:
        print("-- Printing and slicing statistics --", file=f)
        print(f"Number of voxels: {tot_voxels}", file=f)
        vol = tot_voxels * p["VoxelDimX"] * p["VoxelDimY"] * p["VoxelDimZ"] * 1000
        print(f"Print volume: {vol:.3f} liters", file=f)
        print(f"Number of layers: {tot_num_layers}", file=f)
        print("Requested voxel dimensions:", file=f)
        print(f"\t x: {p['VoxelDimX'] * 1000:.3f} mm", file=f)
        print(f"\t y: {p['VoxelDimY'] * 1000:.3f} mm", file=f)
        print(f"\t z: {p['VoxelDimZ'] * 1000:.3f} mm", file=f)
        print("Actual voxel dimensions:", file=f)
        print(f"\t x: {sdx * 1000:.3f} mm" + warn(sdx, p["VoxelDimX"]), file=f)
        print(f"\t y: {sdy * 1000:.3f} mm" + warn(sdy, p["VoxelDimY"]), file=f)
        print(f"\t z: {sdz * 1000:.3f} mm" + warn(sdz, p["VoxelDimZ"]), file=f)
        print("Bounding box [m]:", file=f)
        print(f"\t x: {bbox[0]:.3f} to {bbox[1]:.3f}", file=f)
        print(f"\t y: {bbox[2]:.3f} to {bbox[3]:.3f}", file=f)
        print(f"\t z: {bbox[4]:.3f} to {bbox[5]:.3f}", file=f)
        print("Center of gravity [m]:", file=f)
        print(f"\t x: {cog[0]:.3f}", file=f)
        print(f"\t y: {cog[1]:.3f}", file=f)
        print(f"\t z: {cog[2]:.3f}", file=f)
        print(
            "Model fits into printer bounds." if fit_bool
            else "Model does NOT fit into printer bounds!",
            file=f,
        )


# ── Main public function ──────────────────────────────────────────────────────

def slice_stl(stl_path: Path, p: dict, here: Path = None) -> dict:
    """Slice one STL file and write output files to ``here/output/<stl_name>/``.

    Two DSMN files are written:
    - ``<stl_name>.dsmn``      — the actual print recipe with the voxelised model.
    - ``<stl_name>_bbox.dsmn`` — same header, but only the bounding-box outline
                                  (rectangle perimeter per layer) placed at the
                                  exact position the model occupies on the bed.

    Parameters
    ----------
    stl_path : Path
        Path to the ``.stl`` file to process.
    p : dict
        Configuration dictionary (typically loaded from a YAML config file).
    here : Path, optional
        Root directory used for output. Defaults to ``Path.cwd()``.

    Returns
    -------
    dict
        Summary with keys ``stl``, ``voxels``, ``layers``, ``volume_l``,
        ``fits``, ``output``.
    """
    if here is None:
        here = Path.cwd()

    stl_unit     = float(p.get("stl_unit",    1))
    debug_mode   = bool(p.get("debug_mode",   False))
    align_middle = bool(p.get("align_middle", True))

    stl_name = stl_path.stem
    stl_ext  = stl_path.suffix

    out_dir       = here / "output" / stl_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dsmn      = out_dir / f"{stl_name}.dsmn"
    out_dsmn_bbox = out_dir / f"{stl_name}_bbox.dsmn"
    out_xyz       = out_dir / f"{stl_name}.xyz"
    out_image     = out_dir / f"{stl_name}.png"
    out_log       = out_dir / f"{stl_name}.log"

    logger.info(
        "Slicing '%s'  (stl_unit=%s, debug=%s)",
        stl_path.name, stl_unit, debug_mode,
    )

    # ── Voxelise ───────────────────────────────────────────────────────────────
    centers_stl = _voxelize_stl(stl_path, p["VoxelDimY"] * stl_unit)
    np.savetxt(out_xyz, centers_stl, fmt="%.8f", delimiter=" ")

    voxel_array = centers_stl / stl_unit
    unique_x = np.unique(voxel_array[:, 0])
    unique_y = np.unique(voxel_array[:, 1])
    unique_z = np.unique(voxel_array[:, 2])

    sdx = float(np.sort(unique_x)[1] - np.sort(unique_x)[0])
    sdy = float(np.sort(unique_y)[1] - np.sort(unique_y)[0])
    sdz = float(np.sort(unique_z)[1] - np.sort(unique_z)[0])

    sorted_x = np.arange(unique_x.min(), unique_x.max() + 0.01 * p["VoxelDimX"], sdx)
    sorted_y = np.arange(unique_y.min(), unique_y.max() + 0.01 * p["VoxelDimY"], sdy)
    sorted_z = np.arange(unique_z.min(), unique_z.max() + 0.01 * p["VoxelDimZ"], sdz)

    voxel_3d = np.zeros((sorted_x.shape[0], sorted_y.shape[0], sorted_z.shape[0]))
    for row in voxel_array:
        voxel_3d[
            int(np.argmin(np.abs(sorted_x - row[0]))),
            int(np.argmin(np.abs(sorted_y - row[1]))),
            int(np.argmin(np.abs(sorted_z - row[2]))),
        ] = 1

    nx, ny, nz = voxel_3d.shape

    bbox = [
        float(sorted_x.min()), float(sorted_x.max()),
        float(sorted_y.min()), float(sorted_y.max()),
        float(sorted_z.min()), float(sorted_z.max()),
    ]
    cog = [np.mean(bbox[0:2]), np.mean(bbox[2:4]), np.mean(bbox[4:6])]

    # ── Fit check ──────────────────────────────────────────────────────────────
    max_vx = p["PrintLimitXMax"] - p["PrintLimitXMin"]
    max_vy = p["PrintLimitYMax"] - p["PrintLimitYMin"]
    fit_bool       = (max_vx >= nx) and (max_vy >= ny)
    fit_bool_large = (p["NozzleNum"] >= nx) and (p["LinesNum"] >= ny)

    if not fit_bool_large:
        raise ValueError(
            f"Model ({nx} x-voxels, {ny} y-voxels) does not fit "
            f"into printer bed ({p['NozzleNum']} nozzles x {p['LinesNum']} lines). "
            f"Rescale to fit within {p['PrinterX']} m x {p['PrinterY']} m."
        )
    if p["NozzleNum"] % 4 != 0:
        raise ValueError("NozzleNum is not a multiple of 4.")

    # ── Compute bed placement offsets (constant across all layers) ────────────
    if not fit_bool:
        r0 = (p["NozzleNum"] - nx) // 2 if align_middle else 0
        c0 = (p["LinesNum"]  - ny) // 2 if align_middle else 0
    else:
        r0 = p["PrintLimitXMin"] + (max_vx - nx) // 2 if align_middle else p["PrintLimitXMin"]
        c0 = p["PrintLimitYMin"] + (max_vy - ny) // 2 if align_middle else p["PrintLimitYMin"]

    # ── Build model bed ────────────────────────────────────────────────────────
    bed = np.zeros((p["NozzleNum"], p["LinesNum"], nz))
    for ii in range(nz):
        bed[r0:r0 + nx, c0:c0 + ny, ii] = voxel_3d[:, :, ii]
        if debug_mode and fit_bool:
            xmn, xmx = p["PrintLimitXMin"], p["PrintLimitXMax"]
            ymn, ymx = p["PrintLimitYMin"], p["PrintLimitYMax"]
            bed[xmn, ymn:ymx, ii] = 1
            bed[xmx, ymn:ymx, ii] = 1
            bed[xmn:xmx, ymn, ii] = 1
            bed[xmn:xmx, ymx, ii] = 1

    # ── Build bounding-box bed ─────────────────────────────────────────────────
    # Each layer contains only the outline (rectangle perimeter) of the model's
    # footprint, placed at the same position as the actual model on the bed.
    bed_bbox = np.zeros_like(bed)
    for ii in range(nz):
        bed_bbox[r0,          c0:c0 + ny,  ii] = 1  # x-min edge
        bed_bbox[r0 + nx - 1, c0:c0 + ny,  ii] = 1  # x-max edge
        bed_bbox[r0:r0 + nx,  c0,          ii] = 1  # y-min edge
        bed_bbox[r0:r0 + nx,  c0 + ny - 1, ii] = 1  # y-max edge

    # ── Write DSMN (model) ─────────────────────────────────────────────────────
    pre  = int(p.get("AddZerosPre",  0))
    post = int(p.get("AddZerosPost", 0))
    _write_dsmn(out_dsmn, bed, stl_name, stl_ext, p,
                debug_mode=debug_mode, pre=pre, post=post)

    # ── Write DSMN (bounding box) ─────────────────────────────────────────────
    _write_dsmn(out_dsmn_bbox, bed_bbox, stl_name, stl_ext, p,
                debug_mode=False, pre=pre, post=post)

    # ── Visualisation ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.voxels(voxel_3d.astype(bool), facecolors="gray", edgecolor="k",
              linewidth=0.5, shade=False)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_title(stl_name + " - Voxel Representation")
    ax.set_aspect("equal", adjustable="box")
    plt.savefig(out_image, dpi=800)
    plt.show()
    plt.close()

    # ── Log ────────────────────────────────────────────────────────────────────
    tot_voxels = int(np.count_nonzero(voxel_3d))
    _write_log(out_log, p, tot_voxels, nz, (sdx, sdy, sdz), bbox, cog, fit_bool)

    logger.info(
        "Done: '%s'  ->  %s", stl_path.name, str(out_dir.relative_to(here))
    )
    return {
        "stl":      stl_path.name,
        "voxels":   tot_voxels,
        "layers":   nz,
        "volume_l": tot_voxels * p["VoxelDimX"] * p["VoxelDimY"] * p["VoxelDimZ"] * 1000,
        "fits":     fit_bool,
        "output":   str(out_dir.relative_to(here)),
    }
