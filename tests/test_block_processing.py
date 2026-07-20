import dask.array as da
import numpy as np
import xarray as xr

from ticoi.core import chunk_to_block
from ticoi.cube_data_classxr import CubeDataClass


def _chunked_cube():
    shape = (20, 9, 10)
    chunks = (5, 4, 4)
    variables = {
        name: (("mid_date", "y", "x"), da.zeros(shape, chunks=chunks, dtype="float32"))
        for name in ("vx", "vy", "errorx", "errory")
    }
    ds = xr.Dataset(
        variables,
        coords={"mid_date": np.arange(shape[0]), "y": np.arange(shape[1]), "x": np.arange(shape[2])},
    )
    cube = CubeDataClass()
    cube.ds = ds
    cube.nx, cube.ny, cube.nz = shape[2], shape[1], shape[0]
    return cube


def test_chunk_to_block_counts_all_variables_and_full_time_axis():
    cube = _chunked_cube()
    ds = cube.ds.unify_chunks()
    tile_bytes = ds.isel(x=slice(0, 4), y=slice(0, 4)).nbytes
    blocks = chunk_to_block(cube, block_size=tile_bytes * 1.01 / 1024**3)

    coverage = np.zeros((cube.ny, cube.nx), dtype=np.uint8)
    for x_start, x_end, y_start, y_end in blocks:
        coverage[y_start:y_end, x_start:x_end] += 1
        assert x_end - x_start <= 4
        assert y_end - y_start <= 4

    np.testing.assert_array_equal(coverage, 1)


def test_chunk_to_block_keeps_small_cube_whole():
    cube = _chunked_cube()
    assert chunk_to_block(cube, block_size=1) == [[0, cube.nx, 0, cube.ny]]
