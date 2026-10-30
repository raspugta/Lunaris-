import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from core.ingestion import pds3
from core.ingestion.loader import IngestError, load_raster

CRS = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"


def write_tif(path, arr, nodata=None, crs=CRS, transform=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
                       crs=crs, transform=transform or from_origin(-1000, 1000, 20, 20), nodata=nodata) as d:
        d.write(arr, 1)


def test_geotiff_nodata_becomes_nan_and_manifest(tmp_path):
    a = np.ones((20, 20), np.float32)
    a[0, 0] = -9999
    p = tmp_path / "dem.tif"
    write_tif(p, a, nodata=-9999)
    lyr = load_raster(p, {"source_name": "test", "instrument": "LOLA", "source_url": "https://example.invalid/x"})
    assert np.isnan(lyr.data[0, 0]) and np.isfinite(lyr.data[1, 1])
    m = lyr.manifest
    assert m["nodata"] == -9999 and len(m["sha256"]) == 64 and m["pixel_size_m"] == 20.0
    assert lyr.grid.polar_stereo_south and lyr.grid.calibrated
    assert "acquisition" in m["missing_fields"]  # missing metadata is reported, not invented


def test_checksum_verification(tmp_path):
    p = tmp_path / "dem.tif"
    write_tif(p, np.ones((10, 10), np.float32))
    with pytest.raises(IngestError, match="Checksum mismatch"):
        load_raster(p, verify_checksum="0" * 64)


def test_rejects_unsupported_and_tiny_and_empty(tmp_path):
    (tmp_path / "x.xyz").write_text("nope")
    with pytest.raises(IngestError, match="Unsupported"):
        load_raster(tmp_path / "x.xyz")
    write_tif(tmp_path / "tiny.tif", np.ones((4, 4), np.float32))
    with pytest.raises(IngestError, match="8x8"):
        load_raster(tmp_path / "tiny.tif")
    write_tif(tmp_path / "empty.tif", np.full((10, 10), -1, np.float32), nodata=-1)
    with pytest.raises(IngestError, match="no valid pixels"):
        load_raster(tmp_path / "empty.tif")
    with pytest.raises(IngestError, match="not found"):
        load_raster(tmp_path / "missing.tif")


def test_plain_image_is_uncalibrated_and_unverified(tmp_path):
    from PIL import Image
    p = tmp_path / "moon.png"
    Image.fromarray((np.random.default_rng(0).random((32, 32)) * 255).astype(np.uint8)).save(p)
    lyr = load_raster(p)
    assert not lyr.grid.calibrated and lyr.grid.res_m is None
    assert lyr.manifest["source_verified"] is False
    assert any("uncalibrated" in w.lower() or "no georeferencing" in w.lower() for w in lyr.warnings)


LBL = """PDS_VERSION_ID = "PDS3"
INSTRUMENT_ID = "LOLA"
DATA_SET_ID = "LRO-L-LOLA-4-GDR-V1.0"
START_TIME = 2009-07-13T17:33:17
STOP_TIME = 2017-02-02T22:18:35
OBJECT = UNCOMPRESSED_FILE
FILE_NAME = "T.IMG"
  OBJECT = IMAGE
    LINES = 10
    LINE_SAMPLES = 10
    SAMPLE_TYPE = LSB_INTEGER
    SAMPLE_BITS = 16
    UNIT = METER
    SCALING_FACTOR = 0.5
    OFFSET = 1737400.
  END_OBJECT = IMAGE
END_OBJECT = UNCOMPRESSED_FILE
OBJECT = IMAGE_MAP_PROJECTION
 MAP_PROJECTION_TYPE = "POLAR STEREOGRAPHIC"
 CENTER_LATITUDE = -90 <deg>
 MAP_SCALE = 20 <m/pix>
 LINE_PROJECTION_OFFSET = 4.5 <pix>
 SAMPLE_PROJECTION_OFFSET = 4.5 <pix>
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""


def test_pds3_img_geometry_and_scaling(tmp_path):
    (tmp_path / "t.lbl").write_text(LBL)
    dn = np.arange(100, dtype="<i2").reshape(10, 10)
    dn.tofile(tmp_path / "t.img")
    lyr = load_raster(tmp_path / "t.img")
    assert lyr.data[0, 1] == pytest.approx(0.5)
    # Grid is symmetric about the pole: centre of the 10x10 grid is (0, 0).
    x, y = lyr.grid.pixel_to_world(4.5, 4.5)
    assert float(x) == pytest.approx(0.0) and float(y) == pytest.approx(0.0)
    assert lyr.grid.transform == (20.0, 0.0, -100.0, 0.0, -20.0, 100.0)


def test_pds3_truncated_file(tmp_path):
    (tmp_path / "t.lbl").write_text(LBL)
    np.zeros(10, "<i2").tofile(tmp_path / "t.img")
    with pytest.raises(IngestError, match="truncated"):
        load_raster(tmp_path / "t.img")


def test_pds3_label_parser_units_and_strings():
    lbl = pds3.parse_label(LBL)
    assert lbl["IMAGE_MAP_PROJECTION"]["MAP_SCALE"] == {"value": 20, "unit": "m/pix"}
    assert lbl["UNCOMPRESSED_FILE"]["IMAGE"]["SCALING_FACTOR"] == 0.5
    assert lbl["UNCOMPRESSED_FILE"]["FILE_NAME"] == "T.IMG"
