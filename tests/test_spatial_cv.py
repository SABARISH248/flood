import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from src.spatial import create_spatial_blocks


def test_spatial_block_assignment():
    pts = [Point(85.0, 25.0), Point(85.1, 25.1), Point(86.0, 26.0)]
    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")

    block_ids = create_spatial_blocks(gdf, block_size_deg=0.25)
    assert len(block_ids) == 3
    # First two points are within 0.25 degree block, third point is in a different block
    assert block_ids[0] == block_ids[1]
    assert block_ids[0] != block_ids[2]
