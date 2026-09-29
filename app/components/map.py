import folium
from folium.plugins import Draw
import geopandas as gpd
import numpy as np
import rasterio
from typing import Dict, Any, List, Optional, Tuple


# Colorblind safe palette for 5 susceptibility classes
CLASS_COLORS = {
    1: "#2b83ba", # Very Low (Blue)
    2: "#abdda4", # Low (Greenish Teal)
    3: "#ffffbf", # Moderate (Yellow)
    4: "#fdae61", # High (Orange)
    5: "#d7191c"  # Very High (Red)
}


def create_base_map(
    center: Tuple[float, float] = (25.5941, 85.1376),
    zoom: int = 9,
    tile_provider: str = "OpenStreetMap"
) -> folium.Map:
    """Instantiate base interactive Folium map."""
    m = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=tile_provider,
        control_scale=True
    )
    return m


def add_raster_as_image_overlay(
    folium_map: folium.Map,
    raster_data: np.ndarray,
    transform: Any,
    bounds: List[Tuple[float, float]],
    layer_name: str = "Susceptibility Raster",
    opacity: float = 0.7
):
    """
    Render 2D probability or classified raster array on Folium map.
    Converts raster matrix to PNG color overlay bounds.
    """
    try:
        from matplotlib import cm
        import matplotlib.pyplot as plt

        norm_data = np.clip(raster_data, 0.0, 1.0)
        colormap = cm.get_cmap("YlOrRd")
        colored = colormap(norm_data) # RGBA

        # Mask NoData / non-valid pixels
        colored[raster_data < 0, :] = [0, 0, 0, 0]

        folium.raster_layers.ImageOverlay(
            image=colored,
            bounds=bounds,
            name=layer_name,
            opacity=opacity,
            interactive=True,
            cross_origin=False
        ).add_to(folium_map)
    except Exception as e:
        pass


def add_vector_boundary_layer(
    folium_map: folium.Map,
    gdf: gpd.GeoDataFrame,
    layer_name: str = "Administrative Boundaries",
    name_col: str = "NAME_2"
):
    """Add administrative boundary GeoDataFrame layer to Folium map."""
    if gdf.empty:
        return

    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    folium.GeoJson(
        gdf,
        name=layer_name,
        style_function=lambda feature: {
            'fillColor': '#3182ce',
            'color': '#1a365d',
            'weight': 1.5,
            'fillOpacity': 0.1
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[name_col] if name_col in gdf.columns else [gdf.columns[0]],
            aliases=["Name:"],
            localize=True
        )
    ).add_to(folium_map)


def add_field_feedback_markers(
    folium_map: folium.Map,
    gdf_feedback: gpd.GeoDataFrame
):
    """Add field report point markers to Folium map."""
    if gdf_feedback.empty:
        return

    for idx, row in gdf_feedback.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        verified = bool(row.get("expert_verified", False))
        color = "green" if verified else "orange"
        icon_type = "ok-sign" if verified else "info-sign"

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 180px;">
            <b>Report ID #{row.get('id', idx)}</b><br/>
            <b>Date:</b> {row.get('date', 'N/A')}<br/>
            <b>Flood Depth:</b> {row.get('depth_m', 'N/A')} m<br/>
            <b>Description:</b> {row.get('description', '')}<br/>
            <b>Status:</b> {'Expert Verified' if verified else 'Pending Verification'}
        </div>
        """

        folium.Marker(
            location=[geom.y, geom.x],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"Report #{row.get('id', idx)} ({'Verified' if verified else 'Pending'})",
            icon=folium.Icon(color=color, icon=icon_type)
        ).add_to(folium_map)
