"""
Data loader, supports both cropex14 and hterra22.
Intended for single-pixel tasks -> slc coordinates.
"""

import numpy as np
import pandas as pd
import shapely
from PIL import Image
import fsarcamp as fc
import fsarcamp.cropex14 as cr14
import fsarcamp.hterra22 as ht22
import nb2025earthvision as ev25

CROPEX_CAMPAIGN_PATH = fc.get_polinsar_folder() / "01_projects/CROPEX/CROPEX14"
CROPEX_MOISTURE_PATH = (
    fc.get_polinsar_folder() / "Ground_truth/Wallerfing_campaign_May_August_2014/Data/ground_measurements/soil_moisture"
)
HTERRA_CAMPAIGN_PATH = fc.get_polinsar_folder() / "01_projects/22HTERRA"
HTERRA_MOISTURE_PATH = fc.get_polinsar_folder() / "Ground_truth/HTerra_soil_2022/DataPackage_final"
CROPEX_PHOTO_PATH = fc.get_polinsar_folder() / "Ground_truth/Wallerfing_campaign_May_August_2014/Data/picture"


def get_campaign(pass_name):
    if pass_name.startswith("14cropex"):
        return cr14.CROPEX14Campaign(CROPEX_CAMPAIGN_PATH)
    if pass_name.startswith("22hterra"):
        return ht22.HTERRA22Campaign(HTERRA_CAMPAIGN_PATH)
    raise ValueError(f"Pass name not supported: {pass_name}")


def get_window_size(pass_name, band, look_mode):
    if look_mode == "preview":
        look_mode = "20looks"
    if band != "L":
        raise ValueError("Only L band defined!")
    if pass_name.startswith("14cropex"):
        return {
            "20looks": (19, 6),  # looks az 7.2, rg 2.8, total 20.1; meters az 3.6, rg 3.6
            "80looks": (38, 12),  # looks az 14.3, rg 5.6, total 80.4; meters az 7.2, rg 7.2
            "320looks": (76, 24),  # looks az 28.6, rg 11.2, total 321.6; meters az 14.3, rg 14.4
        }[look_mode]
    elif pass_name.startswith("22hterra"):
        return {
            "20looks": (9, 7),  # looks az 6.3, rg 3.3, total 20.4; meters az 3.8, rg 4.2
            "80looks": (19, 13),  # looks az 13.3, rg 6.0, total 80.1; meters az 8.0, rg 7.8
            "320looks": (38, 26),  # looks az 26.5, rg 12.1, total 320.6; meters az 15.9, rg 15.6
        }[look_mode]
    else:
        raise ValueError(f"Pass name not supported: {pass_name}")


def get_pauli_rgb_max(band):
    return {"X": (0.6, 0.6, 1.2), "C": (0.6, 0.6, 1.1), "L": (0.5, 0.3, 0.8)}[band]


def get_rgi_params(pass_name, band):
    campaign = get_campaign(pass_name)
    fsar_pass = campaign.get_pass(pass_name, band)
    return fsar_pass.load_rgi_params()


def get_slc(pass_name, band, pol):
    campaign = get_campaign(pass_name)
    fsar_pass = campaign.get_pass(pass_name, band)
    return fsar_pass.load_rgi_slc(pol)


def get_region_slc_extent(pass_name, band, region_names, buffer_px=50):
    campaign = get_campaign(pass_name)
    regions = ev25.EarthVision2025Regions()
    lut = campaign.get_pass(pass_name, band).load_gtc_sr2geo_lut()
    geometry_azrg_list = [regions.get_geometry_azrg(region_name, lut) for region_name in region_names]
    az_min, rg_min, az_max, rg_max = shapely.total_bounds(geometry_azrg_list)
    rg_min = int(rg_min - buffer_px)
    az_min = int(az_min - buffer_px)
    rg_max = int(rg_max + buffer_px)
    az_max = int(az_max + buffer_px)
    return az_min, az_max, rg_min, rg_max


def _crop_slc_region(data, az_min, az_max, rg_min, rg_max):
    """Crop data in slc coordinates to the specified region."""
    return np.copy(data[az_min:az_max, rg_min:rg_max])


def get_region_slc(pass_name, band, pol, az_min, az_max, rg_min, rg_max):
    campaign = get_campaign(pass_name)
    fsar_pass = campaign.get_pass(pass_name, band)
    slc = fsar_pass.load_rgi_slc(pol)
    return _crop_slc_region(slc, az_min, az_max, rg_min, rg_max)


def get_region_t3(pass_name, band, az_min, az_max, rg_min, rg_max, window_size):
    campaign = get_campaign(pass_name)
    fsar_pass = campaign.get_pass(pass_name, band)
    window_az, window_rg = window_size
    slc_hh = _crop_slc_region(fsar_pass.load_rgi_slc("hh"), az_min, az_max, rg_min, rg_max)
    slc_hv = _crop_slc_region(fsar_pass.load_rgi_slc("hv"), az_min, az_max, rg_min, rg_max)
    slc_vh = _crop_slc_region(fsar_pass.load_rgi_slc("vh"), az_min, az_max, rg_min, rg_max)
    slc_vv = _crop_slc_region(fsar_pass.load_rgi_slc("vv"), az_min, az_max, rg_min, rg_max)
    t3_matrix = fc.slc_to_coherency_matrix(slc_hh, slc_hv, slc_vh, slc_vv, window_az, window_rg)
    return t3_matrix.astype(np.complex64)


def mask_strong_t3_pixels(t3, max_t3_power):
    pixel_power = t3.real[..., 0, 0] + t3.real[..., 1, 1] + t3.real[..., 2, 2]
    invalid_pixels = pixel_power > max_t3_power
    t3 = t3.copy()
    # mask out invalid with a scaled identity matrix (ensure that matrix is invertible)
    t3[invalid_pixels] = 0
    t3[invalid_pixels, 0, 0] = 0.01
    t3[invalid_pixels, 1, 1] = 0.01
    t3[invalid_pixels, 2, 2] = 0.01
    return t3


def get_region_incidence(pass_name, band, az_min, az_max, rg_min, rg_max):
    campaign = get_campaign(pass_name)
    fsar_pass = campaign.get_pass(pass_name, band)
    incidence = fsar_pass.load_rgi_incidence()
    return _crop_slc_region(incidence, az_min, az_max, rg_min, rg_max)


def get_region_sm_points(pass_name, band, region_names):
    campaign = get_campaign(pass_name)
    regions = ev25.EarthVision2025Regions()
    geometry_list = [regions.get_geometry_longlat(region_name) for region_name in region_names]
    if pass_name.startswith("14cropex"):
        moisture = cr14.CROPEX14Moisture(CROPEX_MOISTURE_PATH)
        # load specific date
        pass_to_date = {
            "14cropex02": cr14.MAY_15,
            "14cropex03": cr14.MAY_22,
            "14cropex06": cr14.JUN_04,
            "14cropex07": cr14.JUN_12,
            "14cropex09": cr14.JUN_18,
            "14cropex11": cr14.JUL_03,
            "14cropex13": cr14.JUL_24,
        }
        date_name = pass_to_date[pass_name[0:10]]
        day_moisture_points = moisture.load_soil_moisture_points(date_name)
        # filter by field
        filtered_moisture_dfs = [
            moisture.filter_points_by_geometry(day_moisture_points, geometry) for geometry in geometry_list
        ]
        filtered_moisture_df = pd.concat(filtered_moisture_dfs, ignore_index=True)
        # geocode
        return moisture.geocode_points(filtered_moisture_df, campaign, pass_name, band)
    if pass_name.startswith("22hterra"):
        moisture = ht22.HTERRA22Moisture(HTERRA_MOISTURE_PATH)
        pass_to_period = {
            "22hterra01": ht22.APR_28_AM,
            "22hterra02": ht22.APR_28_PM,
            "22hterra03": ht22.APR_29_AM,
            "22hterra04": ht22.APR_29_PM,
            "22hterra05": ht22.JUN_15_AM,
            "22hterra06": ht22.JUN_15_PM,
            "22hterra07": ht22.JUN_16_AM,
            "22hterra08": ht22.JUN_16_PM,
        }
        period_name = pass_to_period[pass_name[0:10]]
        sm_points = moisture.load_soil_moisture_points()
        sm_points = moisture.filter_points_by_period(sm_points, period_name)
        # filter by regions
        sm_points_regions = [moisture.filter_points_by_geometry(sm_points, geometry) for geometry in geometry_list]
        sm_points = pd.concat(sm_points_regions, ignore_index=True)
        sm_points = moisture.geocode_points(sm_points, campaign, band)
        return sm_points
    raise ValueError(f"Pass name not supported: {pass_name}")


def pass_name_to_date_label(pass_name: str):
    return {
        "14cropex0210": "May 15, 2014",
        "14cropex0305": "May 22, 2014",
        "14cropex0620": "June 4, 2014",
        "14cropex0718": "June 12, 2014",
        "14cropex0914": "June 18, 2014",
        "14cropex1114": "July 3, 2014",
        "14cropex1318": "July 24, 2014",
        "22hterra0104": "April 28 AM",
        "22hterra0204": "April 28 PM",
        "22hterra0304": "April 29 AM",
        "22hterra0404": "April 28 PM",
        "22hterra0504": "June 15 AM",
        "22hterra0604": "June 15 PM",
        "22hterra0704": "June 16 AM",
        "22hterra0804": "June 16 PM",
    }[pass_name]


def get_cropex_maize_image_by_pass(pass_name):
    """
    Load a photo from the field. The image is cropped to a square.
    """
    subpath, ymin, xmin, height = {
        # "14cropex0210": ("2014_05_15/Soil_Moisture/C1/IMG_1487.JPG", 1300, 1200, 1000), # CORN_C1
        "14cropex0210": ("2014_05_15/Soil_Moisture/C2/DSCN5134.JPG", 0, 0, 2000),  # CORN_C2
        # "14cropex0305": ("2014_05_22/Soil_Moisture/C1/img_1555.jpg", 350, 0, 1500), # CORN_C1
        "14cropex0305": ("2014_05_22/Soil_Moisture/C2/img_1582.jpg", 800, 0, 1500),  # CORN_C2
        # "14cropex0620": ("2014_06_04/Soil_Moisture/C1/cimg5220.jpg", 430, 1430, 2200), # CORN_C1
        "14cropex0620": ("2014_06_04/Soil_Moisture/C2/cimg5226.jpg", 930, 860, 1800),  # CORN_C2
        # "14cropex0718": ("2014_06_12/Soil_Moisture/C1/cimg5265.jpg", 0, 380, 2730), # CORN_C1
        "14cropex0718": ("2014_06_12/Soil_Moisture/C2/PICT0015.JPG", 920, 80, 1150),  # CORN_C2
        # "14cropex0914": ("2014_06_18/Soil_Moisture/C1/PICT0034.JPG", 530, 760, 1500), # cr14.CORN_C1
        "14cropex0914": ("2014_06_18/Soil_Moisture/C2/PICT0038.JPG", 830, 900, 1250),  # cr14.CORN_C2
        # "14cropex1114": ("2014_07_03/Soil_Moisture/C1/PICT0076.JPG", 780, 940, 1280), # cr14.CORN_C1
        "14cropex1114": ("2014_07_03/Soil_Moisture/C2/PICT0078.JPG", 590, 550, 1400),  # cr14.CORN_C2
        # "14cropex1318": ("2014_07_24/Biomass/C1/C1.1/cimg5440.jpg", 350, 0, 2730), # cr14.CORN_C1
        "14cropex1318": ("2014_07_24/Biomass/C2/C2.2/cimg5453.jpg", 110, 200, 2560),  # cr14.CORN_C2
    }[pass_name]
    width = int(height / 1000 * 1200)
    img = np.asarray(Image.open(CROPEX_PHOTO_PATH / subpath))
    img = img[ymin : ymin + height, xmin : xmin + width].copy()
    return img
