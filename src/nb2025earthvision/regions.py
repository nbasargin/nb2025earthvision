import shapely
import fsarcamp as fc
import fsarcamp.cropex14 as cr14
import fsarcamp.hterra22 as ht22


# string constants
CORN_C2_TRAIN = "CORN_C2_TRAIN"
CORN_C2_VAL = "CORN_C2_VAL"


# polygons
class EarthVision2025Regions(fc.GeocodedRegions):
    def __init__(self):
        super().__init__()
        # copy all existing regions
        cropex_regions = cr14.CROPEX14Regions()
        hterra_regions = ht22.HTERRA22Regions()
        for cropex_region_name in cropex_regions.get_geometry_names():
            self.set_geometry_longlat(cropex_region_name, cropex_regions.get_geometry_longlat(cropex_region_name))
        for hterra_region_name in hterra_regions.get_geometry_names():
            self.set_geometry_longlat(hterra_region_name, hterra_regions.get_geometry_longlat(hterra_region_name))
        # define additional regions
        self.set_geometry_longlat(
            CORN_C2_TRAIN,
            shapely.Polygon(
                [
                    (12.87053464083966, 48.69431126433746),
                    (12.87376562043127, 48.69499789574084),
                    (12.87256473020941, 48.69689529921624),
                    (12.86932052746421, 48.69624297290534),
                ]
            ),
        )
        self.set_geometry_longlat(
            CORN_C2_VAL,
            shapely.Polygon(
                [
                    (12.87377509054932, 48.69500016203471),
                    (12.87556291600652, 48.69538260549086),
                    (12.87434155079981, 48.69725743323961),
                    (12.87257095335003, 48.69690133677193),
                ]
            ),
        )
