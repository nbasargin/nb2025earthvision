import numpy as np
import torch
import fsarcamp as fc

code_version = 117

# SAR data config
band = "L"
look_mode = "320looks"
MAX_T3_POWER = 2

# Parameters for soil moisture to dielectrics conversion.
# The HTERRA campaign has two soil texture measurements.
#   CREA farm: sand = 26.8%, clay = 32.4%; Caione farm: sand = 24.5%, clay = 38.2%.
# The CROPEX campaign has no in-situ soil texture measurements.
#   Approximate values from https://soilgrids.org/ on 2024-08-20: sand = 19.2%, clay = 23.9%
sand = torch.tensor(25.0)
clay = torch.tensor(30.0)
frequency = torch.tensor(fc.get_fsar_center_frequency(band))

# Feasible physical parameter ranges
soil_mst_min, soil_mst_max = 0.05, 0.45  # volumetric soil moisture
plant_mst_min, plant_mst_max = 0.10, 0.70  # gravimetric plant moisture
delta_min, delta_max = np.deg2rad(3), np.deg2rad(87)  # X-Bragg delta in radians
phi_min, phi_max = np.deg2rad(-90), np.deg2rad(90)  # dihedral phi in radians

# Physical parameters obtained from calibration
m_s = torch.tensor(0.172)  # surface component scale factor
plant_mst = torch.tensor(0.70)  # gravimetric plant moisture in kg/kg
phi = torch.deg2rad(torch.tensor(-29))  # dihedral phi in radians
