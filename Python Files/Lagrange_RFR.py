# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import Functions as fun
import warnings
warnings.filterwarnings("ignore")
#===========================

# import file
use_cols = [
    'name',
    'star_name',
    'detection_type',
    'radius',
    'mass',
    'star_mass',
    'star_radius',
    'semi_major_axis',
    'orbital_period',
    'star_teff',
    'star_metallicity',
    'mass_error_lower',
    'mass_error_upper',
    'mass_lower_1sigma',
    'mass_upper_1sigma',
    'log_radius',
    'kepler_ratio',
    'star_teff',
    'orbital_velocity',
    'thermal_equilibrium',
    'star_metallicity',
    'star_compactness',
    'radius_period_ratio',
    'log_semi_major',
    'star_radius',
    'star_mass_orbit_ratio',
    'orbital_frequency',
    'star_density',
    'radius_normalized',
    'log_mass'
]
dff = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\RandomForestRegressor_Model.csv', usecols=use_cols)
df = dff.copy()
print("INITIAL INFORMATION OF DATASET: ")
print('')
print(df.info())
print('='*50)
print('')
#================================

# calculate lagrange points for dataset's masses
result = []
for index, row in df.iterrows():
    star_mass = row['star_mass']
    planet_mass = row['mass']
    semi_major_axis = row['semi_major_axis']

    
    # predicted mass with uncertainty
    if (row['mass_error_lower'] > 0 and row['mass_error_upper'] > 0):
        lagrange = fun.calculate_lagrange_uncertainty(
            star_mass=star_mass,
            planet_mass=planet_mass,
            mass_error_lower=row['mass_error_lower'],
            mass_error_upper=row['mass_error_upper'],
            semi_major_axis=semi_major_axis,
            n_samples=10,
            random_state=45
        )

    # observed mass
    else:
        lagrange = fun.calculate_lagrange_points(star_mass=star_mass, planet_mass=planet_mass, semi_major_axis=semi_major_axis)

        # observed masses don't have ML uncertainty
        for key in list(lagrange.keys()):
            lagrange[key + '_error_lower'] = np.nan
            lagrange[key + '_error_upper'] = np.nan

    result.append(lagrange)
df_lagrange = pd.DataFrame(result)
#==============================================

# save changes to final file
df_final = pd.concat([df.reset_index(drop=True), df_lagrange.reset_index(drop=True)], axis=1)
df_final = df_final.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\Lagrange_RFR.csv', index=False)
#=============================================
