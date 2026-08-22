# import libraries
import pandas as pd
import numpy as np
import Functions as fun
import warnings
warnings.filterwarnings("ignore")

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
    'eccentricity',
    'temp_calculated'
]

dff = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\exoplanet.eu_catalog_03-08-26_16_42_07.csv', usecols=use_cols)
df = dff.copy()
#=========================================================

# preprocessing in model A
median_RFR = [
    'star_compactness',
    'star_density',
    'star_mass_orbit_ratio',
    'thermal_equilibrium',
    'star_teff',
    'radius_normalized',
    'star_radius',
    'kepler_ratio',
    'orbital_velocity',
    'semi_major_axis',
    'log_semi_major',
    'radius_period_ratio',
    'orbital_frequency',
    'orbital_period'
]
knn_RFR = [
    'star_mass',
    'star_metallicity',
]
fun.preprocessing(df, radius=False, train_model='RFR', median=median_RFR, knn=knn_RFR)
#===============================================================================
#===============================================================================
#===============================================================================

# import file for training model A
features_RFR = [
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
    'radius_normalized'
]
use_cols1 = [
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

]
use_cols2 = [
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
df = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\preprocessed.csv', usecols=use_cols1)
df = df.copy()
fun.random_forest_reg(df, features=features_RFR, train_model='RFR', median=median_RFR, knn=knn_RFR)
#===========================================================================
#========================================================================
#=========================================================================

# import file for check radius nan values
# and merge filled dataset with original dataset
df = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\random_forest_reg.csv', usecols=use_cols2)
df = df.copy()
df_original = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\exoplanet.eu_catalog_03-08-26_16_42_07.csv', usecols=use_cols)
fun.comp_dataset(df_original, df, train_model='RFR')
#=========================================================
#=================================================================

# fill remain columns by KNN
df = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\comp_dataset.csv')
fun.KNN(df)
#===================================

# check missing values
print('Check Missing Values in Final DataSet')
missing_values = pd.DataFrame(
    {
        'Missing': df.isnull().sum(),
        'Percent': (df.isnull().mean() * 100).round(2)
    }
)
missing_values = missing_values.sort_values('Percent',ascending=True)
print(missing_values)
print('='*50)
print('')
#=============================================
