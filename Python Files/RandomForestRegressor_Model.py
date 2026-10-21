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
    'temp_calculated',
    'mass_error_min',
    'mass_error_max'
    
]

dff = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/exoplanet_catalog.csv', usecols=use_cols)
df = dff.copy()
#=========================================================
# rename two columns
df = df.rename(
    columns = {
        'mass_error_min': 'mass_error_lower',
        'mass_error_max': 'mass_error_upper'
    }
)
#=======================================================
# preprocessing in model A
median_RFR = [
    'star_compactness',
    'star_density',
    'star_mass_orbit_ratio',
    'thermal_equilibrium',
    'radius_normalized',
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
    'star_radius',
    'star_teff',
    'star_metallicity'
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
    'mass_error_lower',
    'mass_error_upper'


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
    'log_mass',
    'mass_error_lower',
    'mass_error_upper'

]
df = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/preprocessed.csv', usecols=use_cols1)
df = df.copy()
fun.random_forest_reg(df, features=features_RFR, train_model='RFR', median=median_RFR, knn=knn_RFR)
#===========================================================================
#========================================================================
#=========================================================================

# import file for check radius nan values
# and merge filled dataset with original dataset
use_cols3 = [
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
    'log_mass',
    'mass_error_lower',
    'mass_error_upper',
    'mass_source'

]
df = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/random_forest_reg.csv', usecols=use_cols3)
df = df.copy()
df_original = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/exoplanet_catalog.csv', usecols=use_cols)
fun.comp_dataset(df_original, df, train_model='RFR')
#=================================================================

# fill remain columns by KNN
df = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/comp_dataset.csv', usecols=use_cols3)
fun.remain(df)
#===================================
#=====================================

# check missing values
df_final = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/RandomForestRegressor_Model.csv', usecols=use_cols3)
print('Check Missing Values in Final DataSet')
missing_values = pd.DataFrame(
    {
        'Missing': df_final.isnull().sum(),
        'Percent': (df_final.isnull().mean() * 100).round(2)
    }
)
missing_values = missing_values.sort_values('Percent', ascending=True)
print(missing_values)
print('='*50)
print('')
# print(df_final['mass_source'].value_counts(dropna=False))

#=============================================
#==============================================
