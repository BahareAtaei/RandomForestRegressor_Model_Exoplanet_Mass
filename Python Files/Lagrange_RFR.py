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
    'log_mass',
    'mass_source'
]
dff = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/RandomForestRegressor_Model.csv', usecols=use_cols)
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
#df_final = df_final.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\Lagrange_RFR.csv', index=False)
#==============================================

# import final file
df_missing_values = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/Lagrange_RFR.csv')
# check missing values
# check missing values
print('Check Missing Values')
missing_values = pd.DataFrame(
    {
        'Missing': df_missing_values.isnull().sum(),
        'Percent': (df_missing_values.isnull().mean() * 100).round(2)
    }
)
missing_values = missing_values.sort_values('Percent', ascending=False)
print(missing_values.to_string())
print('='*50)
print('')
#===========================================

# statistics of the calculated points
print("********** Statistics of Calculated Points **********")
print('')
coordinate_columns = ["x_L1_AU", "x_L2_AU", "x_L3_AU", "x_L4_AU", "y_L4_AU", "x_L5_AU", "y_L5_AU"]
summary = pd.DataFrame({
    "Minimum (AU)": df_missing_values[coordinate_columns].min(),
    "Median (AU)": df_missing_values[coordinate_columns].median(),
    "Maximum (AU)": df_missing_values[coordinate_columns].max(),
    "Number": df_missing_values[coordinate_columns].count()
})
print(summary)
print('='*50)
print('')
#========================================

# distribution of lagrange points distances in boxplot
# it's more clear than scatter plot
print('********** Distribution of Lagrange-Point Distances Plot **********')
distance_columns = ["distance_star_L1", "distance_star_L2", "distance_star_L3", "distance_star_L4", "distance_star_L5"]
data = [df_missing_values[col].dropna() for col in distance_columns]

plt.figure(figsize=(9, 6))
plt.boxplot(data, labels=["L1", "L2", "L3", "L4", "L5"])
plt.xlabel("Lagrange Point")
plt.ylabel("Distance from Star(AU)")
plt.title("Distribution of Lagrange-Point Distances from the Star")
plt.grid(alpha=0.5, ls='--')
plt.tight_layout()
plt.show()
print('='*50)
print('')
#===============================================


# Select one representative planetary system

required_columns = ["x_L1_AU", "x_L2_AU", "x_L3_AU", "x_L4_AU", "y_L4_AU", "x_L5_AU", "y_L5_AU"]

# Remove rows with missing Lagrange coordinates
df_plot = df_missing_values.dropna(subset=required_columns).copy()

# select a representative system
# we selectbthe first valid system is used.
row = df_plot.iloc[0]

star_mass = row["star_mass"]
planet_mass = row["mass"]
semi_major_axis = row["semi_major_axis"]

jupiter_to_sun_mass = 0.000954588
planet_mass_solar = planet_mass * jupiter_to_sun_mass
total_mass = star_mass + planet_mass_solar
b = planet_mass_solar / total_mass
star_x = -b * semi_major_axis
planet_x = (1 - b) * semi_major_axis

# Lagrange-point coordinates
x_L1 = row["x_L1_AU"]
x_L2 = row["x_L2_AU"]
x_L3 = row["x_L3_AU"]
x_L4 = row["x_L4_AU"]
y_L4 = row["y_L4_AU"]
x_L5 = row["x_L5_AU"]
y_L5 = row["y_L5_AU"]

# L1, L2 and L3 are on the x-axis
y_L1 = 0
y_L2 = 0
y_L3 = 0

# plot the results
plt.figure(figsize=(10, 8))
plt.scatter(star_x, 0, s=250, marker="*", label="Star")
plt.scatter(planet_x, 0, s=100, marker="o", label="Planet")
plt.scatter(x_L1, y_L1, s=70, label="L1")
plt.scatter(x_L2, y_L2, s=70, label="L2")
plt.scatter(x_L3, y_L3, s=70, label="L3")
plt.scatter(x_L4, y_L4, s=70, label="L4")
plt.scatter(x_L5, y_L5, s=70, label="L5")

plt.text(star_x, 0, "  Star")
plt.text(planet_x, 0, "  Planet")
plt.text(x_L1, y_L1, "  L1")
plt.text(x_L2, y_L2, "  L2")
plt.text(x_L3, y_L3, "  L3")
plt.text(x_L4, y_L4, "  L4")
plt.text(x_L5, y_L5, "  L5")

plt.axhline(0, linewidth=0.8)
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.title(f"Lagrange Points of {row['name']}")

plt.legend()
plt.grid(alpha=0.3, ls='--')
plt.tight_layout()
plt.show()
print('='*50)
print('')
#================================================
