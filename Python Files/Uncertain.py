# import libraries
import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#=========================================

# display final information of compeleted dataset:
df = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/Lagrange_RFR.csv')
df_rfr_csv = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/RandomForestRegressor_Model.csv')
df_lagrange = pd.read_csv(r'https://raw.githubusercontent.com/BahareAtaei/RandomForestRegressor_Model_Exoplanet_Mass/refs/heads/main/CSV%20Files/Lagrange_RFR.csv')
print('')
print('INFORMATION OF FINAL DATASET(INCLUDE LAGRANGE POINTS)')
print(df.info())
print('='*50)
print('')
#==========================================

#check if there is negative uncertains
error_columns = [col for col in df.columns if 'error_lower' in col or 'error_upper' in col]
print('')
print('NEGATIVE UNCERTAINTIES: ')
for col in error_columns:
    negative = (df[col].dropna() < 0).sum()
    print(f'{col}: negative values = {negative}')
print('='*50)
print('')
#=======================================================

# check uncertainies of mass
print('MASS UNCERTAINIES DESCRIBING')
predicted = df[df['mass_error_lower'].notna() & df['mass_error_upper'].notna()].copy()
predicted['relative_error_lower'] = (predicted['mass_error_lower'] / predicted['mass'])
predicted['relative_error_upper'] = (predicted['mass_error_upper'] / predicted['mass'])
print(predicted[['mass', 'mass_error_lower', 'mass_error_upper', 'relative_error_lower', 'relative_error_upper']].describe())
print('='*50)
print('')

# chack uncertainies of lagrange points
print('LAGRANGE POINTS UNCERTAINIES DESCRIBING')
predicted['L1_relative_error_lower'] = (predicted['x_L1_AU_error_lower'] / predicted['x_L1_AU'].abs())
predicted['L1_relative_error_upper'] = (predicted['x_L1_AU_error_upper'] / predicted['x_L1_AU'].abs())
print(predicted[['x_L1_AU', 'x_L1_AU_error_lower', 'x_L1_AU_error_upper', 'L1_relative_error_lower', 'L1_relative_error_upper']].describe())
print('='*50)
print('')

# # plot relation of mass and poins uncertainies
# plt.figure(figsize=(10, 7))
# plt.scatter(predicted['relative_error_upper'], predicted['L1_relative_error_upper'], color='blue')
# plt.title('Mass Uncertainty vs L1 Uncertainty')
# plt.xlabel('Relative Mass Uncertainty')
# plt.ylabel('Relative L1 Uncertainty')
# plt.grid(alpha=0.5, ls='--')
# plt.tight_layout()
# plt.show()
#=========================================================

# the number of prediction planets
count_mass_predicted = 0
for data in df_rfr_csv['mass_source']:
    if data == 'Predicted':
        count_mass_predicted = count_mass_predicted + 1
print(f"Number of Predicted Masses: {count_mass_predicted}")
print('='*50)
print('')
#==============================================================

# predicted planetary masses distribution
plt.figure(figsize=(8, 5))
plt.hist(predicted["log_mass"], bins=30, label='Predicted Masses')
plt.xlabel("Predicted log10(Planetary Mass)")
plt.ylabel("Number of Planets")
plt.title("Distribution of Predicted Planetary Masses")
plt.grid(alpha=0.5, ls='--')
plt.legend()
plt.tight_layout()
plt.show()
#===========================================

# uncertainy quantities
# lower / upper relative error
predicted_rows = predicted["mass_source"] == "Predicted"
predicted.loc[predicted_rows, "relative_error_lower"] = (predicted.loc[predicted_rows, "mass_error_lower"] / predicted.loc[predicted_rows, "mass"])
predicted.loc[predicted_rows, "relative_error_upper"] = (predicted.loc[predicted_rows, "mass_error_upper"] / predicted.loc[predicted_rows, "mass"])

print("Lower relative uncertainty:")
print(predicted.loc[predicted_rows, "relative_error_lower"].describe())
print("-" * 50)
print('')
print("Upper relative uncertainty:")
print(predicted.loc[predicted_rows, "relative_error_upper"].describe())
print("-" * 50)
print('')
#_____________________________________________

# median of lower / upper 
print(f"Median lower relative uncertainty: {predicted["relative_error_lower"].median()}")
print(f"Median upper relative uncertainty: {predicted["relative_error_upper"].median()}")
print('='*50)
print('')
#==========================================
#==============================================================

# uncertainity of lagrange points 
print("******** Lagrange Point Uncertainty Statistics ********")

df_predicted = df_lagrange[df_lagrange["mass_source"] == "Predicted"]

# definition of the rows
points = {
    "L1": [
        ("x-coordinate", "x_L1_AU_error_lower", "x_L1_AU_error_upper"),
        ("Distance to star", "distance_star_L1_error_lower", "distance_star_L1_error_upper"),
        ("Distance to planet", "distance_planet_L1_error_lower", "distance_planet_L1_error_upper"),
    ],
    "L2": [
        ("x-coordinate", "x_L2_AU_error_lower", "x_L2_AU_error_upper"),
        ("Distance to star", "distance_star_L2_error_lower", "distance_star_L2_error_upper"),
        ("Distance to planet", "distance_planet_L2_error_lower", "distance_planet_L2_error_upper"),
    ],
    "L3": [
        ("x-coordinate", "x_L3_AU_error_lower", "x_L3_AU_error_upper"),
        ("Distance to star", "distance_star_L3_error_lower", "distance_star_L3_error_upper"),
        ("Distance to planet", "distance_planet_L3_error_lower", "distance_planet_L3_error_upper"),
    ],
    "L4": [
        ("x-coordinate", "x_L4_AU_error_lower", "x_L4_AU_error_upper"),
        ("y-coordinate", "y_L4_AU_error_lower", "y_L4_AU_error_upper"),
        ("Distance to star", "distance_star_L4_error_lower", "distance_star_L4_error_upper"),
        ("Distance to planet", "distance_planet_L4_error_lower", "distance_planet_L4_error_upper"),
    ],
    "L5": [
        ("x-coordinate", "x_L5_AU_error_lower", "x_L5_AU_error_upper"),
        ("y-coordinate", "y_L5_AU_error_lower", "y_L5_AU_error_upper"),
        ("Distance to star", "distance_star_L5_error_lower", "distance_star_L5_error_upper"),
        ("Distance to planet", "distance_planet_L5_error_lower", "distance_planet_L5_error_upper"),
    ],
}

# making summary table
rows = []
for point, quantities in points.items():
    for quantity, lower_col, upper_col in quantities:
        lower_median = df_predicted[lower_col].median()
        upper_median = df_predicted[upper_col].median()
        rows.append({
            "Lagrange Point": point,
            "Quantity": quantity,
            "Median Lower Uncertainty (AU)": lower_median,
            "Median Upper Uncertainty (AU)": upper_median
        })

result_lagrange = pd.DataFrame(rows)
print(result_lagrange.to_string(index=False))
#=======================================================

# Relative planetary-mass uncertainty vs relative Lagrange-point uncertainty
print('********** Relative planetary-mass uncertainty vs relative Lagrange-point uncertainty **********')

df_unc = df_lagrange[
    (df_lagrange["mass_source"] == "Predicted") &
    df_lagrange["mass_error_lower"].notna() &
    df_lagrange["mass_error_upper"].notna() &
    (df_lagrange["mass"] > 0)
]

# relative planetary-mass uncertainty
df_unc["mass_relative_lower"] = (df_unc["mass_error_lower"] / df_unc["mass"])
df_unc["mass_relative_upper"] = (df_unc["mass_error_upper"] / df_unc["mass"])

df_unc["mass_relative_uncertainty"] = (df_unc["mass_relative_lower"] + df_unc["mass_relative_upper"]) / 2

# Calculate relative Lagrange-point uncertainty
lagrange_points = ["L1", "L2", "L3", "L4", "L5"]
for point in lagrange_points:

    distance_col = f"distance_star_{point}"
    lower_col = f"distance_star_{point}_error_lower"
    upper_col = f"distance_star_{point}_error_upper"

    # Relative lower uncertainty
    df_unc[f"{point}_relative_lower"] = (df_unc[lower_col].abs() / df_unc[distance_col])

    # Relative upper uncertainty
    df_unc[f"{point}_relative_upper"] = (df_unc[upper_col].abs() / df_unc[distance_col])

    # Mean relative uncertainty
    df_unc[f"{point}_relative_uncertainty"] = (df_unc[f"{point}_relative_lower"] + df_unc[f"{point}_relative_upper"]) / 2

# plot tthe results
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Planetary-Mass Uncertainty vs. Lagrange-Point Uncertainty", fontsize=14)
axes = axes.flatten()
for ax, point in zip(axes, lagrange_points):

    x = df_unc["mass_relative_uncertainty"]
    y = df_unc[f"{point}_relative_uncertainty"]

    # Remove invalid values
    valid = (x.notna() & y.notna() & (x > 0) & (y >= 0))
    df_plot = df_unc[valid].copy()
    ax.scatter(df_plot["mass_relative_uncertainty"],df_plot[f"{point}_relative_uncertainty"], color='blue')
    ax.set_xlabel("Relative Planetary-Mass Uncertainty")
    ax.set_ylabel(f"Relative {point} Distance Uncertainty")
    ax.set_title(point)
    ax.grid(alpha=0.3, linestyle='--')

axes[5].axis("off")
plt.tight_layout()
plt.show()
print('=' * 50)
print('')
#================================================
