# import libraries
import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#=========================================

# display final information of compeleted dataset:
df = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\Lagrange_RFR.csv')
print('')
print('INFORMATION OF FINAL DATASET(INCLUDE LAGRANGE POINTS)')
print(df.info())
print('='*50)
print('')
#==========================================

# df_check = pd.read_csv(
#     r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\A\Lagrange_A.csv'
# )

# problem = df_check[
#     df_check['mass_error_lower'].notna() &
#     df_check['x_L1_AU_error_lower'].isna()
# ]

# print(
#     problem[
#         [
#             'name',
#             'mass',
#             'mass_error_lower',
#             'mass_error_upper',
#             'mass_lower_1sigma',
#             'mass_upper_1sigma',
#             'star_mass',
#             'semi_major_axis'
#         ]
#     ].to_string(index=False)
# )
#=======================================================

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
#======================================================

# chack uncertainies of lagrange points
print('LAGRANGE POINTS UNCERTAINIES DESCRIBING')
predicted['L1_relative_error_lower'] = (predicted['x_L1_AU_error_lower'] / predicted['x_L1_AU'].abs())
predicted['L1_relative_error_upper'] = (predicted['x_L1_AU_error_upper'] / predicted['x_L1_AU'].abs())
print(predicted[['x_L1_AU', 'x_L1_AU_error_lower', 'x_L1_AU_error_upper', 'L1_relative_error_lower', 'L1_relative_error_upper']].describe())
print('='*50)
print('')
#======================================================

# plot relation of mass and poins uncertainies
plt.figure(figsize=(10, 7))
plt.scatter(predicted['relative_error_upper'], predicted['L1_relative_error_upper'], color='blue')
plt.title('Mass Uncertainty vs L1 Uncertainty')
plt.xlabel('Relative Mass Uncertainty')
plt.ylabel('Relative L1 Uncertainty')
plt.grid(alpha=0.5, ls='--')
plt.tight_layout()
plt.show()
