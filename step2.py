"""
step2: RANDOM FOREST REGRESSOR MODEL (without eccentricity, temp_calculated)
----------------------------------------------------------

THE USED FEATURES:

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
"""
#=============================================

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


dff = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\Model_A\preprocessed.csv')
df = dff.copy()
#==============================================

# initial information of radius column
print(f'Number of NaN Values of Radius: {df['radius'].isnull().sum()}')
print(f'\n Number of Not NaN Values of Radius: {df['radius'].notnull().sum()}')
print(f'\n Dimention of DataSet: {df.shape}')
print('='*50)
print('')
#===============================================

# turn mass scale to log10
df['log_mass'] = np.log10(df['mass'])
#=================================================

# features that we are going to use them in model
features = [
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
#=================================================

# split nan mass and not nan mass values to two dataset
known = df[df['mass'].notna()].copy()
unknown = df[df['mass'].isnull()].copy()
#======================================================

# train and test variables
X = known[features]
y = known['log_mass']

RANDOM_STATE = 45
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
#=====================================================

# making random forest model
model = RandomForestRegressor(
    n_estimators = 400,
    max_depth = None,
    min_samples_leaf = 2,
    random_state =RANDOM_STATE,
    n_jobs = -1
)

# calculating cv
cv = KFold(
    n_splits = 5,
    shuffle = True,
    random_state = RANDOM_STATE
)

# calculating cv_scores
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')

# print results
print(f'Cross Validation Score r2 (log_mass) = {cv_scores}')
print(f'\nMean r2 = {cv_scores.mean()}')
print(f'\n Standard Variation = {cv_scores.std()}')
print('='*50)
print('')
#========================================================

# fit model
model.fit(X_train, y_train)
#========================================================

y_predict_test = model.predict(X_test)
# log space
r2_log = r2_score(y_test, y_predict_test)
mae_log = mean_absolute_error(y_test, y_predict_test)

# real scale
y_test_real = 10**y_test
y_predict_test_real = 10**y_predict_test
mae_real = mean_absolute_error(y_test_real, y_predict_test_real)
r2_real = r2_score(y_test_real, y_predict_test_real)

# print results
print(f'r2 Score in Log Space: {r2_log}')
print(f'\nMean Absolute Error(log): {mae_log}')
print(f'\nr2 Score in Real Space: {r2_real}')
print(f'\nMean Absolute Error(Real): {mae_real}')
print('='*50)
print('')

# calculate relative error
relative_error = np.abs(y_predict_test - y_test) / y_test
relative_error_real = np.abs(y_predict_test_real - y_test_real) / y_test_real
print(f'Median Relative Error: {np.median(relative_error)}')
print(f'Mean Relative Error: {np.mean(relative_error)}')
print('='*50)
print('')
#==========================================================
#===========================================================

# Error Analysis in log space
error_df = pd.DataFrame(
    {
        'Actual_Mass': y_test,
        'Predicted_Mass': y_predict_test,
        'Relative_Error': relative_error
    }
)

# mass groups
mass_bins = [0, 0.03, 0.3, 13, np.inf]
mass_labels = [
    'super_earth',
    'neptune',
    'jupiter',
    'massive_planet'
]

error_df['Mass_Group'] = pd.cut(
    error_df['Actual_Mass'],
    bins=mass_bins,
    labels=mass_labels
)

# calculate error for each group
group_error = error_df.groupby('Mass_Group', observed=True)['Relative_Error'].agg(['count', 'mean', 'median'])
print("Error Analysis by Mass Group")
print(group_error)
print('='*50)
print('')

# plot error analysis
plt.figure(figsize=(12, 8))
group_error['mean'].plot(kind='bar')
plt.title('Prediction Erro by Planet Mass(log)')
plt.xlabel('Planet Mass Group')
plt.ylabel('Mean Relative Error')
plt.grid(alpha=0.5, ls='--')
plt.tight_layout()
plt.show()
#=========================================================

# Error Analysis in real space
error_df = pd.DataFrame(
    {
        'Actual_Mass': y_test_real,
        'Predicted_Mass': y_predict_test_real,
        'Relative_Error': relative_error_real
    }
)

# mass groups
mass_bins = [0, 0.03, 0.3, 13, np.inf]
mass_labels = [
    'super_earth',
    'neptune',
    'jupiter',
    'massive_planet'
]

error_df['Mass_Group'] = pd.cut(
    error_df['Actual_Mass'],
    bins=mass_bins,
    labels=mass_labels
)

# calculate error for each group
group_error = error_df.groupby('Mass_Group', observed=True)['Relative_Error'].agg(['count', 'mean', 'median'])
print("Error Analysis by Mass Group")
print(group_error)
print('='*50)
print('')

# plot error analysis
plt.figure(figsize=(12, 8))
group_error['mean'].plot(kind='bar')
plt.title('Prediction Erro by Planet Mass')
plt.xlabel('Planet Mass Group')
plt.ylabel('Mean Relative Error')
plt.grid(alpha=0.5, ls='--')
plt.tight_layout()
plt.show()
#=================================================================

# plot predictted / actual masses
plt.figure(figsize=(12, 8))
plt.scatter(
    y_test,
    y_predict_test,
    color='blue',
    label='Masses'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    ls='--',
    color='red',
    label='Best Prediction'
)
plt.title("Predicted Vs Actual log(Mass)")
plt.xlabel("Actual Mass(log)")
plt.ylabel("Predicted Mass(log)")
plt.legend()
plt.grid(alpha=0.5, ls='--')
plt.tight_layout()
plt.show()
#==========================================================

# residual plot
residual = y_test - y_predict_test

plt.figure(figsize=(12, 8))
plt.scatter(
    y_predict_test,
    residual,
    label ='Residuals of Prediction',
    color='blue'
)

plt.axhline(
    y=0,
    ls='--'
)
plt.title("Residual Plot")
plt.xlabel("Predicted Mass(log)")
plt.ylabel("Resiiduals")
plt.legend()
plt.grid(alpha=0.5, ls='--')
plt.tight_layout()
plt.show()
#==============================================================
#==============================================================

# now train model to predict unknown masses
X_unknown = unknown[features]
predict_log_mass = model.predict(X_unknown)
predict_mass = 10**predict_log_mass

unknown['log_mass'] = predict_log_mass
unknown['mass'] = predict_mass
unknown['mass_source'] = 'Predicted'
known['mass_source'] = 'Observed'
#===============================================

# merge datasets
filled_mass_dataset = pd.concat([known, unknown], ignore_index=True)
print('')
print(f"Number of NaN Mass Values After using Model: {filled_mass_dataset['mass'].isnull().sum()}")
print('-'*50)

print('')
print('Final Information After Using Model: ')
print('-'*50)
print(filled_mass_dataset.info())
#=============================================

# save dataset to a new csv file
filled_mass_dataset.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\Model_A\filled_modelA.csv', index=False)
