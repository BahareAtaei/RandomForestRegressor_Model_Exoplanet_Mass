"""
step3: COMPELETE DATASET FOR CALCULATION

(1) check that why we removed NaN radiuses in the first place
(2) merging prepricessed.csv and filled_modelA.csv and save final dataset to modelA.csv
(3) dataset is ready for calculation!!

"""
#===============================

# import libraries
import pandas as pd

# import files
df_original = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\Model_A\preprocessed.csv')
df_modelA = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\Model_A\filled_modelA.csv')
#=============================================

# planets with their detection type
print("Detection Type of the Most of NaN Radius: ")
print('')
nan_radius = df_original[df_original['radius'].isnull() & df_original['detection_type'].notna()]
detect_type_count = nan_radius['detection_type'].value_counts().sort_values(ascending=False)
print(detect_type_count.head(20))
print("""
IN THESE DETECTION TYPES WE HAVE MASS NOT RADIUS 
RADIUS AND MASS HAVE THE MOST CORRALATION WITH EACH OTHER
SO WE CANT FILL RADIUS BY ANOTHER MODEL
WE HAVE TO REMOVE THESE RADIUSES FROM OUR DATASET 
""")
print('='*50)
print('')
#============================================

# check if all these 10 detection types have mass or not
removed_planets = df_original[df_original['detection_type'].isin(['Radial Velocity', 'Microlensing', 'Radial Velocity, Astrometry', 'Timing', 'Imaging'])]
mask_rp = removed_planets['mass'].notna() & removed_planets['radius'].isnull()
removed_planets = removed_planets[mask_rp]
removed_planets['mass_source'] = 'No Radius Observed'

print('Number of NaN Radius / Not NaN values Mass in Radial Velocity Detection Type: ')
print(len(removed_planets[mask_rp]['mass']))
print('='*50)
print('')
#===============================================

# concat this dataset with the dataset which filled by model A
df_final = pd.concat([df_modelA, removed_planets], ignore_index = True)
print('Information The Final State of DataSet: ')
print('-'*50)
print(df_final.info())
#==================================================

# save finale changes to a final dataset
df_final.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\Model_A\modelA.csv')
