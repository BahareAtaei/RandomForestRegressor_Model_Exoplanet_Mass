# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, r2_score)
import warnings
warnings.filterwarnings("ignore")
#==========================================

def preprocessing(df, radius=False, train_model='RFR', median=None, knn=None):
    '''
    Parameters: 
    ------------------
    df:          dataframe exoplanet.eu_catalog_03-08-26_16_42_07.csv
    radius:      bulean
    train_model: string
    median:      list strings
    knn:         list strings

    Returns:
    ------------------
    Dataframe: preprocessed.csv
    '''
    print(f'****************** PREPROCESSING FUNCTION MODEL {train_model} ********************')
    print('='*50)
    print('='*50)
    print('='*50)
    print('')
    #=====================================================
    if median == None:
        median = []

    if knn == None:
        knn = []

    # first drop duplicated rows
    print('Numbers of Duplicated Rows: ')
    print(df.duplicated().sum())
    print('')
    print('Dimention of DataSet after Removing Duplication Rows: ')
    df = df.drop_duplicates().copy()
    print(df.shape)
    print('='*50)
    print('')

    #=====================================================

    # check missing values
    print('Check Missing Values')
    missing_values = pd.DataFrame(
        {
            'Missing': df.isnull().sum(),
            'Percent': (df.isnull().mean() * 100).round(2)
        }
    )
    missing_values = missing_values.sort_values('Percent', ascending=False)
    print(missing_values)
    print('='*50)
    print('')
    #===================================================

    # fill semi_mjor_Axis and orbital_period
    print('Before Filling Semi_Major_Axis and Orbital_Period: ')
    print(
        df[
            ['semi_major_axis', 'orbital_period']
        ].isnull().sum()
    )
    print('='*50)
    print('')

    # orbital_period(days) -> orbital_period(years)
    # 365.25 days = 1 year
    orbital_period = df['orbital_period'] / 365.25

    # where "semi_major_axis" is NaN but
    # "orbital_period" and "star_mass" aren't NaN
    mask_sma = (df['semi_major_axis'].isna() & df['orbital_period'].notna() & df['star_mass'].notna())
    df.loc[mask_sma, 'semi_major_axis'] = (orbital_period[mask_sma]**2 * df.loc[mask_sma, 'star_mass'])**(1 / 3)

    # where "orbital_period" is NaN but
    # "semi_major_axis" and "star_mass" aren't NaN
    mask_op = (df['orbital_period'].isna() & df['semi_major_axis'].notna() & df['star_mass'].notna())
    df.loc[mask_op, 'orbital_period'] = (np.sqrt(df.loc[mask_op, 'semi_major_axis']**3 / df.loc[mask_op, 'star_mass']) * 365.25)

    # print results
    print('NaN Values of "Semi Major Axis" and "Orbital_period":  ')
    print(df[['semi_major_axis', 'orbital_period']].isnull().sum())
    print('='*50)
    print('')
    #=========================================================

    # check missing values after filling
    # semi_major_axis and orbital_period
    print('Check Missing Values: After Filling Semi_Major_Axis and Orbital_Period')
    missing_values = pd.DataFrame(
        {
            'Missing': df.isnull().sum(),
            'Percent': (df.isnull().mean() * 100).round(2)
        }
    )
    missing_values = missing_values.sort_values('Percent',ascending=False)
    print(missing_values)
    print('='*50)
    print('')
    #=========================================================

    # with radius / without radius
    if radius == False:
        df = df.dropna(subset=['radius']).copy()
    #=========================================================

    # check radius before log
    if (df['radius'] <= 0).any():
        raise ValueError('There is zero or negative Radius values')
    #=========================================================

    # feature engineering
    df['kepler_ratio'] = (df['orbital_period'] ** 2) / (df['semi_major_axis'] ** 3)
    df['orbital_velocity'] = (2 * np.pi * df['semi_major_axis']) / df['orbital_period']
    df['thermal_equilibrium'] = (df['star_teff'] ** 4) / (df['semi_major_axis'] ** 2)
    df['star_compactness'] = (df['star_mass'] / df['star_radius'])
    df['radius_period_ratio'] = (df['radius'] / df['orbital_period'])
    df['log_semi_major'] = np.log10(df['semi_major_axis'])
    df['star_mass_orbit_ratio'] = (df['star_mass'] / df['semi_major_axis'])
    df['orbital_frequency'] = (1 / df['orbital_period'])
    df['star_density'] = (df['star_mass'] / (df['star_radius'] ** 3))
    df['radius_normalized'] = (df['radius'] / df['star_radius'])
    #===================================================

    # again checking missing values
    # after feature engineering
    print('Check Missing Values: After Feature Engineering')
    missing_values = pd.DataFrame(
        {
            'Missing': df.isnull().sum(),
            'Percent': (df.isnull().mean() * 100).round(2)
        }
    )
    missing_values = missing_values.sort_values('Percent', ascending=False)
    print(missing_values)
    print('='*50)
    print('')
    #========================================================

    # Median and KNN imputation are NOT performed here.
    # They will be fitted only on X_train inside
    print('Check Missing Values: Before Train/Test Imputation')
    missing_values = pd.DataFrame(
        {
            'Missing': df.isnull().sum(),
            'Percent': (df.isnull().mean() * 100).round(2)
        }
    )
    missing_values = missing_values.sort_values('Percent', ascending=False)
    print(missing_values)
    print('='*50)
    print('')
    #==============================================================

    # log radius
    df['log_radius'] = np.log10(df['radius'])
    #==============================================================
    # # round numerical columns to 4 decimal places
    # float_cols = df.select_dtypes(include=['floating']).columns
    # df[float_cols] = df[float_cols].round(4)

    # # save changes to a csv file
    # df = df.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\preprocessed.csv', index=False)

    return df
#===========================================================================
#==============================================================================
#============================================================================

# RandomForestRegressor training
def random_forest_reg(dff,features=None,train_model='RFR',median=None,knn=None):
    '''
    Parameters:
    --------------------------
    dff:          Dataframe preprocessed.csv
    features:     list of strings
    train_model:  string
    median:       list of strings
    knn:          list of strings

    Returns:
    ---------------------------
    Dataframe:   random_forest_reg.csv
    '''
    print(f'\n******************** RandomForestRegressor FUNCTION MODEL {train_model} ********************')
    print('='*50)
    print('='*50)
    print('='*50)
    print('')
    #=========================================================

    # initial information of radius column
    print(f"Number of NaN Values of Radius: {dff['radius'].isnull().sum()}")
    print(f"\n Number of Not NaN Values of Radius: {dff['radius'].notnull().sum()}")
    print(f'\n Dimention of DataSet: {dff.shape}')
    print('='*50)
    print('')
    #=================================================

    dff = dff.copy()
    # check mass before log
    if (dff.loc[dff['mass'].notna(),'mass'] <= 0).any():
        raise ValueError('There is zero or negative Mass values')

    # turn mass scale to log10
    dff['log_mass'] = np.log10(dff['mass'])
    #=================================================

    # split nan mass and not nan mass
    # to two dataset
    known = dff[dff['mass'].notna()].copy()
    unknown = dff[dff['mass'].isnull()].copy()
    #======================================================

    # train and test variables
    X = known[features].copy()
    y = known['log_mass'].copy()
    #======================================================

    RANDOM_STATE = 45
    # split by star_name
    # planets in the same star
    # will not appear in both train and test
    groups = known['star_name']

    # split train / test data
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)

    # split train/test data according to their group (star name)
    train_index, test_index = next(group_split.split(X, y, groups=groups))

    X_train = X.iloc[train_index].copy() # train features
    X_test = X.iloc[test_index].copy()   # test features
    y_train = y.iloc[train_index].copy() # train target
    y_test = y.iloc[test_index].copy()   # test target
    #=====================================================

    # All parameters are fitted only on X_train.
    median_features = [col for col in median if col in features]
    knn_features = [col for col in knn if col in features]

    # features which are not filled with KNN
    # are filled using train median
    other_features = [col for col in features if col not in knn_features and col not in median_features]
    median_features_all = (median_features + other_features)
    #=====================================================

    # group k-fold cross validation
    # imputation is inside each fold
    cv = GroupKFold(n_splits=5)

    # empty list for saving R^2 in each fold
    cv_scores = []

    # split groups of train
    # and sort its index from 0
    groups_train = groups.iloc[train_index].reset_index(drop=True)

    # sorting indexes of train from 0
    # to be the same as indexes of cv
    X_train_cv = X_train.reset_index(drop=True)
    y_train_cv = y_train.reset_index(drop=True)

    # for each fold split data to cv train / cv test 
    for fold, (cv_train_index, cv_test_index) in enumerate(cv.split(X_train_cv,y_train_cv,groups=groups_train), start=1):

        # making cv train / cv test in each fold
        X_cv_train = X_train_cv.iloc[cv_train_index].copy() 
        X_cv_test = X_train_cv.iloc[cv_test_index].copy()   
        y_cv_train = y_train_cv.iloc[cv_train_index].copy()
        y_cv_test = y_train_cv.iloc[cv_test_index].copy()   
        #------------------------------------

        # fill nan values in cv train
        # by median of than cv train
        fold_medians = X_cv_train[median_features_all].median()
        X_cv_train.loc[:, median_features_all] = (X_cv_train[median_features_all].fillna(fold_medians))
        X_cv_test.loc[:, median_features_all] = (X_cv_test[median_features_all].fillna(fold_medians))
        #--------------------------------

        # kmm imputation fitted omly on cv train fold
        fold_standard_scaler = StandardScaler()
        fold_imputer = KNNImputer(n_neighbors=5)

        # fit scaler only on cv train
        X_cv_train_knn_scaled = (fold_standard_scaler.fit_transform(X_cv_train[knn_features]))

        # fit knn only on cv train
        X_cv_train_knn_imputed = (fold_imputer.fit_transform(X_cv_train_knn_scaled))

        # transform cv test
        X_cv_test_knn_scaled = (fold_standard_scaler.transform(X_cv_test[knn_features]))
        X_cv_test_knn_imputed = (fold_imputer.transform(X_cv_test_knn_scaled))

        # return to original scale
        X_cv_train.loc[:,knn_features] = (fold_standard_scaler.inverse_transform(X_cv_train_knn_imputed))
        X_cv_test.loc[:,knn_features] = (fold_standard_scaler.inverse_transform(X_cv_test_knn_imputed))
        #----------------------------------------

        # make sure no missing values remain
        if X_cv_train[features].isnull().any().any():
            raise ValueError(f'NaN values remain in CV train fold {fold}')

        if X_cv_test[features].isnull().any().any():
            raise ValueError(f'NaN values remain in CV test fold {fold}')
        #-----------------------------------------        
        
        # making random forest model
        cv_model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        # fit model
        cv_model.fit(X_cv_train,y_cv_train)

        # prediction
        y_cv_predict = cv_model.predict(X_cv_test)

        # calculate r2
        fold_r2 = r2_score(y_cv_test, y_cv_predict)
        cv_scores.append(fold_r2)
        print(f'Fold {fold} r2 = {fold_r2}')

    # convert to numpy array
    cv_scores = np.array(cv_scores)

    # print results
    print(f'Cross Validation Score r2 (log_mass) = {cv_scores}')
    print(f'\nMean r2 = {cv_scores.mean()}')
    print(f'\n Standard Variation = {cv_scores.std()}')
    print('='*50)
    print('')
    #========================================================
    # =====================================================
    # final train / test imputation
    # fit on X_train
    # median imputation
    if len(median_features_all) > 0:
        train_medians = X_train[median_features_all].median()
        X_train.loc[:, median_features_all] = (X_train[median_features_all].fillna(train_medians))
        X_test.loc[:, median_features_all] = (X_test[median_features_all].fillna(train_medians))
    else:
        train_medians = pd.Series(dtype=float)
    #=====================================================

    # knn imputation
    if len(knn_features) > 0:
        standard_scaler = StandardScaler()
        imputer = KNNImputer(n_neighbors=5)

        # fit scaler only on train
        X_train_knn_scaled = (standard_scaler.fit_transform(X_train[knn_features]))

        # fit knn only on train
        X_train_knn_imputed = (imputer.fit_transform(X_train_knn_scaled))

        # transform test using train scaler/imputer
        X_test_knn_scaled = (standard_scaler.transform(X_test[knn_features]))
        X_test_knn_imputed = (imputer.transform(X_test_knn_scaled))

        # return to original scale
        X_train.loc[:,knn_features] = (standard_scaler.inverse_transform(X_train_knn_imputed))
        X_test.loc[:,knn_features] = (standard_scaler.inverse_transform(X_test_knn_imputed))
    #=====================================================

    # make sure no missing values remain
    if X_train[features].isnull().any().any():
        raise ValueError('NaN values remain in X_train after imputation.')

    if X_test[features].isnull().any().any():
        raise ValueError('NaN values remain in X_test after imputation.')
    #=====================================================

    # making random forest model
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    #========================================================

    # fit model
    model.fit(X_train, y_train)
    #========================================================


    # features importance
    print("******************** Feature Importance ********************")
    print('='*50)
    print('')
    feature_importance = pd.DataFrame(
        {
        'Feature': features,
        'Importance': model.feature_importances_
        }
    )

    # sort features by importance
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # print results
    print(feature_importance.to_string(index=False))
    print('='*50)
    print('')
    #=====================================================

    # predict test values
    y_predict_test = model.predict(X_test)
    #========================================================

    # log space
    r2_log = r2_score(y_test, y_predict_test)
    mae_log = mean_absolute_error(y_test, y_predict_test)
    #========================================================

    # real scale
    y_test_real = 10**y_test
    y_predict_test_real = 10**y_predict_test
    mae_real = mean_absolute_error(y_test_real, y_predict_test_real)
    r2_real = r2_score(y_test_real, y_predict_test_real)
    #========================================================

    # print results
    print(f'r2 Score in Log Space: {r2_log}')
    print(f'\nMean Absolute Error(log): {mae_log}')
    print(f'\nr2 Score in Real Space: {r2_real}')
    print(f'\nMean Absolute Error(Real): {mae_real}')
    print('='*50)
    print('')
    #==========================================================

    # calculate relative error
    # relative error must be calculated using
    # real mass, not log mass.
    relative_error_real = (np.abs(y_predict_test_real - y_test_real) / y_test_real)
    print(f'Median Relative Error: {np.median(relative_error_real)}')
    print(f'Mean Relative Error: {np.mean(relative_error_real)}')
    print('='*50)
    print('')
    #===========================================================
    #===========================================================

    # Error Analysis in real space
    error_df = pd.DataFrame(
        {
            'Actual_Mass': y_test_real,
            'Predicted_Mass': y_predict_test_real,
            'Relative_Error': relative_error_real
        }
    )
    #=========================================================

    # mass groups
    mass_bins = [0,0.03,0.3,13,np.inf]
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
    #=========================================================

    # calculate error for each group
    group_error = error_df.groupby('Mass_Group',observed=True)['Relative_Error'].agg( ['count','mean','median'])
    print("Error Analysis by Mass Group")
    print(group_error)
    print('='*50)
    print('')
    #=========================================================

    # plot error analysis
    plt.figure(figsize=(12, 8))
    group_error['mean'].plot(kind='bar')
    plt.title(f'Prediction Error by Planet Mass Model {train_model}')
    plt.xlabel('Planet Mass Group')
    plt.ylabel('Mean Relative Error')
    plt.grid(alpha=0.5,ls='--')
    plt.tight_layout()
    plt.show()
    #=========================================================

    # plot predicted / actual mass(log / reall)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle("Predicted Vs Actual Mass (Log / Real)")

    # colors for mass groups
    group_colors = {
        'super_earth': 'blue',
        'neptune': 'green',
        'jupiter': 'orange',
        'massive_planet': 'purple'
    }

    # log space
    for group, color in group_colors.items():
        mask = error_df['Mass_Group'] == group
        sns.scatterplot(x = y_test[mask], y = y_predict_test[mask], color = color, label=group.replace('_', ' ').title(), ax=axes[0])

    # best prediction line
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], ls='--', color='red', label='Best Prediction')
    axes[0].set_title(f"Predicted Vs Actual Mass(Log) Model {train_model}")
    axes[0].set_xlabel("Actual Mass(log)")
    axes[0].set_ylabel("Predicted Mass(log)")
    axes[0].legend()
    axes[0].grid(alpha=0.5, ls='--')
    
    # real space
    for group, color in group_colors.items():
        mask = error_df['Mass_Group'] == group
        sns.scatterplot(x=y_test_real[mask], y=y_predict_test_real[mask], color=color, label=group.replace('_', ' ').title(), ax=axes[1])

    # best prediction line
    axes[1].plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], ls='--', color='red', label='Best Prediction')
    axes[1].set_title(f"Predicted Vs Actual Mass(Real) Model {train_model}")
    axes[1].set_xlabel("Actual Mass(Real)")
    axes[1].set_ylabel("Predicted Mass(Real)")
    axes[1].legend()
    axes[1].grid(alpha=0.5, ls='--')
    plt.tight_layout()
    plt.show()
    #==========================================================

    # residual plot
    residual = y_test - y_predict_test

    # colors for mass groups
    group_colors = {
        'super_earth': 'blue',
        'neptune': 'green',
        'jupiter': 'orange',
        'massive_planet': 'purple'
    }

    plt.figure(figsize=(12, 8))
    # plot each mass group
    for group, color in group_colors.items():
        mask = error_df['Mass_Group'] == group
        plt.scatter(y_predict_test[mask], residual[mask], color=color, label=group.replace('_', ' ').title())

    # zero residual line
    plt.axhline(y=0, ls='--', color='red', label='Zero Error')
    plt.title(f"Residual Plot by Mass Group - Model {train_model}")
    plt.xlabel("Predicted Mass(log)")
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid(alpha=0.5, ls='--')
    plt.tight_layout()
    plt.show()

    # residual = (y_test - y_predict_test)
    # plt.figure(figsize=(12, 8))
    # plt.scatter(y_predict_test,residual,color='blue',label='Residuals of Prediction')
    # plt.axhline(y=0,ls='--')
    # plt.title(f"Residual Plot Model {train_model}")
    # plt.xlabel("Predicted Mass(log)")
    # plt.ylabel("Residuals")
    # plt.legend()
    # plt.grid(alpha=0.5,ls='--')
    # plt.tight_layout()
    # plt.show()
    #==============================================================
    #==============================================================

    # now train model to predict unknown masses
    X_unknown = unknown[features].copy()
    #=====================================================

    # apply SAME median values learned from X_train
    if len(median_features_all) > 0:
        X_unknown.loc[:,median_features_all] = X_unknown[median_features_all].fillna(train_medians)
    #=====================================================

    # apply SAME scaler and KNN imputer
    # learned from X_train
    if len(knn_features) > 0:
        X_unknown_knn_scaled = (standard_scaler.transform(X_unknown[knn_features]))
        X_unknown_knn_imputed = (imputer.transform(X_unknown_knn_scaled))
        X_unknown.loc[:,knn_features] = ( standard_scaler.inverse_transform(X_unknown_knn_imputed))
    #=====================================================

    # make sure no missing values remain
    if X_unknown[features].isnull().any().any():
        raise ValueError('NaN values remain in X_unknown after imputation.')
    #=====================================================

    # prediction + uncertainly for unknown mass
    # prediction of every individual tree
    tree_predictions_log = np.array([tree.predict(X_unknown) for tree in model.estimators_])

    # mean prediction in log space
    predict_log_mass = tree_predictions_log.mean(axis=0)

    # standard deviation of tree predictions in log space
    predict_log_mass_std = tree_predictions_log.std(axis=0, ddof=1)

    # 16th and 84th percentiles
    # approximately corresponds to ±1 sigma
    predict_log_mass_lower = np.percentile(tree_predictions_log, 16, axis=0)
    predict_log_mass_upper = np.percentile(tree_predictions_log, 84, axis=0)

    # convert from log10 mass to real mass
    predict_mass = 10**predict_log_mass
    predict_mass_lower = 10**predict_log_mass_lower
    predict_mass_upper = 10**predict_log_mass_upper

    # asymmetric uncertainty in real mass
    mass_error_lower = predict_mass - predict_mass_lower
    mass_error_upper = predict_mass_upper - predict_mass

    # save predictions
    unknown['log_mass'] = predict_log_mass
    unknown['mass'] = predict_mass
    unknown['mass_error_lower'] = mass_error_lower
    unknown['mass_error_upper'] = mass_error_upper
    unknown['mass_lower_1sigma'] = predict_mass_lower
    unknown['mass_upper_1sigma'] = predict_mass_upper
    unknown['log_mass_error'] = predict_log_mass_std
    unknown['mass_source'] = 'Predicted'

    known['mass_source'] = 'Observed'
    # known['mass_error_lower'] = np.nan
    # known['mass_error_upper'] = np.nan
    known['mass_lower_1sigma'] = np.nan
    known['mass_upper_1sigma'] = np.nan
    known['log_mass_error'] = np.nan
    #===============================================


    # merge datasets
    filled_mass_dataset = pd.concat([known,unknown],ignore_index=True)

    # # round numerical columns to 4 decimal places
    # float_cols = filled_mass_dataset.select_dtypes(include=['floating']).columns
    # filled_mass_dataset[float_cols] = (filled_mass_dataset[float_cols].round(4))

    print('')
    print(f"Number of NaN Mass Values After using Model: {filled_mass_dataset['mass'].isnull().sum()}")
    print('-'*50)
    print('')
    print('Final Information After Using Model: ')
    print('-'*50)
    print(filled_mass_dataset.info())
    print('='*50)
    print('='*50)
    print('='*50)
    #============================================

    # # save changes to new dataset
    # filled_mass_dataset = filled_mass_dataset.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\random_forest_reg.csv', index=False)

    return filled_mass_dataset
#=============================================================
#=============================================================
#==================================================================

def comp_dataset(df_original, df_model, train_model='RFR'):
    '''
    Parameters:
    --------------------
    df_original:      Dataframe exoplanet.eu_catalog_03-08-26_16_42_07.csv
    df_model:         Dataframe random_forest_reg.csv
    train_model:      string

    Returns:
    ---------------------
    Dataframe:        comp_dataset.csv
    '''

    print('')
    print(f'******************** FUNCTION COMPLETE DATASET MODEL {train_model} ********************')
    print('='*50)
    print('='*50)
    print('='*50)
    print('')
    # planets with their detection type
    print("Detection Type of the Most of NaN Radius: ")
    print('')

    nan_radius = df_original[df_original['radius'].isnull() & df_original['detection_type'].notna()].copy()
    detect_type_count = (nan_radius['detection_type'].value_counts().sort_values(ascending=False))
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

    # check if all these detection types have mass or not
    removed_planets = df_original[df_original['detection_type'].isin(['Radial Velocity','Microlensing','Radial Velocity, Astrometry','Timing','Imaging'])].copy()
    mask_rp = (removed_planets['mass'].notna() & removed_planets['radius'].isnull())
    removed_planets = removed_planets.loc[mask_rp].copy()
    removed_planets['mass_source'] = 'No Radius Observed'
    print('Number of NaN Radius / Not NaN values Mass in Radial Velocity Detection Type: ')
    print(len(removed_planets['mass']))
    print('='*50)
    print('')
    #===============================================

    # concat this dataset with the dataset
    # which filled by model A
    df_final = pd.concat([df_model, removed_planets],ignore_index=True)
    
    # # round numerical columns to 4 decimal places
    # float_cols = df_final.select_dtypes(include=['floating']).columns
    # df_final[float_cols] = df_final[float_cols].round(4)

    print('='*50)
    print('='*50)
    print('='*50)
    print('')
    df_final['mass_source'] = df_final['mass_source'].fillna('Observed')
    # # save changer to a new dataset
    # df_final = df_final.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\comp_dataset.csv', index=False)
    return df_final
#=======================================================================

def remain(df):
    '''
    Parameters:
    -----------------------
    df:         Dataframe comp_dataset.csv

    Returns:
    -----------------------
    Dataframe:  RandomForestRegressor_Model.csv
    '''
    print('***************** REMAIN FUNCTION **********************')
    print('='*50)
    print('')

    # fill with KNN
    num_cols = [
    'star_mass'
    ]
    standard_scaler = StandardScaler()
    scaled_values = standard_scaler.fit_transform(df[num_cols])
    imputer = KNNImputer(n_neighbors = 5)
    imputed_scaled = imputer.fit_transform(scaled_values)
    df[num_cols] = standard_scaler.inverse_transform(imputed_scaled)

    # fill with Median
    df['semi_major_axis'] = (df['semi_major_axis'].fillna(df['semi_major_axis'].median())).round(4)

    # print missing values for the last time
    print('Check Missing Values after Filling Remain Missin Values with KNN(Star Mass) and Median(Semi Major Axis)')
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

    # print final information of dataset
    print("The Final Information of Dataset: ")
    print(df.info())
    print('='*50)
    print('='*50)
    print('='*50)
    print('')

    # # round numerical columns to 4 decimal places
    # float_cols = df.select_dtypes(include=['floating']).columns
    # df[float_cols] = df[float_cols].round(4)

    # # save changes to new dataset
    # df = df.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\RFR\RandomForestRegressor_Model.csv', index=False)
    return df
#==================================================================
#===================================================================
#===================================================================

# lagrange's functions
# jupiter mass in solar masses
jupiter_to_sun_mass = 0.000954588
#==================================

# lagrange equation for L1, L2, L3
def lagrange_equation(x, b):
    '''
    Parameters:
    ---------------- 
    x:     float
    b:     float

    Returns:
    -------------------
    lagrange equation:     float
    '''
    star_x = -b
    planet_x = 1 - b

    term_star = (1 - b) * (x - star_x) / abs(x - star_x)**3
    term_planet = b * (x - planet_x) / abs(x - planet_x)**3

    return x - term_star - term_planet
#=========================================

# calculation of L1, L2, L3
def calculate_L1_L2_L3(b):
    '''
    Parameters:
    ---------------
    b:     float

    Returns:
    -------------
    x_L1, x_L2, x_L3:     float
    '''
    star_x = -b
    planet_x = 1 - b

    # epsilon
    epsilon = 1e-10

    # calculate L1
    x_L1 = brentq(lagrange_equation, star_x + epsilon, planet_x - epsilon, args=(b,))

    # calculate L2
    x_L2 = brentq(lagrange_equation, planet_x + epsilon, planet_x + 2, args=(b,))

    # calculate L3
    x_L3 = brentq(lagrange_equation, -2, star_x - epsilon, args=(b,))

    return x_L1, x_L2, x_L3
#====================================================


# calculate L1 ... L5 for one star_planet
def calculate_lagrange_points(star_mass, planet_mass, semi_major_axis):
    '''
    Parameter:
    ----------------
    star_mass:       float
    planet_mass:     float
    semi_major_axis: float

    Returns:
    ----------------
    'x_L1_AU':             float
    'distance_star_L1':    float 
    'distance_planet_L1':  float
    'x_L2_AU':             float
    'distance_star_L2':    float 
    'distance_planet_L2':  float
    'x_L3_AU':             float
    'distance_star_L3':    float 
    'distance_planet_L3':  float
    'x_L4_AU':             float
    'y_L4_AU':             float
    'distance_star_L4':    float 
    'distance_planet_L4':  float
    'x_L5_AU':             float
    'y_L5_AU':             float
    'distance_star_L5':    float 
    'distance_planet_L5':  float
    '''
    # turn mass unit (jupiter -> solar)
    planet_mass_solar = planet_mass * jupiter_to_sun_mass
    total_mass = star_mass + planet_mass_solar

    # mass ratio
    b = planet_mass_solar / total_mass

    # calculate L1, L2, L3
    x_L1, x_L2, x_L3 = calculate_L1_L2_L3(b)

    # calculate L4, L5
    x_L4 = 0.5 - b
    y_L4 = np.sqrt(3) / 2
    x_L5 = 0.5 - b
    y_L5 = -np.sqrt(3) / 2

    # turn cordinate unit to AU
    x_L1_AU = x_L1 * semi_major_axis
    x_L2_AU = x_L2 * semi_major_axis
    x_L3_AU = x_L3 * semi_major_axis
    x_L4_AU = x_L4 * semi_major_axis
    x_L5_AU = x_L5 * semi_major_axis
    y_L4_AU = y_L4 * semi_major_axis
    y_L5_AU = y_L5 * semi_major_axis
    #=====================================

    # star and planet position in barycentric coordinates
    star_position = -b * semi_major_axis
    planet_position = (1 - b) * semi_major_axis
    #==============================================

    # calculate points distance from star 
    distance_star_L1 = abs(x_L1_AU - star_position)
    distance_star_L2 = abs(x_L2_AU - star_position)
    distance_star_L3 = abs(x_L3_AU - star_position)
    distance_star_L4 = np.sqrt((x_L4_AU - star_position)**2 + y_L4_AU**2)
    distance_star_L5 = np.sqrt((x_L5_AU - star_position)**2 + y_L5_AU**2)
    #===============================================

    # calculate points distnce from planet
    distance_planet_L1 = abs(x_L1_AU - planet_position)
    distance_planet_L2 = abs(x_L2_AU - planet_position)
    distance_planet_L3 = abs(x_L3_AU - planet_position)
    distance_planet_L4 = np.sqrt((x_L4_AU - planet_position)**2 + y_L4_AU**2)
    distance_planet_L5 = np.sqrt((x_L5_AU - planet_position)**2 + y_L5_AU**2)

    return {
        'x_L1_AU': x_L1_AU, 
        'distance_star_L1': distance_star_L1, 
        'distance_planet_L1': distance_planet_L1,
        'x_L2_AU': x_L2_AU, 
        'distance_star_L2': distance_star_L2, 
        'distance_planet_L2': distance_planet_L2,
        'x_L3_AU': x_L3_AU, 
        'distance_star_L3': distance_star_L3, 
        'distance_planet_L3': distance_planet_L3,
        'x_L4_AU': x_L4_AU, 
        'y_L4_AU': y_L4_AU, 
        'distance_star_L4': distance_star_L4, 
        'distance_planet_L4': distance_planet_L4,
        'x_L5_AU': x_L5_AU, 
        'y_L5_AU': y_L5_AU, 
        'distance_star_L5': distance_star_L5, 
        'distance_planet_L5': distance_planet_L5
    }
#============================================================


def calculate_lagrange_uncertainty(star_mass, planet_mass, mass_error_lower, mass_error_upper, semi_major_axis, n_samples=5000, random_state=45):
    '''
    Parameters:
    -----------------------
    star_mass:          float
    planet_mass:        float
    mass_error_lower:   float
    mass_error_upper:   float
    semi_major_axis:    float
    n_samples:          int
    random_state:       int

    Returns:
    -------------------------
    dict:               Dictionary containing the median values of the Lagrange points and 
                        their lower and upper uncertainties.
    '''
    rng = np.random.default_rng(random_state)

    # Create asymmetric normal-like mass distribution
    # sigma values in real mass
    sigma_lower = mass_error_lower
    sigma_upper = mass_error_upper

    # random normal samples
    random_values = rng.normal(loc=0, scale=1, size=n_samples)

    # asymmetric uncertainty
    mass_samples = np.where(random_values < 0,planet_mass + random_values * sigma_lower,planet_mass + random_values * sigma_upper)

    # remove physically impossible negative masses
    mass_samples = mass_samples[mass_samples > 0]

    # Calculate Lagrange points for every mass sample

    lagrange_samples = []
    for sampled_mass in mass_samples:
        points = calculate_lagrange_points(star_mass=star_mass, planet_mass=sampled_mass, semi_major_axis=semi_major_axis)
        lagrange_samples.append(points)

    lagrange_samples = pd.DataFrame(lagrange_samples)

    # Calculate median and uncertainty
    result = {}
    for column in lagrange_samples.columns:
        median_value = np.median(lagrange_samples[column])
        lower_value = np.percentile(lagrange_samples[column], 16)
        upper_value = np.percentile(lagrange_samples[column], 84)
        result[column] = median_value
        result[column + '_error_lower'] = (median_value - lower_value)
        result[column + '_error_upper'] = (upper_value - median_value)

    return result
#===========================================================
