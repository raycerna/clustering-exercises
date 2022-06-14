from collections import Counter
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import math

from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#########################################################################################
def split_zillow_data(df):
    '''
    This function performs split on zillow data.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('Train: %d rows, %d cols' % train.shape)
    print('Validate: %d rows, %d cols' % validate.shape)
    print('Test: %d rows, %d cols' % test.shape)
    return train, validate, test

##########################################################################################
def handle_missing_values(df, prop_required_column=0.5 , prop_required_row=0.5):
    '''
    This function takes in a pandas dataframe, default proportion of required columns (set to 50%)
    and proportion of required rows (set to 75%). It drops any rows or columns that contain null
    values more than the threshold specified from the original dataframe and returns that dataframe.
    Prior to returning that data, prints statistics and list counts/names of removed rows/cols 
    '''
    original_cols = df.columns.to_list()
    original_rows = df.shape[0]
    threshold = int(round(prop_required_column * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    remaining_cols = df.columns.to_list()
    remaining_rows = df.shape[0]
    dropped_col_count = len(original_cols) - len(remaining_cols)
    dropped_cols = list((Counter(original_cols) - Counter(remaining_cols)).elements())
    
    print(f'The following {dropped_col_count} columns were dropped because they were missing more\
    than {prop_required_column * 100}% of data: \n{dropped_cols}\n')
    dropped_rows = original_rows - remaining_rows
    
    print(f'{dropped_rows} rows were dropped because they were missing more than\
          {prop_required_row * 100}% of data')
          
    return df
############################################################################################
# combined in one function
def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.5):
    '''
    This function calls the remove_columns and handle_missing_values
    to drop columns that need to be removed. It also drops rows and columns that have more 
    missing values than the specified threshold.
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df
############################################################################################
def remove_columns(df, cols_to_remove):
    '''
    This function takes in a pandas dataframe and a list of columns to remove.
    It drops those columns from the original df and returns the df.
    '''
    df = df.drop(columns=cols_to_remove)
    return df
############################################################################################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe and return that dataframe'''  
    for col in col_list:
        # get quartiles
        q1, q3 = df[f'{col}'].quantile([.25, .75])  
        # calculate interquartile range
        iqr = q3 - q1   
        # get upper bound
        upper_bound = q3 + k * iqr 
        # get lower bound
        lower_bound = q1 - k * iqr   
        # return dataframe without outliers        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]        
    return df
############################################################################################

def prep_zillow(df):
    '''
    Returns clean dataframe
    '''
    df = data_prep(df)    
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
      (df.propertylandusedesc == 'Mobile Home') |
      (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes') |
      (df.propertylandusedesc == 'Cluster Home')]    
    # Remove properties that couldn't even plausibly be a studio. 
    df= df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]    
    # Remove properties where there is not a single bathroom.
    df = df[df.bathroomcnt > 0]    
    # keep only properties with square footage greater than 70 (legal size of a bedroom)
    df = df[df.calculatedfinishedsquarefeet > 70]    
    # Minimum lot size of single family units.
    df = df[df.lotsizesquarefeet >= 5000].copy()    
    #df = df[~df['propertylandusetypeid'].isin([263, 265, 275])]
    # Clear indicators of single unit family. Other codes non-existent or indicate commercial sites. 
    # 0100 - Single Residence
    # 0101 Single residence with pool
    # 0104 - Single resident with therapy pool 
    df = df[(df.propertycountylandusecode == '0100') |
            (df.propertycountylandusecode == '0101') |
            (df.propertycountylandusecode == '0104') |
            (df.propertycountylandusecode == '122') | 
            (df.propertycountylandusecode == '1111') |
            (df.propertycountylandusecode == '1110') |
            (df.propertycountylandusecode == '1')
           ]   
    # Remove 13 rows where unit count is 2. The NaN's can be safely
    # assumed as 1 and were just mislabeled in other counties.  
    df = df[df['unitcnt'] != 2]
    df['unitcnt'].fillna(1) 
    # Property where finished area is 152 but bed count is 5. 
    df = df.drop(labels=75325, axis=0)          
    # Redudant columns or uninterpretable columns
    # Unit count was dropped because now its known that theyre all 1. 
    # Finished square feet is equal to calculated sq feet. 
    # full bathcnt and calculatedbathnbr are equal to bathroomcnt
    # property zoning desc is unreadable. 
    # assessment year is unnecessary, all values are 2016. 
    # property land use desc is always single family residence 
    # same with property landuse type id. 
    # room count must be for a different category, as it is always 0.
    # regionidcounty reveals the same information as FIPS. 
    # heatingorsystemtypeid is redundant. Encoded descr. 
    # Id does nothing, and parcelid is easier to represent.
    df =df.drop(columns= ['finishedsquarefeet12', 'fullbathcnt', 'calculatedbathnbr',
                      'propertyzoningdesc', 'unitcnt', 'propertylandusedesc',
                      'assessmentyear', 'roomcnt', 'regionidcounty', 'propertylandusetypeid',
                      'heatingorsystemtypeid', 'id', 'heatingorsystemdesc', 'buildingqualitytypeid'],
            axis=1)    
    # The last nulls can be dropped altogether. 
    df = df.dropna()
    df['yearbuilt'] = df['yearbuilt'].astype(int)
    df.yearbuilt = df.yearbuilt.astype(object) 
    df['age'] = 2017-df['yearbuilt']
    df = df.drop(columns='yearbuilt')
    df['age'] = df['age'].astype('int')
    print('Yearbuilt converted to age. \n')   
    # Removing problematic outlier groups.  
    df = remove_outliers(df, 3, ['lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                                'landtaxvaluedollarcnt', 'taxamount', 'calculatedfinishedsquarefeet']) 
    #df = df.set_index('parcelid')
    df['county'] = np.where(df.fips == 6037, 'Los Angeles', np.where(df.fips == 6059, 'Orange','Ventura') )
    #df = df.drop(columns = ‘fips’)

    return df
    

############################################################################################   
    
def nulls_by_col(df):
    '''
    This function  takes in a dataframe of observations and attributes(or columns)
    and returns a dataframe where each row is an atttribute name, the first column is the 
    number of rows with missing values for that attribute, and the second column is percent
    of total rows that have missing values for that attribute.
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = (num_missing / rows * 100)
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 
                                 'percent_rows_missing': prcnt_miss})\
    .sort_values(by='percent_rows_missing', ascending=False)
    return cols_missing.applymap(lambda x: f"{x:0.1f}")
############################################################################################
def nulls_by_row(df):
    '''
    This function takes in a dataframe and returns a dataframe with 3 columns:
    the number of columns missing, percent of columns missing, 
    and number of rows with n columns missing.
    '''
    num_missing = df.isnull().sum(axis = 1)
    prcnt_miss = (num_missing / df.shape[1] * 100)
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 
                                 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index().set_index('num_cols_missing')\
    .sort_values(by='percent_cols_missing', ascending=False)
    return rows_missing
############################################################################################ 

def overview(df):
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))
############################################################################################

def percentage_stacked_plot(columns_to_plot, title, prep_zillow):
    
    '''
    Returns a 100% stacked plot of the response variable for independent variable of the list columns_to_plot.
    Parameters: columns_to_plot (list of string): Names of the variables to plot
    '''
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(title, fontsize=22,  y=.95)
 

    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(prep_zillow[column], prep_zillow['logerror']).apply(lambda x: x/x.sum()*100, axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['#94bad4','#ebb086'])

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='LogError', fancybox=True)

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)

    return percentage_stacked_plot


###