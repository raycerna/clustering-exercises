# IMPORTS
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from env import host, username, password


# SQL CONNECTION
def get_db_url(db_name):

    '''
    Connect to the SQL database with credentials stored in env file.
    Function parameter is the name of the database to connect to.
    Returns url.
    '''
    
    # Creates the url and the function returns this url
    url = f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    return (url)

    

# ACQUIRE
def get_zillow_data():

    '''
    Connect to SQL Database with url function called within this function.
    Checks if database is already saved to computer in csv file.
    If no file found, saves to a csv file and assigns database to df variable.
    If file found, just assigns database to df variable.
    '''
    
    # data_name allows the function to work no matter what a user might have saved their file name as
    # First, we check if the data is already stored in the computer
    # First conditional runs if the data is not already stored in the computer
    if os.path.isfile('zillow.csv') == False:

    # Querry selects the whole predicstion_2017 table from the database
        sql = '''
                        SELECT
                        prop.*,
                        predictions_2017.logerror,
                        predictions_2017.transactiondate,
                        air.airconditioningdesc,
                        arch.architecturalstyledesc,
                        build.buildingclassdesc,
                        heat.heatingorsystemdesc,
                        landuse.propertylandusedesc,
                        story.storydesc,
                        construct.typeconstructiondesc
FROM properties_2017 prop
JOIN (
    SELECT parcelid, MAX(transactiondate) AS max_transactiondate
    FROM predictions_2017
    GROUP BY parcelid
) pred USING(parcelid)
JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                      AND pred.max_transactiondate = predictions_2017.transactiondate
LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
LEFT JOIN storytype story USING (storytypeid)
LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
WHERE prop.latitude IS NOT NULL
  AND prop.longitude IS NOT NULL
  AND transactiondate <= '2017-12-31'
'''
        # Connecting to the data base and using the query above to select the data
        # the pandas read_sql function reads the query into a DataFrame
        df = pd.read_sql(sql, get_db_url('zillow'))
        # The pandas to_csv function writes the data frame to a csv file
        # This allows data to be stored locally for quicker exploration and manipulation
        df.to_csv('zillow.csv')

        # If any duplicates found, this removes them
        # df.columns.duplicated() returns a boolean array, True for a duplicate or False if it is unique up to that point
        # Use ~ to flip the booleans and return the df as any columns that are not duplicated
        # df.loc accesses a group of rows and columns by label(s) or a boolean array
        df = df.loc[:,~df.columns.duplicated()]
        df = df.drop('pid',axis=1)

         # The pandas to_csv function writes the data frame to a csv file
        # This allows data to be stored locally for quicker exploration and manipulation
        df.to_csv('zillow.csv')

    # This conditional runs if the data has already been saved as a csv (if the function has already been run on your computer)
    else:
        # Reads the csv saved from above, and assigns to the df variable
        df = pd.read_csv('zillow.csv', index_col=0)

    return df

def overview(df):
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing/rows
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'pct_rows_missing': pct_missing})
    cols_missing.set_index(df.columns)

def nulls_by_rows(df):
    num_missing = df.isnull().sum(axis=1)
    rows = df.shape[1]
    pct_missing = num_missing/rows
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'pct_cols_missing': pct_missing})

def handle_missing_values(df, prop_required_column, prop_required_row):
    cols_missing = nulls_by_columns(df)
    drop_list = list(cols_missing[cols_missing.pct_rows_missing > prop_required_column].index)
    df = df.drop(columns=drop_list)
    rows_missing = nulls_by_rows(df)
    df = df[rows_missing.pct_cols_missing > prop_required_row]
    return df



#def acquire():
   # if os.path.exists('zillow.csv'):
    #    df = pd.read_csv('zillow.csv')
   # else:
     #   database = 'zillow'
     #   url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
     #   df = pd.read_sql(query, url)
     #   df.to_csv('zillow.csv', index=False)
   # return df


