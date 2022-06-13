'''
Functions to acquire data for Zillow clustering model.
Seperate functions to connect to sql database and save data to csv.
Login credentials in env file are required.
'''

# IMPORTS
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from env import host, username, password


# SQL CONNECTION
#def get_db_url(db_name):

'''
    Connect to the SQL database with credentials stored in env file.
    Function parameter is the name of the database to connect to.
    Returns url.
'''
    
    # Creates the url and the function returns this url
    #url = f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    #return (url)


# ACQUIRE
#def get_zillow_data():

'''
    Connect to SQL Database with url function called within this function.
    Checks if database is already saved to computer in csv file.
    If no file found, saves to a csv file and assigns database to df variable.
    If file found, just assigns database to df variable.
'''
    
    # data_name allows the function to work no matter what a user might have saved their file name as
    # First, we check if the data is already stored in the computer
    # First conditional runs if the data is not already stored in the computer
    #if os.path.isfile('zillow.csv') == False:

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

def overview(df):
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def nulls_by_rows(df):
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()

def handle_missing_values(df, prop_required_column, prop_required_row):
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df

#def drop_column(df):
    #df.iloc[:,1]
    #return df

def acquire():
    if os.path.exists('zillow.csv'):
        df = pd.read_csv('zillow.csv')
    else:
        database = 'zillow'
        url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv', index=False)
    return df

        # Connecting to the data base and using the query above to select the data
        # the pandas read_sql function reads the query into a DataFrame
        #df = pd.read_sql(sql, get_db_url('zillow'))

        # The pandas to_csv function writes the data frame to a csv file
        # This allows data to be stored locally for quicker exploration and manipulation
        #df.to_csv('zillow.csv')

    # This conditional runs if the data has already been saved as a csv (if the function has already been run on your computer)
    #else:
        # Reads the csv saved from above, and assigns to the df variable
       # df = pd.read_csv('zillow.csv', index_col=0)

   # return df
