'''
  Helpers for artificial datasets

  Updates:
  12.11.2020 - Add artificial dataset as a time-series

'''

import pandas as pd
import os
import numpy as np
import fcn_helpers as fhelp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Generate samples
num_samples = 1000

# Define the products
sku_id = ['sku_A','sku_B','sku_C']
mu_sales = [100, 10, 30]
sigma_sales = [20, 4, 8]

baseline_prices_mu = [7,3,12]
baseline_prices_sigma = [1,0.5,1.5]

price_sales_corrcoef = [0.6,0.2,0.4]

sales_mu_SKU_C_promo = 80
sales_sigma_SKU_C_promo = 10
price_mu_SKU_C_promo = 9
price_sigma_SKU_C_promo = 1.2
price_sales_corrcoef_SKU_C_promo = 0.80


def atomic_sales(num_samples, sales_mu, sales_sigma, 
  price_mu, price_sigma, 
  price_sales_corrcoef=0.25,
  noise_strength = 1/6,
  shelf_capacity=None, shelf_impact=None):
  '''
    Sales model for the most atomic unit. It can be a product or a store.

    The method correlates prices and sales,
    although the correlation coefficient can be set to 0.

  '''
  # Equivalent of the Cholesky decomposition of the covariance matrix for two signals
  L = np.array([[1,price_sales_corrcoef], [0,np.sqrt(1-price_sales_corrcoef**2)]])
  
  # Define two random gaussian signals to produce the correlated ones
  noise = np.random.normal(0.0, 1.0, size=(num_samples, 2))
  correlated_signals = np.matmul(noise, L)


  # 1 - Scale the signals to get price and sales
  prices = correlated_signals[:,0]*price_sigma + price_mu
  baseline_sales = correlated_signals[:,1]*sales_sigma + sales_mu


  # 2 - Shelves
  if shelf_capacity and shelf_impact:
    shelf_dump_factor = sales_mu*shelf_impact
    shelves = np.random.choice(shelf_capacity, num_samples, replace=True)
    shelf_sales = np.divide(shelves, baseline_sales)*shelf_dump_factor
  else:
    shelf_sales = np.zeros(num_samples)
    shelves = np.zeros(num_samples)

  # 3 - Noise. Uniform noise from 0 to mu_sales*noise_strength
  noise = np.random.randint(0, round(sales_mu*noise_strength), size=(num_samples))

  df_atomic = pd.DataFrame({
    'baseline_sales': baseline_sales,
    'price': prices,
    'shelf_sales': shelf_sales,
    'shelves': shelves,
    'noise': noise
  })
  return df_atomic



def generate_dataset():
  '''
    Parameters defined in the script (outside this function)
  '''

  idx_promos = np.zeros(num_samples, dtype=bool)
  week_number = np.linspace(1, num_samples, num=num_samples)
  total_sales_volume = np.zeros(num_samples)

  all_skus = []
  for idx_store in range(0, len(sku_id)):

    df = atomic_sales(num_samples, mu_sales[idx_store], sigma_sales[idx_store], 
      baseline_prices_mu[idx_store], baseline_prices_sigma[idx_store], 
      price_sales_corrcoef=price_sales_corrcoef[idx_store],
      noise_strength = 1/6,
      shelf_capacity=None, shelf_impact=None)

    # Just in case
    df['price'] = df['price'].apply(lambda x: np.abs(x))
    
    #sales = noise + randn(mu_sales, sigma_sales) + shelf_capacity*2 + discount*5 
    
    df['sku_id'] = sku_id[idx_store]
    df['promos'] = idx_promos
    df['week_number'] = week_number

    #  Calculate the sales volume
    sales_volume = (df['price']*df['baseline_sales']).values
    df['sales_volume'] = sales_volume
    # Total volume across all SKUs
    total_sales_volume += sales_volume

    df['x1'] = np.random.randint(0, 100, size=(num_samples))
    df['x2'] = np.random.randint(0, 200, size=(num_samples))

    # For backwards compatibility
    #df['sales'] = df['baseline_sales']

    col_map = {iCol: f'{iCol}_{ sku_id[idx_store]}' for iCol in df.columns.tolist()}
    df.rename(columns=col_map, inplace=True)

    all_skus.append(df)

  all_skus_df = pd.concat(all_skus, axis=1)
  all_skus_df.reset_index(inplace=True)

  # get the total volume of sales per category
  all_skus_df['category_volume'] = total_sales_volume  
  
  # Prepare the promos for SKU C (random)
  num_promos = int(num_samples*0.2)
  promos_rnd = np.random.choice(num_samples-1, num_promos, replace=False)
  idx_promos[promos_rnd] = True

  
  #sales_multiplier = np.random.uniform(low=2.0, high=3.0, size=num_promos)

  # Model the promotions on sku 3
  # Equivalent of the Cholesky decomposition of the covariance matrix for two signals
  L = np.array([[1,price_sales_corrcoef_SKU_C_promo], [0,np.sqrt(1-price_sales_corrcoef_SKU_C_promo**2)]])
  # Define two random gaussian signals to produce the correlated ones
  noise = np.random.normal(0.0, 1.0, size=(num_promos, 2))
  correlated_signals = np.matmul(noise, L)
  # Scale the signals to get price and sales
  prices = correlated_signals[:,0]*price_sigma_SKU_C_promo + price_mu_SKU_C_promo
  baseline_sales = correlated_signals[:,1]*sales_sigma_SKU_C_promo + sales_mu_SKU_C_promo

  #sales_multiplier = np.random.uniform(low=2.0, high=3.0, size=num_promos)

  # On promotion, SKU C at SKU A's price range increases the sales
  all_skus_df.loc[idx_promos, 'promos_sku_C'] = True
  all_skus_df.loc[idx_promos, 'baseline_sales_sku_C'] = baseline_sales
  all_skus_df.loc[idx_promos, 'price_sku_C'] = prices
  # Update the volume accordingly
  all_skus_df.loc[idx_promos, 'sales_volume_sku_C'] = \
      all_skus_df.loc[idx_promos, 'baseline_sales_sku_C'] * all_skus_df.loc[idx_promos, 'price_sku_C']

  # Victimise/cannibalise product A.
  #
  # The idea here is that the total volume does not change as 
  # customer's budget is pretty steady, so the increased sales in C
  # effectively shifts SKU A

  # updated_sku_A_volume = Total volume - partial volumes
  partial_volumes = all_skus_df.loc[idx_promos, 'sales_volume_sku_C'] + \
    all_skus_df.loc[idx_promos, 'sales_volume_sku_B']

  updated_sku_A_volume = \
    all_skus_df.loc[idx_promos, 'category_volume'] - partial_volumes
  updated_sku_A_volume[updated_sku_A_volume < 0] = 0

  # Update the volume in the DF
  all_skus_df.loc[idx_promos, 'sales_volume_sku_A'] = updated_sku_A_volume
  # Update the unit sales in the DF
  all_skus_df.loc[idx_promos, 'baseline_sales_sku_A'] = \
    all_skus_df.loc[idx_promos, 'sales_volume_sku_A']/all_skus_df.loc[idx_promos, 'price_sku_A']

  return all_skus_df


def split_datasets(all_skus_df):

  # Split into train, evaluation and test.
  idx_promos_num = np.where(all_skus_df.promos_sku_C.values)[0]
  num_promos = idx_promos_num.shape[0]

  # 25% of the promos
  num_test_set = int(num_promos*0.25)
  idx_test_set_A = idx_promos_num[0:num_test_set]
  # Add the same number of non-promos
  idx_non_promos_num = np.where(all_skus_df.promos_sku_C.values==False)[0]
  idx_test_set_B = idx_non_promos_num[0:num_test_set]

  idx_test_set = np.concatenate([idx_test_set_A, idx_test_set_B])
  df_test = all_skus_df.loc[idx_test_set].copy()
  df_test.reset_index(inplace=True)

  # Remove from the train/eval set
  all_skus_df.drop(idx_test_set, inplace=True)


  # to-do: random split into training and test.
  df_train, df_eval = \
    train_test_split(all_skus_df, \
    test_size=0.15, random_state=42)

  return df_train, df_eval, df_test


def generate_dataset_time(_num_samples = 365):
  '''
    Parameters defined in the script (outside this function)
  '''

  idx_promos = np.zeros(_num_samples, dtype=bool)
  week_number = np.linspace(1, _num_samples, num=_num_samples)
  total_sales_volume = np.zeros(_num_samples)

  all_skus = []
  for idx_store in range(0, len(sku_id)):

    df = atomic_sales(_num_samples, mu_sales[idx_store], sigma_sales[idx_store], 
      baseline_prices_mu[idx_store], baseline_prices_sigma[idx_store], 
      price_sales_corrcoef=price_sales_corrcoef[idx_store],
      noise_strength = 1/6,
      shelf_capacity=None, shelf_impact=None)

    # Just in case
    df['price'] = df['price'].apply(lambda x: np.abs(x))
    
    #sales = noise + randn(mu_sales, sigma_sales) + shelf_capacity*2 + discount*5 
    
    df['sku_id'] = sku_id[idx_store]
    df['promos'] = idx_promos
    df['day_index'] = week_number

    #  Calculate the sales volume
    sales_volume = (df['price']*df['baseline_sales']).values
    df['sales_volume'] = sales_volume
    # Total volume across all SKUs
    total_sales_volume += sales_volume

    df['x1'] = np.random.randint(0, 100, size=(_num_samples))
    df['x2'] = np.random.randint(0, 200, size=(_num_samples))

    # For backwards compatibility
    #df['sales'] = df['baseline_sales']

    col_map = {iCol: f'{iCol}_{sku_id[idx_store]}' for iCol in df.columns.tolist()}
    df.rename(columns=col_map, inplace=True)

    all_skus.append(df)

  all_skus_df = pd.concat(all_skus, axis=1)
  all_skus_df.reset_index(inplace=True)

  # get the total volume of sales per category
  all_skus_df['category_volume'] = total_sales_volume  
  
  # Prepare the promos for SKU C (random)
  '''
  We are going to set 10 promotions, but only 7 of them cannibalise
  Let's add here the 7 that cannibalise
  '''
  #num_promos = int(_num_samples*0.2)
  start_end= [(13,19), (63,70), (103,109), \
              (199,206), (267,274), (323,333), (352,360)]
  idx_promos = np.zeros(_num_samples, dtype=bool)
  for iDuple in start_end:
    idx_promos[iDuple[0]:iDuple[1]]=True
  
  num_promos = idx_promos.sum()
  #assert idx_promos.sum() == num_promos, f'non congruent number of promos {idx_promos.sum()}!={num_promos}'

  # Model the promotions on sku 3
  # Equivalent of the Cholesky decomposition of the covariance matrix for two signals
  L = np.array([[1,price_sales_corrcoef_SKU_C_promo], [0,np.sqrt(1-price_sales_corrcoef_SKU_C_promo**2)]])
  # Define two random gaussian signals to produce the correlated ones
  noise = np.random.normal(0.0, 1.0, size=(num_promos, 2))
  correlated_signals = np.matmul(noise, L)
  # Scale the signals to get price and sales
  prices = correlated_signals[:,0]*price_sigma_SKU_C_promo + price_mu_SKU_C_promo
  baseline_sales = correlated_signals[:,1]*sales_sigma_SKU_C_promo + sales_mu_SKU_C_promo

  #sales_multiplier = np.random.uniform(low=2.0, high=3.0, size=num_promos)

  # On promotion, SKU C at SKU A's price range increases the sales
  all_skus_df.loc[idx_promos, 'promos_sku_C'] = True
  all_skus_df.loc[idx_promos, 'baseline_sales_sku_C'] = baseline_sales
  all_skus_df.loc[idx_promos, 'price_sku_C'] = prices
  # Update the volume accordingly
  all_skus_df.loc[idx_promos, 'sales_volume_sku_C'] = \
      all_skus_df.loc[idx_promos, 'baseline_sales_sku_C'] * all_skus_df.loc[idx_promos, 'price_sku_C']

  # Victimise/cannibalise product A.
  #
  # The idea here is that the total volume does not change as 
  # customer's budget is pretty steady, so the increased sales in C
  # effectively shifts SKU A

  # updated_sku_A_volume = Total volume - partial volumes
  partial_volumes = all_skus_df.loc[idx_promos, 'sales_volume_sku_C'] + \
    all_skus_df.loc[idx_promos, 'sales_volume_sku_B']

  updated_sku_A_volume = \
    all_skus_df.loc[idx_promos, 'category_volume'] - partial_volumes
  updated_sku_A_volume[updated_sku_A_volume < 0] = 0

  # Update the volume in the DF
  all_skus_df.loc[idx_promos, 'sales_volume_sku_A'] = updated_sku_A_volume
  # Update the unit sales in the DF
  all_skus_df.loc[idx_promos, 'baseline_sales_sku_A'] = \
    all_skus_df.loc[idx_promos, 'sales_volume_sku_A']/all_skus_df.loc[idx_promos, 'price_sku_A']
    
  #  
  all_skus_df['promos_sku_C_type'] = 'None'
  all_skus_df.loc[idx_promos, 'promos_sku_C_type'] = 'Price cut'

  # here update the promotions that do not have a cannibalisation effect
  start_end= [(46,51),(117,124), (301,309)]
  for iDuple in start_end:
    idx_promos[iDuple[0]:iDuple[1]]=True
    all_skus_df.loc[iDuple[0]:iDuple[1], 'promos_sku_C_type'] = '3x2'
    all_skus_df.loc[iDuple[0]:iDuple[1], 'baseline_sales_sku_C'] *= 1.35
  all_skus_df.loc[idx_promos, 'promos_sku_C'] = True  

  return all_skus_df


def get_artificial_dataset():
  current_folder = fhelp.get_current_folder()
  dataFolder = fhelp.fullfile(current_folder, os.path.join('data', 'cannibalisation'))

    
  file_train = fhelp.fullfile(dataFolder, 'cannibalisation_' + 'train' + '.pickle')
  file_test = fhelp.fullfile(dataFolder, 'cannibalisation_' + 'test' + '.pickle')
  file_eval = fhelp.fullfile(dataFolder, 'cannibalisation_' + 'eval' + '.pickle')
  
  if not os.path.exists(dataFolder):
    
    fhelp.makeFolder(dataFolder)
    all_skus_df = generate_dataset()
    df_train, df_eval, df_test = split_datasets(all_skus_df)  
    
    fhelp.toPickleFile(df_train, file_train)
    fhelp.toPickleFile(df_test, file_test)
    fhelp.toPickleFile(df_eval, file_eval)

    # Same with Excel
    fhelp.dataFrameToXLSv2(df_train, file_train.replace('pickle', 'xlsx'))
    fhelp.dataFrameToXLSv2(df_test, file_test.replace('pickle', 'xlsx'))
    fhelp.dataFrameToXLSv2(df_eval, file_eval.replace('pickle', 'xlsx'))
    

  else:
    df_train = fhelp.readPickleFile(file_train)
    df_eval = fhelp.readPickleFile(file_eval)
    df_test = fhelp.readPickleFile(file_test)
  
  return df_train, df_eval, df_test




def time_split_datasets(all_skus_df, test_size = 0.15, eval_size = 0.08):

  # Split into train, evaluation and test.
  _num_samples = all_skus_df.shape[0]

  # Test
  test_samples = int(_num_samples*test_size)
  test_start = _num_samples-test_samples-1
  test_end = test_start + test_samples

  df_test = all_skus_df.loc[test_start:test_end].copy()
  df_test.reset_index(inplace=True)

  # Evaluation
  eval_samples = int(_num_samples*eval_size)
  eval_start = test_start - eval_samples-1
  eval_end = eval_start + eval_samples
  eval_samples = int(_num_samples*eval_size)

  df_eval = all_skus_df.loc[eval_start:eval_end].copy()
  df_eval.reset_index(inplace=True)
  
  # Remove from the train/eval set
  df_train = all_skus_df.loc[0:eval_start].copy()
  df_train.reset_index(inplace=True)

  return df_train, df_eval, df_test


def get_artificial_time_series(_num_samples = 365):
  
  #current_folder = fhelp.get_current_folder()
  #dataFolder = fhelp.fullfile(current_folder, os.path.join('data', 'fake_cannibalisation'))
  dataFolder = os.path.expanduser('~/Google Drive/order/Machine Learning Part/data/fake_cannibalisation')
    
  file_train = fhelp.fullfile(dataFolder, 'cannibalisation_' + 'train' + '.pickle')
  file_test = fhelp.fullfile(dataFolder, 'cannibalisation_' + 'test' + '.pickle')
  file_eval = fhelp.fullfile(dataFolder, 'cannibalisation_' + 'eval' + '.pickle')
  
  if not os.path.exists(dataFolder):
    
    fhelp.makeFolder(dataFolder)
    all_skus_df = generate_dataset_time(_num_samples = _num_samples)
    df_train, df_eval, df_test = time_split_datasets(all_skus_df)  
    
    fhelp.toPickleFile(df_train, file_train)
    fhelp.toPickleFile(df_test, file_test)
    fhelp.toPickleFile(df_eval, file_eval)

    # Same with Excel
    fhelp.dataFrameToXLSv2(df_train, file_train.replace('pickle', 'xlsx'))
    fhelp.dataFrameToXLSv2(df_test, file_test.replace('pickle', 'xlsx'))
    fhelp.dataFrameToXLSv2(df_eval, file_eval.replace('pickle', 'xlsx'))
    

  else:
    df_train = fhelp.readPickleFile(file_train)
    df_eval = fhelp.readPickleFile(file_eval)
    df_test = fhelp.readPickleFile(file_test)
  
  return df_train, df_eval, df_test