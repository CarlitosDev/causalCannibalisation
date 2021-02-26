import uuid
import pandas as pd
import os
import pickle
import subprocess
import sys
from datetime import datetime
from shutil import copyfile, move
import uuid
import matplotlib.pyplot as plt
import matplotlib
from os import path as _p
import numpy as np
from statsmodels.tsa.seasonal import STL

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, explained_variance_score
import umap.umap_ as umap
from scipy.stats import skew, kurtosis

def_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']


def writeTextFile(thisStr, thisFile):
    with open(thisFile, 'w') as f:
        f.write(thisStr)

def print_every_n(str_to_print, idx, n):
    if idx % n==0:
        print(str_to_print)

def get_current_folder():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)))


def to_excel_file(df_to_save, xls_filepath, open_file=False):
    '''
      Save to Excel
    '''
    datetimeVars = df_to_save.select_dtypes(
        include=['datetime64[ns, UTC]']).columns.tolist()

    for i_var in datetimeVars:
        df_to_save[i_var] = df_to_save[i_var].astype(str)

    dataFrameToXLSv2(df_to_save, xls_filepath)
    if open_file:
        osOpenFile(xls_filepath)


def to_random_excel_file(df_to_save, writeIndex=False):
    '''
      Only use with the TF analysis data
    '''
    datetimeVars = df_to_save.select_dtypes(
        include=['datetime64[ns, UTC]']).columns.tolist()

    for i_var in datetimeVars:
        df_to_save[i_var] = df_to_save[i_var].astype(str)

    outputFolder = os.path.join(get_current_folder(), 'data', 'xls_to_delete')
    makeFolder(outputFolder)
    xls_filepath = os.path.join(outputFolder, str(uuid.uuid4()) + '.xlsx')
    dataFrameToXLSv2(df_to_save, xls_filepath, writeIndex=writeIndex)
    osOpenFile(xls_filepath)


def dataFrameToXLSv2(df, xlsFile, sheetName='DF',
                     writeIndex=False, float_format='%.2f', freezeTopRow=True,
                     remove_timezone=True):
    '''
    Write DF to Excel centering the columns and freezing the top row
    '''
    if not df.empty:
        xlsWriter = pd.ExcelWriter(xlsFile, engine='xlsxwriter',
                                   options={'remove_timezone': remove_timezone})

        if remove_timezone:
            for i_var in df.select_dtypes(include=['datetime64[ns, UTC]']).columns.tolist():
                df[i_var] = df[i_var].astype(str)

        df.to_excel(xlsWriter, sheetName,
                    index=writeIndex, float_format=float_format)

        # Get the xlsxwriter workbook and worksheet objects.
        workbook = xlsWriter.book
        worksheet = xlsWriter.sheets[sheetName]

        if freezeTopRow:
            worksheet.freeze_panes(1, 0)

        # set the format for the cells
        cell_format = workbook.add_format()
        cell_format.set_align('center')
        cell_format.set_align('vcenter')

        # set the col format (fake Autolimit)
        colNames_lenght = df.columns.str.len().values
        for col in range(0, df.shape[1]):
            maxWidth = 1 + max(colNames_lenght[col],
                               df.iloc[:, col].astype(str).str.len().max())
            worksheet.set_column(col, col, maxWidth, cell_format)

        xlsWriter.save()


def osOpenFile(filePath):
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, filePath])


def makeFolder(thisPath):
    if not os.path.exists(thisPath):
        os.makedirs(thisPath)

def fullfile(foldername, filename):
    return os.path.join(foldername, filename)

def fileparts(thisPath):
    [fPath, fName] = os.path.split(thisPath)
    [file, ext] = os.path.splitext(fName)
    return fPath, file, ext


# Wrap some plotting fuctions into Matlab's style
def figure():
    return plt.figure()


def plot(x=None, y=None):
    '''
    Missing Matlab so much
        x = np.linspace(0,30,26)
        y = np.random.rand(26)
        plot(x, y)
    '''
    plt.plot(x, y)
    plt.show()


def save_plot_to_pfg(plt, pfg_file_name):
    '''
    Save figure for LaTex pfg
        x = np.linspace(0,30,26)
        y = np.random.rand(26)
        plot(x, y)
    '''
    matplotlib.use('pgf')

    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{cmbright}",
        ]
    })
    [_, fName] = os.path.split(pfg_file_name)
    plt.savefig(pfg_file_name)

    instructions = '''\\usepackage{caption}
        \\usepackage{pgf}
        \\usepackage{import}
        ...
        \\begin{figure}
        \\begin{center}
        ''' + '\\input{' + fName + '.pgf}' + '''
        \\end{center}
        \\caption{Made with matplotlib's PGF backend.}
        \\end{figure}
        '''
    print(instructions)


# Pickle helpers
def toPickleFile(data, filePath):
    with open(filePath, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
    print(f'Pickle file saved to {filePath}')

# Binary data, not text


def readPickleFile(filePath):
    with open(filePath, 'rb') as fId:
        pickleData = pickle.load(fId)
    return pickleData

# Write text file


def writeTextFile(thisStr, thisFile):
    with open(thisFile, 'w') as f:
        f.write(thisStr)


def prepareTableLaTeX(df_latex):
    '''
        Prepare the table for the paper
    '''
    latex_prefix = '''\\begin{table*}[t]
    \centering
    \\resizebox{\\textwidth}{!}{'''

    latex_postfix = '''}
    \\caption{Results for}
    \\label{tab:real_forecast_}
    \\end{table*}
    '''
    str_latex = df_latex.to_latex(index=False, float_format='{:3.2f}'.format)
    # replace the formulas
    str_latex = str_latex.replace('0.0', '')
    str_latex = str_latex.replace('y\_train', '''$\mathbf{y}$ (sales)''')
    str_latex = str_latex.replace(
        'delta\_y\_train', '''$\mathbf{y_{\Delta}}$''')
    str_latex = str_latex.replace(
        'y\_train\_plus\_delta', '''$\mathbf{y + y_{\Delta}}$''')
    str_latex = str_latex.replace('nan', '')
    # Add variable importance to the text
    str_latex = latex_prefix + str_latex + latex_postfix

    return str_latex


def promotions_reader(baseFolder, baseFile):
    dataFile = _p.expanduser(_p.join(baseFolder, baseFile))
    df_eng_all = pd.read_pickle(dataFile)
    return df_eng_all


def describe_bpns_from_file(baseFolder, baseFile, bpns):
  df_eng_all = BPNS_reader(baseFolder, baseFile, bpns)
  idx_product = df_eng_all.base_product_number_std == bpns
  print(df_eng_all[idx_product].iloc[0])

def describe_bpns(df_eng_all, bpns):
  idx_product = df_eng_all.base_product_number_std == bpns
  print(df_eng_all[idx_product].iloc[0])

def BPNS_reader(baseFolder, baseFile, bpns):
    '''
        Speed things up
        Look for the file with the BPNS. If it does not exist,
        filter by the PSGC and save the file.
    '''
    bpns_filepath = _p.join(get_current_folder(), 'data', bpns + '.pickle')
    if os.path.exists(bpns_filepath):
        df_eng_all = pd.read_pickle(bpns_filepath)
    else:
        dataFile = _p.expanduser(_p.join(baseFolder, baseFile))
        df_temp = pd.read_pickle(dataFile)
        idx_product = df_temp.base_product_number_std == bpns
        this_psgc = df_temp.loc[idx_product, 'product_sub_group_code'].iloc[0]
        idx_psgc = df_temp.product_sub_group_code.str.contains(this_psgc)
        df_eng_all = df_temp[idx_psgc].copy()
        print(f'Saving file {bpns_filepath}...')
        toPickleFile(df_eng_all, bpns_filepath)

    return df_eng_all

def frc_with_random_neighbours(X_train, X_test, num_neighbours, contrastiveReg):
    '''
        Benchmark the Euclidean search vs just random search
    '''

    trainingSize = X_train.shape[0]
    testSize = X_test.shape[0]

    y_k_all_list = []
    y_k_list = []
    y_k_weighted_list = []
    y_delta_list = []

    for idx_test in range(0, testSize):

        idxSorted = np.random.choice(
            trainingSize-1, num_neighbours, replace=False)
        x_A = X_train[idxSorted]
        x_B = np.tile(X_test[idx_test], (num_neighbours, 1))
        X_AB_test = np.concatenate([x_A, x_B], axis=1)

        # differences regarding the reference promotions
        xgb_frc = contrastiveReg.regressor.predict(X_AB_test)

        # Get the average
        y_delta_list.append(xgb_frc)
        y_k_hat_all = xgb_frc + contrastiveReg.y_train[idxSorted]
        y_k_hat = np.mean(y_k_hat_all)

        # overwrite for compatibility
        y_k_hat_distances = y_k_hat
        y_k_weighted_list.append(y_k_hat_distances)

        # Append to the list
        y_k_all_list.append(y_k_hat_all)
        y_k_list.append(y_k_hat)

    # Arrange the forecast as np-arrays
    y_hat_random = np.array(y_k_list)

    return y_hat_random


def frc_plain_CatBoost(num_neighbours, validation_test_size,
                       num_iterations, learning_rate, depth, X_train, y_train, X_test):

    # CatBoost
    cb_model = CatBoostRegressor(iterations=num_iterations, learning_rate=learning_rate,
                                 depth=depth, loss_function='RMSE', cat_features=None, silent=False)

    # Split into training and evaluation
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = \
        train_test_split(X_train, y_train, test_size=validation_test_size)

    #
    eval_set = [(X_val_xgb, y_val_xgb)]

    print(f'')
    cb_model.fit(X_train_xgb, y_train_xgb, eval_set=eval_set,
                 verbose=50)  # , logging_level='Info')

    # differences regarding the reference promotions
    cb_frc = cb_model.predict(X_test)
    return cb_frc, cb_model



def frc_AutoGluon(df_train, df_test, 
    categoricalVars, responseVar = 'wk1_sales_all_stores'):
    
    import autogluon as ag
    from autogluon import TabularPrediction as task

    for varName in categoricalVars:
        df_train[varName] = df_train[varName].astype(str)
        df_test[varName] = df_test[varName].astype(str)

    # AutoGluon format
    train_data = task.Dataset(df=df_train)
    test_data = task.Dataset(df=df_test)

    model = task.fit(train_data=train_data, 
    output_directory="auto_gluon", label=responseVar,
    hyperparameter_tune=False)


    # Forecast with the best model
    autogluon_frc = model.predict(test_data)
    return {'autoGluon_frc': autogluon_frc, 'autoGluon_model':model}

def get_frc_errors(y, y_hat, verbose=True):
      '''
      Get forecast residuals as e_t = \hat{y} - y
          so e_t > 0 overforecast
          so e_t < 0 underforecast
          e_t = 0 : the dream
      '''
      var_explained = explained_variance_score(y, y_hat)
      e_t = y - y_hat
      abs_e_t = np.abs(e_t)
      
      frc_error = np.sum(abs_e_t)/np.sum(y)
      frc_bias = np.sum(e_t)/np.sum(y_hat)
      frc_acc  = 1.0 - frc_bias
      
      MAE = abs_e_t.mean()
      MSE = np.power(e_t, 2).mean()
      meanError = e_t.mean()
      MAPE = 100*(abs_e_t/np.abs(y)).mean()
      r2 = r2_score(y, y_hat)

      d = {'MAE': MAE,
      'MSE': MSE, 
      'RMSE': np.sqrt(MSE), 
      'meanError': meanError,
      'MAPE': MAPE,
      'R2': r2,
      'frc_error': frc_error,
      'frc_bias': frc_bias,
      'frc_acc': frc_acc,
      'Var explained': var_explained,
      'mu_y': y.mean(),
      'mu_y_hat': y_hat.mean(),
      'sigma_y': y.std(),
      'sigma_y_hat': y_hat.std()}
      #,'residuals': e_t}
      if verbose:
        for k,v in d.items():
          print(f'{k}: {v:3.2f}')
      return d


def train_CatBoost(num_iterations, learning_rate, \
                   depth, idx_categorical_features, \
                   validation_test_size, X_train, y_trainloss_function, loss_function='RSME'):
    # CatBoost
    cb_model = CatBoostRegressor(iterations=num_iterations, learning_rate=learning_rate, \
                                 depth=depth, loss_function=loss_function, cat_features=None, silent=False)

    # Split into training and evaluation
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = \
        train_test_split(X_train, y_train, test_size=validation_test_size)

    eval_set = [(X_val_xgb, y_val_xgb)]
    cb_model.fit(X_train_xgb, y_train_xgb, eval_set=eval_set, verbose=50)

    return cb_model


def get_taxonomy_from_sku_name(sku_name):
    sku_chunks = sku_name.split('-')
    sku_id = sku_chunks[1]
    sku_chunk_B = sku_chunks[1].split('_')
    store_name = sku_chunks[-1]
    category_id = sku_chunk_B[0]
    dept_id = sku_chunk_B[0] + '_' + sku_chunk_B[1]
    return category_id, dept_id, sku_id, store_name


# Season-Trend decomposition using LOESS.
def decompose_signal(input_signal, period_in_days=14, minimum_heartbeat=0.85):
    sales_decomposition_LOESS = STL(input_signal, period=period_in_days).fit()
    seasonality_flag = sales_decomposition_LOESS.trend > minimum_heartbeat
    return pd.DataFrame({'heartbeat_flag': seasonality_flag,
            'trend': sales_decomposition_LOESS.trend,
            'seasonal': sales_decomposition_LOESS.seasonal,
            'residual': sales_decomposition_LOESS.resid})

# Get an indicator of the decomposition
def measure_strength_decomposition(df_decomposition: 'DF resulted from decomposing with decompose_signal'):
    '''
        Get an idea of the strength of the components by measuring their relative variances
        The lower the indicator is, the more similar they are - so not a great decomposition
    '''
    var_residual = df_decomposition.residual.var()
    var_residual_seasonal = df_decomposition[['residual','seasonal']].sum(axis=1).var()
    var_residual_trend = df_decomposition[['residual','trend']].sum(axis=1).var()
    var_residual_trend, var_residual_seasonal, var_residual
    f_seasonal = np.max([0, 1-(var_residual/var_residual_seasonal)])
    f_trend = np.max([0, 1-(var_residual/var_residual_trend)])
    return f_trend, f_seasonal

def get_dataFolder():
    return os.path.expanduser('~/Google Drive/order/Machine Learning Part/data/Walmart(M5)')


def plot_sales_price_pairs(df_store, sku_A, sku_B, MI=None, save_to_file=True, fig_h = 10, fig_w = 18, plot_snap_days=True, snap_varname=None):
    
    def_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    category_id, dept_id, sku_id, store_name = get_taxonomy_from_sku_name(sku_A)
    store_state = store_name.split('_')[0]

    sku_A_price_var = sku_A.replace('sales', 'price')
    sku_B_price_var = sku_B.replace('sales', 'price')

    fig, ax = plt.subplots(figsize=(fig_w*1.5, fig_h/1.25), nrows=2, ncols=1)


    x_axis = df_store.date
    ax[0].plot(x_axis, df_store[sku_A], label=f'Sales {sku_A} (cannibal)', 
            color=def_colours[0], linewidth=2.5, alpha=0.75)

    ax[0].plot(x_axis, df_store[sku_B], label=f'Sales {sku_B}', 
            color=def_colours[3], linewidth=2.5, alpha=0.75)

    ax[0].legend()
    ax[0].set_xlabel('dates')
    ax[0].set_ylabel('Store sales')
    ax[0].grid(True)
    ax[0].set_title(f'Sales of {sku_A} vs sales of {sku_B} (Approx MI {MI})')


    ax[1].plot(x_axis, df_store[sku_A_price_var], label=f'Sales {sku_A}', 
            color=def_colours[0], linewidth=2.5, alpha=0.75)

    ax[1].plot(x_axis, df_store[sku_B_price_var], label=f'Sales {sku_B}', 
            color=def_colours[3], linewidth=2.5, alpha=0.75)

    if plot_snap_days:
        if snap_varname:
            idx_label_B = df_store[snap_varname]
        else:
            idx_label_B = df_store[f'snap_{store_state}']
        ax[0].plot(x_axis[idx_label_B], df_store[sku_B][idx_label_B], '*', label=f'Snap days', 
            color=def_colours[-2], linewidth=2.5, alpha=0.95)

    ax[1].set_xlabel('dates')
    ax[1].set_ylabel('SKU price')
    ax[1].grid(True)

    fig.tight_layout()
    if save_to_file:
        foldername_png = os.path.join(get_dataFolder(), 'sku_pair_analysis', category_id, dept_id, store_name)
        makeFolder(foldername_png)
        plt_filename = os.path.join(foldername_png, f'{store_name}-{sku_A}-{sku_B}.png')
        plt.savefig(plt_filename, format='png')
        plt.close()
        
def simple_XY_plot(x_axis, y_data, fig_h = 10, fig_w = 18):
    def_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Plot one store
    f, ax = plt.subplots(1,1,figsize=(fig_w*1.5, fig_h/1.5))
    ax.plot(x_axis, y_data, color=def_colours[3], linewidth=2, alpha=0.75)
    #plt.legend()
    plt.xlabel('dates')
    plt.grid(True)
    plt.show()
    
        
def plot_LOESS_decomposition(df_A, sku_A, fig_h = 10, fig_w = 18, is_CFAV=False):
        
    if is_CFAV:
        category_id, dept_id, sku_id, store_name = get_taxonomy_from_sku_name_CFAV(sku_A)
    else:
        category_id, dept_id, sku_id, store_name = get_taxonomy_from_sku_name(sku_A)

    def_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_axis = df_A.date
    
    idx_store = 0
    # Plot one store
    f, ax = plt.subplots(1,1,figsize=(fig_w*1.5, fig_h/1.5))

    idx_axis = 0
    ax.plot(x_axis, df_A['sales-' + f'{sku_id}-{store_name}'], label=f'Sales {store_name}', 
            color=def_colours[idx_store], linewidth=2, alpha=0.75)

    ax.plot(x_axis, df_A['trend-' + f'{sku_id}-{store_name}'], label=f'LOESS trend', 
            color=def_colours[idx_store+1], linewidth=1.5, alpha=0.95)

    ax.plot(x_axis, df_A['seasonal-' + f'{sku_id}-{store_name}'], label=f'Seasonal', 
            color=def_colours[idx_store+2], linewidth=1.0, alpha=0.85)


    ax.plot(x_axis, df_A['heartbeat_flag-' + f'{sku_id}-{store_name}'], label=f'Availability', 
            color=def_colours[idx_store+3], linewidth=2.5, alpha=0.95)

    plt.legend()
    plt.xlabel('dates')
    plt.ylabel(f'Store sales for {sku_A} ')
    plt.grid(True)
    plt.show()
    
    
def add_snap_groups(df_store, snap_varname, shift_size = 5):
    '''
        From a DF with a SNAP days binary variable,
        get the SNAP periods as groups and the previous and post 
        5 days (SNAP is 10 days)
    '''
    a = df_store[snap_varname]
    b = (a.shift(1)!=a).cumsum()
    b[~a] = 0
    val_mapper = {value:idx for idx, value in enumerate(b.unique())}
    snap_groups = b.map(val_mapper)
    pre_post_snap_groups = snap_groups.shift(5, fill_value=0)+snap_groups.shift(-5, fill_value=0)
    pre_post_snap_groups[a] = 0
    df_store[snap_groups]= snap_groups
    df_store[pre_post_snap_groups]= pre_post_snap_groups
    return df_store


def get_snap_groups(snap_data, shift_size = 5):
    '''
        From a DF with a SNAP days binary variable,
        get the SNAP periods as groups and the previous and post 
        5 days (SNAP is 10 days)
    '''
    a = snap_data.copy()
    b = (a.shift(1)!=a).cumsum()
    b[~a] = 0
    val_mapper = {value:idx for idx, value in enumerate(b.unique())}
    snap_groups = b.map(val_mapper)
    pre_post_snap_groups = snap_groups.shift(shift_size, fill_value=0)+snap_groups.shift(-shift_size, fill_value=0)
    pre_post_snap_groups[a] = 0
    return snap_groups, pre_post_snap_groups


def cfav_get_map_dept_to_cat():
    a = {'AUTOMOTIVE': 'AUTOMOTIVE',
         'BABY CARE': 'HEALTH_BEAUTY',
         'BEAUTY': 'HEALTH_BEAUTY',
          'BEVERAGES': 'DRINKS',
          'BOOKS': 'HOBBIES',
          'SCHOOL AND OFFICE SUPPLIES': 'SCHOOL',
          'CELEBRATION': 'CELEBRATION',
          'HARDWARE': 'HARDWARE',
          'HOME AND KITCHEN I': 'HOME',
          'HOME AND KITCHEN II': 'HOME',
          'HOME APPLIANCES': 'HOME',
          'MAGAZINES': 'HOBBIES',
          'PLAYERS AND ELECTRONICS': 'HOBBIES',
          'CLEANING': 'HOUSEHOLD',
          'DAIRY': 'FOOD',
          'DELI': 'FOOD',
          'EGGS': 'FOOD',
          'FROZEN+FOODS': 'FOOD',
          'GROCERY+I': 'GROCERY',
          'GROCERY+II': 'GROCERY',
          'HOME CARE': 'HOUSEHOLD',
          'LADIESWEAR': 'CLOTHES',
          'LAWN AND GARDEN': 'GARDEN',
          'LINGERIE': 'CLOTHES',
          'LIQUOR,WINE,BEER': 'DRINKS',
          'MEATS': 'FOOD',
          'PERSONAL+CARE': 'HEALTH_BEAUTY',
          'PET+SUPPLIES': 'PETS',
          'POULTRY': 'FOOD',
          'PREPARED+FOODS': 'FOOD',
          'PRODUCE': 'FOOD',
          'SEAFOOD': 'FOOD',
          'BREAD+BAKERY': 'FOOD'}
    new_entries = {}
    for k, v in a.items():
         if (' ' in k) or ('+' in k):
            new_entries.update({k.replace(' ', '+'): v})
            new_entries.update({k.replace(' ', '_'): v})
            new_entries.update({k.replace('+', '_'): v})
    a.update(new_entries)
    return a


def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric)
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], s=80)
    plt.title(title, fontsize=18)
    return u


def get_taxonomy_from_sku_name_CFAV(sku_name):
    '''
        Work out the SKU taxonomy for CFAV products
    '''
    sku_chunks = sku_name.split('-')
    if len(sku_chunks) == 2:
        sku_id, store_name = sku_chunks[0], sku_chunks[1]
    elif len(sku_chunks) == 3:
        sku_id, store_name = sku_chunks[1], sku_chunks[2]
    else:
        sku_id, store_name = 'none', 'none'

    # Get the department
    sku_id_chunks = sku_id.split('_')
    if len(sku_id_chunks)==3:
        dept_id = f'{sku_id_chunks[0]}_{sku_id_chunks[1]}'
    else:
        dept_id = sku_id_chunks[0]

    # Get the category


    category_id = cfav_get_map_dept_to_cat().get(dept_id, str(uuid.uuid4()).split('-')[0])
    category_id, dept_id, sku_id, store_name
    return category_id, dept_id, sku_id, store_name


def get_basic_stats(seasonal_signal: 'np 1D array'):
    # excess kurtosis (already substracted 3)
    
    signal_max = seasonal_signal.max()
    signal_min = seasonal_signal.min()
    
    return {'mean':seasonal_signal.mean(),
    'std': seasonal_signal.std(),
    'skewness':skew(seasonal_signal),
    'eKurtosis':kurtosis(seasonal_signal),
    'max':signal_max,
    'min':signal_min,
    'vpp':signal_max-signal_min}




def plot_two_skus(df_store, sku_A, sku_B, sku_B_reg, sku_B_cannibalised, fig_h = 10, fig_w = 18, save_to_file=True):
    
    
    category_id, dept_id, sku_id_A, store_name = get_taxonomy_from_sku_name_CFAV(sku_A)
    _, _, sku_id_B, _ = get_taxonomy_from_sku_name_CFAV(sku_B)

    x_axis = df_store.date

    idx_store = 0
    # Plot one store
    f, ax = plt.subplots(1,1,figsize=(fig_w*1.5, fig_h/1.5))

    idx_axis = 0
    ax.plot(x_axis, df_store[f'sales-{sku_id_A}-{store_name}'], label=f'Sales CN {sku_id_A}-{store_name}',
            color=def_colours[idx_store], linewidth=2, alpha=0.75)

    ax.plot(x_axis, df_store[f'sales-{sku_id_B}-{store_name}'], label=f'Sales VC {sku_id_B}-{store_name} (reg={sku_B_reg:3.2f})',
            color=def_colours[idx_store+1], linewidth=2, alpha=0.95)


    promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
    ax.plot(x_axis[promo_sku_A], df_store[sku_A][promo_sku_A], '.', label=f'Promo days {sku_id_A} (can={sku_B_cannibalised:3.2f})', 
                color='g', linewidth=2.5, alpha=0.85)

    promo_sku_B = df_store[f'promotion_flag-{sku_id_B}-{store_name}']
    ax.plot(x_axis[promo_sku_B], df_store[sku_B][promo_sku_B], '*', label=f'Promo days {sku_id_B}', 
                color=def_colours[-4], linewidth=2.5, alpha=0.95)


    plt.legend()
    plt.xlabel('dates')
    plt.ylabel(f'Cannibalisation analysis')
    plt.grid(True)
    
    if save_to_file:
        foldername_png = os.path.join(dataFolder, results_folder, category_id, dept_id, 'plots', store_name)
        fhelp.makeFolder(foldername_png)
        plt_filename = os.path.join(foldername_png, f'{sku_id_A}-{sku_id_B}.png')
        plt.savefig(plt_filename, format='png')
        plt.close()
    else:
        plt.show()
        
        
        
def split_promos_into_sequences(idx_promos: 'pd.Series', min_promo_days=4, min_regular_days=6):
    '''
    Group the indices of a promotion into sequences of pre and post promotion
    '''
    # Groups/sequences
    seqs = (idx_promos.shift(1)!=idx_promos).cumsum()
    promo_seqs = seqs[idx_promos]
    # Indices
    idx_pre_intervention = []
    idx_post_intervention = []
    for value_promo_seqs in promo_seqs.unique():
        idx_current_promo = seqs==value_promo_seqs
        prev_seq = value_promo_seqs-1
        idx_current_regular = seqs==prev_seq
        current_promo_length = idx_current_promo.sum()
        current_regular_length = idx_current_regular.sum()
        if (current_promo_length >= min_promo_days) and (current_regular_length >= min_regular_days):
            idx_pre_intervention.append(idx_current_regular)
            idx_post_intervention.append(idx_current_promo)
    return idx_pre_intervention, idx_post_intervention


def plot_causal_pairs(irow, df_store, fig_h = 10, fig_w = 18, folder_to_save_plots=None, save_to_file=True):

    sku_id_A = irow['cannibal']
    sku_id_B = irow['victim']
    
    # get the taxonomy
    sku_A = df_store.filter(regex=f'sales-{sku_id_A}').columns[0]
    category_id, dept_id, _, store_name = get_taxonomy_from_sku_name_CFAV(sku_A)

    start_period = irow.idx_promo_days[0]
    end_period = irow.idx_promo_days[1]+1

    sku_B_reg = irow['sku_B_regular_avg_sales']
    sku_B_cannibalised = irow['sku_B_avg_sales_during_promo_sku_A']
    sku_B_predicted = irow['avg_predicted']
    
    slot_number = irow['slot_number']

    x_axis = df_store.date

    idx_store = 0
    # Plot one store
    f, ax = plt.subplots(1,1,figsize=(fig_w*1.5, fig_h/1.5))

    idx_axis = 0
    sales_sku_A = df_store[f'sales-{sku_id_A}-{store_name}']
    ax.plot(x_axis, sales_sku_A, label=f'Sales CN {sku_id_A}-{store_name}',
            color=def_colours[idx_store], linewidth=2, alpha=0.65)

    sales_sku_B = df_store[f'sales-{sku_id_B}-{store_name}']
    ax.plot(x_axis, sales_sku_B, label=f'Sales VC {sku_id_B}-{store_name} (reg={sku_B_reg:3.2f})',
            color=def_colours[idx_store+1], linewidth=2, alpha=0.65)

    ax.plot(x_axis.iloc[start_period:end_period], sales_sku_A.iloc[start_period:end_period],
            color=def_colours[idx_store], linewidth=3, alpha=0.95)

    ax.plot(x_axis.iloc[start_period:end_period], sales_sku_B.iloc[start_period:end_period],
            color=def_colours[idx_store+1], linewidth=3, alpha=0.95)

    promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
    ax.plot(x_axis[promo_sku_A], sales_sku_A[promo_sku_A], '.', label=f'Promo days {sku_id_A} (can={sku_B_cannibalised:3.2f}, pred={sku_B_predicted:3.2f})', 
                color='g', linewidth=3.5, alpha=0.85)


    promo_sku_B = df_store[f'promotion_flag-{sku_id_B}-{store_name}']
    ax.plot(x_axis[promo_sku_B], sales_sku_B[promo_sku_B], '*', label=f'Promo days {sku_id_B}', 
                color=def_colours[-4], linewidth=3.5, alpha=0.95)

    plt.legend()
    plt.xlabel('dates')
    plt.ylabel(f'Cannibalisation analysis')
    plt.grid(True)
    
    if save_to_file:
        foldername_png = os.path.join(folder_to_save_plots, category_id, dept_id, 'causal_plots', store_name)
        makeFolder(foldername_png)
        plt_filename = os.path.join(foldername_png, f'{sku_id_A}-{sku_id_B}-{slot_number}.png')
        plt.savefig(plt_filename, format='png')
        plt.close()
    else:
        plt.show()
        
        
def plot_causal_pairs_exogenous(irow, df_store, fig_h = 10, fig_w = 18, folder_to_save_plots=None, 
                                save_to_file=True, use_trend_total_sales=True):

    sku_id_A = irow['cannibal']
    sku_id_B = irow['victim']
    
    # get the taxonomy
    sku_A = df_store.filter(regex=f'sales-{sku_id_A}').columns[0]
    category_id, dept_id, _, store_name = get_taxonomy_from_sku_name_CFAV(sku_A)

    start_period = irow.idx_promo_days[0]
    end_period = irow.idx_promo_days[1]+1

    sku_B_reg = irow['sku_B_regular_avg_sales']
    sku_B_cannibalised = irow['sku_B_avg_sales_during_promo_sku_A']
    sku_B_predicted = irow['avg_predicted']
    
    slot_number = irow['slot_number']

    x_axis = df_store.date

    idx_store = 0
    # Plot one store
    f, ax = plt.subplots(2,1,figsize=(fig_w*1.5, fig_h))

    idx_axis = 0
    sales_sku_A = df_store[f'sales-{sku_id_A}-{store_name}']
    ax[0].plot(x_axis, sales_sku_A, label=f'Sales CN {sku_id_A}-{store_name}',
            color=def_colours[idx_store], linewidth=2, alpha=0.65)

    sales_sku_B = df_store[f'sales-{sku_id_B}-{store_name}']
    ax[0].plot(x_axis, sales_sku_B, label=f'Sales VC {sku_id_B}-{store_name} (reg={sku_B_reg:3.2f})',
            color=def_colours[idx_store+1], linewidth=2, alpha=0.65)

    ax[0].plot(x_axis.iloc[start_period:end_period], sales_sku_A.iloc[start_period:end_period],
            color=def_colours[idx_store], linewidth=3, alpha=0.95)

    ax[0].plot(x_axis.iloc[start_period:end_period], sales_sku_B.iloc[start_period:end_period],
            color=def_colours[idx_store+1], linewidth=3, alpha=0.95)

    ax[0].axvspan(x_axis.iloc[start_period], x_axis.iloc[end_period], alpha=0.1, color='red')

    promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
    ax[0].plot(x_axis[promo_sku_A], sales_sku_A[promo_sku_A], '.', label=f'Promo days {sku_id_A} (can={sku_B_cannibalised:3.2f}, pred={sku_B_predicted:3.2f})', 
                color='g', linewidth=3.5, alpha=0.85)


    promo_sku_B = df_store[f'promotion_flag-{sku_id_B}-{store_name}']
    ax[0].plot(x_axis[promo_sku_B], sales_sku_B[promo_sku_B], '*', label=f'Promo days {sku_id_B}', 
                color=def_colours[-4], linewidth=3.5, alpha=0.95)
    
    ax[0].legend()
    ax[0].set_xlabel('dates')
    ax[0].set_ylabel('Cannibalisation analysis')
    ax[0].grid(True)
    ax[0].margins(0,0)
    
    
    # Add the exogenous data
    present_var = 'total_units_trend' in df_store.columns.tolist()
    if use_trend_total_sales & present_var:
        total_units_signal = df_store['total_units_trend']
        ax[1].plot(x_axis, df_store['total_units_trend'], label=f'Trend total sales CN {dept_id}-{store_name}',
                color=def_colours[idx_store], linewidth=2, alpha=0.85)
        ax[1].plot(x_axis, df_store['total_units'], color=def_colours[idx_store], linewidth=1, alpha=0.35)
    else:
        total_units_signal = df_store['total_units']
        ax[1].plot(x_axis, total_units_signal, label=f'Total sales CN {dept_id}-{store_name}',
                color=def_colours[idx_store], linewidth=2, alpha=0.85)
    


    ax2 = ax[1].twinx()

    ax2.plot(x_axis, df_store['T2M_MAX_adj'], label=f'Avg day temperature (C) for {store_name}',
            color='g', linewidth=2, alpha=0.45)


    lines, labels = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax[1].set_xlabel('dates')
    ax[1].set_ylabel('Exogenous variables')
    ax[1].grid(True)
    ax[1].margins(0,0)
    if save_to_file:
        print(category_id, dept_id)
        foldername_png = os.path.join(folder_to_save_plots, category_id, dept_id, 'causal_plots', store_name)
        makeFolder(foldername_png)
        plt_filename = os.path.join(foldername_png, f'{sku_id_A}-{sku_id_B}-{slot_number}.png')
        plt.savefig(plt_filename, format='png')
        plt.close()
    else:
        plt.show()


def plot_causal_pairs_with_price(irow, df_store, df_transactions_at_store, fig_h = 10, fig_w = 18, folder_to_save_plots=None, 
                                save_to_file=True, use_trend_total_sales=True, save_as_pdf=False):

    sku_id_A = irow['cannibal']
    sku_id_B = irow['victim']
    
    # get the taxonomy
    sku_A = df_store.filter(regex=f'sales-{sku_id_A}').columns[0]
    _, dept_id, _, store_name = get_taxonomy_from_sku_name_CFAV(sku_A)
    category_id = dunnhumby_get_map_dept_to_cat()[dept_id]

    start_period = irow.idx_promo_days[0]
    end_period = irow.idx_promo_days[1]+1

    sku_B_reg = irow['sku_B_regular_avg_sales']
    sku_B_cannibalised = irow['sku_B_avg_sales_during_promo_sku_A']
    sku_B_predicted = irow['avg_predicted']
    
    slot_number = irow['slot_number']

    x_axis = df_store.date

    idx_store = 0
    # Plot one store
    f, ax = plt.subplots(2,1,figsize=(fig_w*1.5, fig_h))

    idx_axis = 0
    sales_sku_A = df_store[f'sales-{sku_id_A}-{store_name}']
    ax[0].plot(x_axis, sales_sku_A, label=f'Sales CN {sku_id_A}-{store_name}',
            color=def_colours[idx_store], linewidth=3, alpha=0.65)

    sales_sku_B = df_store[f'sales-{sku_id_B}-{store_name}']
    ax[0].plot(x_axis, sales_sku_B, label=f'Sales VC {sku_id_B}-{store_name} (reg={sku_B_reg:3.2f})',
            color=def_colours[idx_store+1], linewidth=3, alpha=0.65)

    ax[0].plot(x_axis.iloc[start_period:end_period], sales_sku_A.iloc[start_period:end_period],
            color=def_colours[idx_store], linewidth=4, alpha=0.95)

    ax[0].plot(x_axis.iloc[start_period:end_period], sales_sku_B.iloc[start_period:end_period],
            color=def_colours[idx_store+1], linewidth=4, alpha=0.95)

    ax[0].axvspan(x_axis.iloc[start_period], x_axis.iloc[end_period], alpha=0.1, color='red')

    promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
    ax[0].plot(x_axis[promo_sku_A], sales_sku_A[promo_sku_A], 'o', label=f'Promo days {sku_id_A} (can={sku_B_cannibalised:3.2f}, pred={sku_B_predicted:3.2f})', 
                color='r', linewidth=5.5, alpha=0.95)


    promo_sku_B = df_store[f'promotion_flag-{sku_id_B}-{store_name}']
    ax[0].plot(x_axis[promo_sku_B], sales_sku_B[promo_sku_B], 'o', label=f'Promo days {sku_id_B}', 
                color=def_colours[-4], linewidth=5.5, alpha=0.95)
    
    ax[0].legend(prop={'size': 16})
    ax[0].set_xlabel('dates', fontsize=16)
    ax[0].set_ylabel('Cannibalisation analysis', fontsize=16)
    ax[0].grid(True)
    ax[0].margins(0,0.05)
    
    
    # Add the cannibal price
    idx_valid = df_transactions_at_store.date.isin(x_axis) & \
        (df_transactions_at_store.item_id == int(sku_id_A.split('_')[-1]))

    price_cannibal = df_transactions_at_store[idx_valid]
    ax[1].plot(price_cannibal.date, price_cannibal.PRICE, label=f'Cannibal price',
            color=def_colours[idx_store], linewidth=3, alpha=0.65)
    


    ax2 = ax[1].twinx()

    idx_valid_B = df_transactions_at_store.date.isin(x_axis) & \
        (df_transactions_at_store.item_id == int(sku_id_B.split('_')[-1]))
    
    price_victim = df_transactions_at_store[idx_valid_B]
    
    ax2.plot(price_victim.date, price_victim.PRICE, label=f'Victim price',
            color=def_colours[idx_store+1], linewidth=3, alpha=0.65)


    lines, labels = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc=0, prop={'size': 18})
    ax2.legend(lines + lines2, labels + labels2, loc=1, prop={'size': 16})

    ax[1].set_xlabel('dates', fontsize=16)
    ax[1].set_ylabel('Price', fontsize=16)
    ax[1].grid(True)
    ax[1].margins(0,0.05)
    f.tight_layout()
    if save_to_file:
        print(category_id, dept_id)
        foldername_png = os.path.join(folder_to_save_plots, category_id, dept_id, 'causal_plots', store_name)
        makeFolder(foldername_png)
        if save_as_pdf:
            plt_filename = os.path.join(foldername_png, f'{sku_id_A}-{sku_id_B}-{slot_number}.pdf')
            plt.savefig(plt_filename)
        else:
            plt_filename = os.path.join(foldername_png, f'{sku_id_A}-{sku_id_B}-{slot_number}.png')
            plt.savefig(plt_filename, format='png')
        plt.close()
    else:
        plt.show()
        
        
def dunnhumby_get_map_dept_to_cat():
    return {'PRETZELS': 'BAG.SNACKS', 'ADULT.CEREAL': 'COLD.CEREAL', 'ALL.FAMILY.CEREAL': 'COLD.CEREAL', 
            'KIDS.CEREAL': 'COLD.CEREAL', 'PIZZA.PREMIUM': 'FROZEN.PIZZA', 
            'MOUTHWASHES.(ANTISEPTIC)': 'ORAL.HYGIENE.PRODUCTS', 
            'MOUTHWASH.RINSES.AND.SPRAYS': 'ORAL.HYGIENE.PRODUCTS'}