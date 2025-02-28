import pandas as pd
import ast
pd.options.mode.chained_assignment = None  # default='warn'
import math
import numpy as np
import yfinance as yf
import requests
import csv
import os
import time
import matplotlib.pyplot as plt
from numpy.random import default_rng

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier


global tickers
tickers = ['AAPL','ABT','AJG','AVB','BDX','BIIB','BXP','CPRT','EMR','ERIE', 'EXC','FRT','HIG','HWM','IBM','IP','MKC','MSFT','NTRS','NXPI','OMC','PAYX','PEP','PCG','PNW','SYK','TMUS','GWW','WDC','WMB']



#------------------------Data Creation---------------------------------------
def create_stock_data(stock_tickers):    
    data_dct = {}
    for ticker in stock_tickers:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period = '15y')
        #stock_data['Date'] = stock_data['Date'].str[:10]
        
        data_dct[ticker] = stock_data
    return data_dct

#data_dct = create_stock_data(tickers)

def save_stock_data(data_dct, folder = 'Stock Data'):
    folder_path = os.path.join(os.getcwd(), folder)
    
    # Make folder if it does not exist
    os.makedirs(folder_path, exist_ok = True)
    
    for stock, df in data_dct.items():
        file_path = os.path.join(folder_path, f"{stock}_data.csv")
        df.to_csv(file_path, index= True)
    
    
    # muted the print when testing
    #print(f"All csv files saved in: {folder_path}")

#save_stock_data(data_dct, folder='Stock Data')



###-------------------------Model Creation----------------------------- 
def load_stock_data(stock, folder = 'Stock Data'):
    file_path = os.path.join(os.getcwd(), folder, f"{stock}_data.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
                            
        # muted the print when testing
        #print(f"Loaded {stock}_data.csv successfully.")
        df['Date'] = df['Date'].str[:10]

        return df
    else:
        print(f"{stock}_data.csv not found in {folder}")
        return None
    


#---------------------Feature Creation---------------------
 #  Feature Creation
 # -Moving Averages (MA): 50-day, 200-day
 # -Relative Strength Index (RSI)
 # -Moving Average Convergence Divergence (MACD)
 # -Boilinger Bands
 # -Moving average of volume
 # -Price changes per day
 # -1,2,3,4,5 day lags
#We want to obtain all of this data, and then add it to the stock's dataframe

def feature_creation(stock_data, SMA_50: bool, SMA_200: bool,EMA_12: bool, EMA_26: bool,RSI: bool, MACD: bool, Bollinger: bool,
                     Volume_MA: bool, Price_Changes: bool, Lag: bool):

    # Simple / Exponential Moving Average.
    if SMA_50 == True:
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    if SMA_200 == True:
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    if EMA_12 == True:
        stock_data['EMA_12'] = stock_data['Close'].ewm(span = 12, adjust=False).mean()
    if EMA_26 == True:
        stock_data['EMA_26'] = stock_data['Close'].ewm(span = 26, adjust=False).mean()
       
                                                                             
    # Relative Strength Index                                                            
    if RSI == True:
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window = 14).mean() #14 day average gain
        loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean() #14 day average loss
        rsi = gain / loss
        stock_data['RSI'] = 100*(1 - 1/(1+rsi))
    
    
    # Moving Average Convergence Divergence
    if MACD == True:
        stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']  # MACD Line
        stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line
        stock_data['MACD_Histogram'] = stock_data['MACD'] - stock_data['Signal_Line']  # MACD Histogram

    
    # Bollinger Bands
    if Bollinger == True:
        stock_data['Middle_Band'] = stock_data['Close'].rolling(window=20).mean()  # 20-day SMA (Middle Band)
        stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data['Close'].rolling(window=20).std()  # Upper Band
        stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data['Close'].rolling(window=20).std()  # Lower Band

    # Moving Average of Volume
    if Volume_MA == True:
        stock_data['Volume_MA_20'] = stock_data['Volume'].rolling(window=20).mean()  
    
    # Price Changes per day
    if Price_Changes == True:
        stock_data['Price_Change_5d'] = (stock_data['Close'].shift(-5) - stock_data['Close']) / stock_data['Close']
        
        
    # Lagges from 1,2,3,4,5 days
    if Lag == True:
        for lag in [1,2,3,4,5]:
            stock_data[f'Lag_{lag}'] = stock_data['Price_Change_5d'].shift(lag)
    #print(f'Features for {stock} added.')
    return None  
# We add all the features
SMA_50 = True
SMA_200 = True
EMA_12 = True
EMA_26 = True                     
RSI = True
MACD = True
Bollinger = True
Volume_MA = True
Price_Changes = True
Lag = True
#feature_creation(MSFT_data, SMA_50, SMA_200,EMA_12, EMA_26,RSI, MACD, Bollinger,Volume_MA,Price_Changes)

#-----------------------------------Machine Learning-------------------------------------------
def run_machine_learning(stock_data) -> dict:
    """ Return a dict that contains 3 keys
    'Data': List of 4 datasets
    'Classification_Model': List of the classification model and importance matrix, Accuracy
    'Regression_Model': List of the regression model and importance matrix, MSE
    """
    features = ['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram', 
            'Middle_Band', 'Upper_Band', 'Lower_Band', 'Volume_MA_20','Lag_1','Lag_2','Lag_3','Lag_4','Lag_5']
    estimators = 200
    depth = 5
    

    ml_dict = {'Data': [],'Classification_Model': [],'Regression_Model': []}
    
    
    stock_data = stock_data.dropna()
    features.append('Date')
    X = stock_data[features]    
    
    
    # Y_binary Data Splitting
    stock_data['Binary'] = stock_data['Price_Change_5d'].apply(lambda x: 1 if x>0 else 0)
    Y_binary = stock_data[['Date','Binary']] 
    
    # Y_change Data Splitting   
    Y_change = stock_data[['Date','Price_Change_5d']]    
    
    # Data Splitting with features  
    
    X_train, X_test, Y_binary_train, Y_binary_test, Y_change_train, Y_change_test = train_test_split(X,Y_binary,Y_change,
                                                                                                         test_size = 0.20, 
                                                                                                         shuffle = False)   
    
    ml_dict['Data'].extend([X_train, X_test, Y_binary_train, Y_binary_test, Y_change_train, Y_change_test])
    
    
    X_train = X_train.drop('Date', axis=1)
    X_test = X_test.drop('Date', axis=1)

    
    ### Classification (binary)
    # Model Creation Attempt
    try:
        #start = time.time()

        model_classification = XGBClassifier(n_estimators=estimators, learning_rate=0.1, max_depth=depth, eval_metric='logloss')
        model_classification.fit(X_train, Y_binary_train['Binary'])
        ml_dict['Classification_Model'].append(model_classification)


        # Importance Matrix
        classification_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model_classification.feature_importances_})
        classification_importance = classification_importance.sort_values(by='Importance', ascending=False)
        ml_dict['Classification_Model'].append(classification_importance)    


        # Prediction and Accuracy
        accuracy = accuracy_score(Y_binary_test['Binary'], model_classification.predict(X_test))
        ml_dict['Classification_Model'].append(accuracy)

        #end = time.time()
        #length = end - start
        #print(f"Classification done in:  {round(length,4)} seconds.")
    except: 
        ml_dict['Classification_Model'].append(['Na', 'Na', 'Na'])


    ### Regression (Change)
    # Model Creation Attempt
    try:
        #start = time.time()
        model_regression = RandomForestRegressor(n_estimators = estimators, random_state = 42, n_jobs = -1)
        model_regression.fit(X_train, Y_change_train['Price_Change_5d'])
        ml_dict['Regression_Model'].append(model_regression)
    
        # Importance Matrix
        regression_importance = pd.DataFrame({'Feature': X_train.columns,'Importance': model_regression.feature_importances_})

        # Sort by importance
        regression_importance = regression_importance.sort_values(by='Importance', ascending=False)
        ml_dict['Regression_Model'].append(regression_importance)    
    
        # Prediction and Accuracy
        
        MSE = mean_squared_error(Y_change_test['Price_Change_5d'], model_regression.predict(X_test))
        ml_dict['Regression_Model'].append(MSE)
        end = time.time()
        #length = end - start
        #print(f"Regression done in:  {round(length,4)} seconds.")
    except: 
        ml_dict['Regression_Model'].append(['Na', 'Na', 'Na'])
    return ml_dict


def create_prediction_model(tickers: list):
    ### Given a list of tickers, return back a dictionary of dictonaries, keys are the stocks, the values are the stocks ml_dict
    start = time.time()
    
    model_dict = {}
    print(f"There are {len(tickers)} tickers to predict.")
    for i, stock in enumerate(tickers):
        # load the stock data
        stock_data = load_stock_data(stock)
        # feature creation
        feature_creation(stock_data, SMA_50, SMA_200, EMA_12, EMA_26, RSI, MACD, Bollinger,Volume_MA, Price_Changes, Lag)

        # regression running
        ml_dict = run_machine_learning(stock_data)


        # add to our dictonary
        model_dict[stock] = ml_dict
        if len(tickers) < 10:
            print(f'Done {stock}')
        if i % 5 == 0:
            print('Running...')
    end = time.time()
    length = end - start
    
    print("Model Completed. Creation time was:", round(length,2), "seconds.")
    
    return model_dict


#We can create our final data dictonary by running our create_prediction_model(tickers)

#test_dict = create_prediction_model(tickers[:20])

def binary_actual_to_predicted(test_dict, stock) -> list:
    """ Given ml_dct, and a stock in ml_dict, return a list that contains the
    dataframe for the actual binary results from the test data, and the dataframe for the
    predicted binary results.
    """
    
    model_classification = test_dict[stock]['Classification_Model'][0]
    
    #obtaining the datasets
    
    X_train = test_dict[stock]['Data'][0]
    X_test = test_dict[stock]['Data'][1]

    X_train = X_train.drop('Date', axis=1)
    
    #preserving the date column
    Keep_Dates = X_test['Date'].copy()  
    X_test= X_test.loc[:, ~X_test.columns.duplicated(keep='first')]
    Keep_Dates = X_test.pop('Date')
    
    # Re-running our model
    model_classification = test_dict[stock]['Classification_Model'][0]
    model_classification.fit(X_train, test_dict[stock]['Data'][2]['Binary'])
    
    Y_binary_actual = test_dict[stock]['Data'][3]
    
    Y_binary_predicted = model_classification.predict(X_test)

    
    #dataframe creation
    ## actual
    df_Y_binary_actual = Y_binary_actual 
    df_Y_binary_actual.reset_index(drop=True, inplace=True)
    
    
    ## predicted
    df_Y_binary_predicted = pd.DataFrame(Y_binary_predicted)
    df_Y_binary_predicted = df_Y_binary_predicted.rename(columns={df_Y_binary_predicted.columns[0]: 'Binary'})

    df_Y_binary_predicted.reset_index(drop=True)
    Keep_Dates = Keep_Dates.reset_index(drop=True)
    df_Y_binary_predicted['Date'] = Keep_Dates


    return [df_Y_binary_actual,df_Y_binary_predicted]

def regression_actual_to_predicted(test_dict, stock) -> list:
    """ Given ml_dct, and a stock in ml_dict, return a list that contains the
    dataframe for the actual % price change results from the test data, and the dataframe for the
    predicted % price change results.
    """
    
    model_regression = test_dict[stock]['Regression_Model'][0]
    
    #obtaining the datasets
    
    X_train = test_dict[stock]['Data'][0]
    X_test = test_dict[stock]['Data'][1]

    X_train = X_train.drop('Date', axis=1)
    
    #preserving the date column
    Keep_Dates = X_test['Date'].copy()  
    X_test= X_test.loc[:, ~X_test.columns.duplicated(keep='first')]
    Keep_Dates = X_test.pop('Date')
    
    # Re-running our model
    model_regression = test_dict[stock]['Regression_Model'][0]
    model_regression.fit(X_train, test_dict[stock]['Data'][4]['Price_Change_5d'])
    
    Y_regression_actual = test_dict[stock]['Data'][5]
    
    Y_regression_predicted = model_regression.predict(X_test)

    
    #dataframe creation
    ## actual
    df_Y_regression_actual = Y_regression_actual 
    df_Y_regression_actual.reset_index(drop=True, inplace=True)
    
    
    ## predicted
    df_Y_regression_predicted = pd.DataFrame(Y_regression_predicted)
    df_Y_regression_predicted = df_Y_regression_predicted.rename(columns={df_Y_regression_predicted.columns[0]: 'Price_Change_5d'})

    df_Y_regression_predicted.reset_index(drop=True)
    Keep_Dates = Keep_Dates.reset_index(drop=True)
    df_Y_regression_predicted['Date'] = Keep_Dates

    return [df_Y_regression_actual, df_Y_regression_predicted]


#-------------------------------Plotting----------------------------------------------------
def create_overlay_plot(test_dict, modeltype ,stock, date_col, value_col, ax, show_legend=False, show_ylabel=False):
    
    if modeltype == 'Classification':
        lst = binary_actual_to_predicted(test_dict, stock)
    
    elif modeltype == 'Regression':
        lst = regression_actual_to_predicted(test_dict, stock)
    else:
        return 'Not a valid modeltype'
    

    df1 = lst[0].tail(30)
    df2 = lst[1].tail(30)


    # Plot actual movement
    ax.plot(df1[date_col], df1[value_col], label='Actual', linestyle='-', marker='o')

    # Plot predicted movement
    ax.plot(df2[date_col], df2[value_col], label='Predicted', linestyle='--', marker='s')

    # Formatting
    ax.set_xlabel('Date')
    if show_ylabel:
        ax.set_ylabel('Binary Movement')

    #if show_legend:
    ax.legend()
        
    if modeltype == 'Classification':
        accuracy = 100 * test_dict[stock]['Classification_Model'][2]
        ax.set_title(f'Overlayed Plot of Actual & Predicted of: {stock}. The Accuracy is: {round(accuracy,2)}% ')
        if show_ylabel:
            ax.set_ylabel('Binary Movement (0: Negative Change, 1: Positive Change)')
        
    elif modeltype == 'Regression':
        MSE = test_dict[stock]['Regression_Model'][2]
        ax.set_title(f'Overlayed Plot of Actual & Predicted of: {stock}. The MSE is: {round(MSE,6)}')
        if show_ylabel:
            ax.set_ylabel('% Price Change in 5 Days (0.01 = 1%)')

    ax.grid(True)
    t = np.arange(-0.1, 0.15, 0.02)
    ax.set_yticks(t)
    ax.set_xticks(df1[date_col])
    ax.set_xticklabels(df1[date_col], rotation=45, ha='right')

def plot_multiple_stocks(test_dict, stock_names, date_col, value_col, modeltype):
    # Create a 2 x k grid that fits all the stock graphs on one page

    n=len(stock_names)
    k = n // 2
    if n % 2 == 1:
        k = k + 1

    fig, axs = plt.subplots(k, 2, figsize=(30, 10*k))

    # Flatten the axes array to make indexing easier
    axs = axs.flatten()


    for i, stock in enumerate(stock_names):
        create_overlay_plot(test_dict, modeltype, stock,
                            date_col, value_col, 
                            axs[i],show_legend = (i == 0), show_ylabel = (i == 0) )

    #plt.tight_layout()
    plt.show()
#----------------------------------------Prompt------------------------------------
def get_list_input():
    """Use for list inputs"""
    while True:
        user_input = input("Enter a list of stock tickers (e.g., ['AAPL', 'MSFT']): ").strip()
        try:
            stock_list = ast.literal_eval(user_input)
            if isinstance(stock_list, list):
                return stock_list
            else:
                print("Error: Please enter a valid list.")
        except (SyntaxError, ValueError):
            print("Invalid input. Please enter a valid list.")

def main_menu():
    """First Menu, Prompt: Create new or load existing stock data."""
    while True:
        print("\nMain Menu:")
        print("1 - Create new stock data")
        print("2 - Load existing stock data")

        try:
            choice = input("Enter your choice: ").strip()
        except KeyboardInterrupt:
            print('\nExiting Menu, call start_menu() to return')
            return
        if choice == "1":
            stock_list = get_list_input()
            if stock_list is None:
                return
            print('Creating Stock Data')
            save_stock_data(create_stock_data(stock_list), folder = 'Stock Data')
        elif choice == "2":
            stock_list = get_list_input()
            if stock_list is None:
                return
        else:
            print("Invalid choice, try again.")
            continue

        #function_c_menu(stock_list)
        machine_learning_menu(stock_list)

def machine_learning_menu(stock_list):
#def function_c_menu(stock_list):
    """Prompt: Run Machine Learning on the stock list."""
    while True:
        print(f"\nThe stock list is: {stock_list}")
        print("Run Machine Learning on this list?")
        print("1 - Yes")
        print("2 - No (Go back to Main Menu)")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            print('Running Machine Learning on this list')
            stock_data = create_prediction_model(stock_list)
            continuing_on_menu(stock_data)
        elif choice == "2":
            return
        else:
            print("Invalid choice, try again.")

def continuing_on_menu(stock_data):
    """Prompt: Options after Machine Learning has been run."""
    while True:
        print("\nMachine Learning Results:")
        print("1 - Stock Data")
        print("2 - Classification Plotting")
        print("3 - Regression Plotting")
        print("4 - Go back to previous prompt")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            stock_data_menu(stock_data)
        elif choice == "2":
            #function_d_menu(stock_data)
            classificaton_menu(stock_data)
        elif choice == "3":
            #function_e_menu(stock_data)
            regression_menu(stock_data)
        elif choice == "4":
            return
        else:
            print("Invalid choice, try again.")

def stock_data_menu(stock_data):
    """Prompt: Retrieve a single stock's data"""
    while True:

        print(f"Stocks: {list(stock_data)}")
        stock = input("Enter a stock ticker to retrieve data (or type 'back' to return): ").strip()
        if stock.lower() == "back":
            return
        elif stock in stock_data:
            print(f"Data for {stock}: {stock_data[stock]}")
        else:
            print("Stock not found. Try again.")
def classificaton_menu(stock_data):
#def function_d_menu(stock_data):
    """Prompt: Classification Graph of up to 2 graphs."""
    while True:
        print(f"Stocks: {list(stock_data)}")
        stock_list = get_list_input()
        #FunctionD(stock_list)
        print('Plotting')
        plot_multiple_stocks(stock_data, stock_list, date_col = 'Date',value_col = 'Binary', modeltype = 'Classification')
        while True:
            print("\nWould you like to plot a different stock?")
            print("1 - Yes (Enter a new stock list)")
            print("2 - Go back")

            choice = input("Enter your choice: ").strip()
            if choice == "1":
                break
            elif choice == "2":
                return
            else:
                print("Invalid choice, try again.")
def regression_menu(stock_data):
#def function_e_menu(stock_data):
    """Prompt: Regression Graph of up to 2 graphs."""
    while True:
        print(f"Stocks: {list(stock_data)}")
        stock_list = get_list_input()
        #FunctionE(stock_list)
        print('Plotting')
        plot_multiple_stocks(stock_data, stock_list,
                             date_col = 'Date', 
                             value_col = 'Price_Change_5d', 
                             modeltype = 'Regression')
        

        while True:
            print("\nWould you like to plot a different stock?")
            print("1 - Yes (Enter a new stock list)")
            print("2 - Go back")

            choice = input("Enter your choice: ").strip()
            if choice == "1":
                break
            elif choice == "2":
                return
            else:
                print("Invalid choice, try again.")
if __name__ == "__main__":
    main_menu()
#def start_menu():
#    """Function to enter the prompt system."""
#    main_menu()
#
#if __name__ == "__main__":
##    print("Type `start_menu()` to enter Menu Mode.")