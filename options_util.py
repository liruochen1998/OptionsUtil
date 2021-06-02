#%% Imports
from unicodedata import category
import QuantLib as ql
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import statistics
import math
from db_connect import DB_connect

#%% Global Constants for defaults
DIVIDEND_YIELD = 0.01
RISK_FREE_RATE = 0.03
OPT_COLs = ["date", "maturity_date", "strike_price", "call_put", "opt_price", "spot_price"]
OPT_COLs_MIN = ['time', 'mat_date', 'strike', 'call_put', 'close_opt', 'close_spot']
UNDERLIER = '510050.SH'
NUM_TRADING_DAYS = 250
DEFAULT_DATE = '20210104'
DEFAULT_FREQ = 'day'
# profiler: kernprof -l -v run.py

#%% Option Class, Print Util
class OptionsUtil():
    """
    An Option Utility Class
    """
    def __init__(self, eval_dttm, mat_dttm, type_, strike_, spot_, opt_):
        """
        The data follows this standard order
        [curr_date, maturity_date, strike_price, type, opt_price, spot_price]
        """
        self.eval_dttm = eval_dttm 
        self.mat_dttm = mat_dttm 
        self.strike_ = strike_
        self.type_ = type_
        self.opt_ = opt_
        self.spot_ = spot_

        self.q_ = DIVIDEND_YIELD 
        self.rf_ = RISK_FREE_RATE 
        self.ivol_ = 1

    def __str__(self):
        return f"Opt from {self.eval_dttm} to {self.mat_dttm}: Type {self.type_}, \
                Strike: {self.strike_}, Spot: {self.spot_}, Opt: {self.opt_}"
    
    
#%% SeriesCollectionUtil Class
class SeriesCollectionUtil():
    """
    Util for a collection of series
    """
    def __init__(self):
        self.all_series = dict()
        self.length = 0
    
    def add_series(self, data, name):
        # check whether the length confrom before every adding
        length = len(data.series)
        print(length)
        if self.add_series == False and length != self.length:
            raise Exception(f"The new series to add has length {length}, which is not equal to the length of others {self.length}.")
        self.all_series[name] = data.series
        # if we have min freq, need to convert dates (dict) to (array)
        if data.freq == 'min':
            dates_arr = []
            for curr_t in data.dates.values():
                dates_arr += list(set(curr_t))
            self.all_series['dates'] = dates_arr
        else:
            self.all_series['dates'] = data.dates
    
    def plot_series(self):
        df = pd.DataFrame(self.all_series)
        df = df.set_index('dates')
        df.plot(figsize=(40, 10))
    
    def to_csv(self):
        df = pd.DataFrame(self.all_series)
        df = df.set_index('dates')
        df.to_csv('output.csv')



#%% Series Class 
class SeriesUtil():
    """
    Super Class Series for Utilities
    """
    def __init__(self, underlier, method, from_='begin', to_='end', freq='day'):
        self._underlier = underlier
        self._method = method
        self.from_ = from_
        self.to_ = to_
        self.freq = freq
        self._series = []
        self._data = []
        self._dates = []

        self.load_dates()
        self.load_data()

    def __str__(self):
        return f"Series with underlier {self._underlier}, from {self.from_} to {self.to_}, with {len(self._data)} data"

    @property
    def data(self):
        return self._data

    @property
    def underlier(self):
        return self._underlier
    
    @underlier.setter
    def underlier(self, underlier):
        self._underlier = underlier
    
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, method):
        self._method = method

    @property
    def series(self):
        return self._series
    
    @property
    def dates(self):
        return self._dates
    
    def load_dates(self):
        if self.freq == 'day':
            if self.from_ == 'begin' and self.to_ == 'end':
                self._dates = get_dates(self.underlier)
            else:
                self._dates = get_dates_from_to(self.from_, self.to_, self._underlier)
        if self.freq == 'min':
            dates_dict = dict()
            data_dict = dict()
            all_trading_days = get_trading_days(self.from_, self.to_)
            for day in all_trading_days:
                all_data = csv_get_opt_at(day, self.underlier)
                if all_data is None: break
                dates_dict[day] = all_data['time'].to_numpy()
                data_dict[day] = all_data
            self._dates = dates_dict
            self._data = data_dict

    def load_data(self):
        if self.freq == 'day':
            if self.from_ == 'begin' and self.to_ == 'end':
                self._data = get_opt_infos(self.underlier)
            else:
                self._data = get_opt_infos_from_to(self.from_, self.to_, self._underlier) 
        # if self.freq == 'min':
        #     # store as dict??
        #     data_dict = dict()
        #     for day in self._dates:
        #         data_dict[day] = csv_get_opt_at(day, self.underlier)
        #         if data_dict[day] == None: break
        #     self._data = data_dict



    def plot(self):
        return
    
#%% VIX Class 
class VIXSeriesUtil(SeriesUtil):
    """
    Util for VIX
    """
    def __init__(self, underlier=UNDERLIER, method='var_swap', from_='begin', to_='end', freq='min'):
        super().__init__(underlier, method, from_, to_, freq)
        self._method = method

    @SeriesUtil.method.setter
    def method(self, method):
        if method != 'implied_vol' or method != 'var_swap': raise ValueError("Method must be either 'implied_vol' or 'var_swap'")
        self._method = method
    
    def calculate_vix(self):
        if self.freq == 'day':
            self._series = calculate_VIX_series(self.data, self._method, self.freq)
        if self.freq == 'min':
            # load data from csv first
            for day in self._data:
                # print(self._data[day])
                self._series += calculate_VIX_series(self._data[day], self._method, freq='min')
                
    

#%% Rolling Std and Avg Class
class RollingSeriesUtil(SeriesUtil):
    """
    Util for rolling series
    """
    def __init__(self, underlier=UNDERLIER, window_size=20, method='simple', from_='begin', to_='end'):
        super().__init__(underlier, method, from_, to_)
        self._window_size = window_size
        self._rolling_avg = []
        self._rolling_std = []
    
    @property
    def window_size(self):
        return self._window_size
    
    @window_size.setter
    def window_size(self, window_size):
        self._window_size = window_size

    def load_data(self):
        if self.from_ == 'begin' and self.to_ == 'end':
            self._data = get_opt_underlier_spot_price(self._underlier)
        else:
            self._data = get_opt_underlier_spot_price_from_to(self.from_, self.to_, self._underlier) 

    def calculate_avg(self):
        self._rolling_avg = calculate_rolling_average(self._data, self._window_size)
        self._series = self._rolling_avg
    
    def calculate_std(self):
        self._rolling_std = calculate_rolling_std(self._data, self._window_size) 
        self._series = self._rolling_std * 100
        
#%% Options Utilities
def calculate_VIX_series(data, method='var_swap', freq=DEFAULT_FREQ):
    """
    Calculate iVIX series feeding all options data
    :param data: <raw data pulled from get_opt_infos()
    :param method: <str> 'implied_vol' or 'var_swap'
    """
    if method != 'implied_vol' and method != 'var_swap': raise Exception(f"Do not have method {method}.")
    vix = []
    if freq == 'day':
        df = pd.DataFrame(data, columns=OPT_COLs)
        df = df.replace([708001000, 708002000], ['CALL', 'PUT']) # code -> text
        for idx, group in df.groupby(['date']):
            df_sub = group
            terms = implied_volatility_method(df_sub) if method == 'implied_vol' else variance_swap_method(df_sub)
            near_term = terms[0]
            next_term = terms[1]
            vix_curr = calculate_VIX_from_sigmas(near_term[0], next_term[0], near_term[1], next_term[1])
            vix.append(vix_curr)
        return vix  
    elif freq == 'min':
        col_name_dict = col2col(OPT_COLs_MIN, OPT_COLs)
        df = data.rename(columns=col_name_dict)
        date = ''
        # the date here is actually by min
        for idx, group in df.groupby(['date']):
            curr_date = idx[0:10] # only get the date
            if curr_date != date:
                date = curr_date
                print(date)
            df_sub = group
            if df_sub.isnull().values.any(): 
                vix.append(np.nan)
                continue # some opt_price and spot_price is NaN at 14:59
            terms = implied_volatility_method(df_sub) if method == 'implied_vol' else variance_swap_method(df_sub)
            near_term = terms[0]
            next_term = terms[1]
            vix_curr = calculate_VIX_from_sigmas(near_term[0], next_term[0], near_term[1], next_term[1])
            vix.append(vix_curr)
        return vix
    else:
        raise ValueError(f"Frequency {freq} is not a valid frequency")
    

def implied_volatility_method(df):
    """
    Helper for using implied volatility average method
    """
    near = calculate_IV_curve(df, 1)
    next = calculate_IV_curve(df, 2)
    return near, next

def variance_swap_method(df):
    """
    Helpder for using variance swap method
    """
    near = calculate_variance_swap_vol(df, 1)
    next = calculate_variance_swap_vol(df, 2)
    if near == None:
        near = next
        next = calculate_variance_swap_vol(df, 3) 
    return near, next

# @profile
def calculate_variance_swap_vol(df, term):
    """
    Calculate Sigma of a given unit of time, and Time-to-expiration
    :param data 
    :param term
    :return sigma
    :return TTE
    """
    # df = pd.DataFrame(data, columns=OPT_COLs)
    df['maturity_group'] = pd.factorize(df["maturity_date"], sort=True)[0] + 1
    df_grouped = df.groupby(df["maturity_group"]) # split by expiration date 1->near, 2->next, ignore 3&4
    df_near = df_grouped.get_group(term)
    # calcualte time-to-expiration
    t_curr = pd.to_datetime(df_near["date"].iat[0])
    t_exp = pd.to_datetime(df_near["maturity_date"].iat[0])
    tte_near = (t_exp - t_curr).days
    if tte_near <= 7:
        # print("Warning: Less than 7 days, omit!")
        return None
    tte_near_frac = tte_near / 365
    df_near = df_near.pivot(index="strike_price", columns="call_put", values='opt_price')
    # df_near = pd.pivot_table(df_near, values='opt_price', index="strike_price", columns="call_put")
    # df_near = df_near.groupby(['call_put', 'strike_price']).agg({'opt_price': np.mean}).unstack(level='call_put')
    df_near["callPutDiff"] = (df_near["CALL"] - df_near["PUT"]).abs() # find the diff between call&put

    strikePrice = df_near["callPutDiff"].idxmin() # find the strike price of the minimum difference
    callPrice = df_near["CALL"][strikePrice]
    putPrice = df_near["PUT"][strikePrice]

    forwardPrice = calculate_forward_price(strikePrice, callPrice, putPrice, tte_near_frac)

    # Calculate Q(Ki)
    df = df_near.reset_index() # colapse columns after pivoting
    # K0 - the strike price equal to or immediately below F (for near- next- term)
    k0 = nearest_below(df['strike_price'].copy().to_numpy(), forwardPrice)
    # Ki - if Ki > K0: use put prices; if Ki < K0: use call prices; if Ki = K0, use average of call/put
    # df['p_ki'] = ""
    # LOOP
    # for ki in df['strike_price']:
    #     idx = df['strike_price'] == ki
    #     if ki > k0:
    #         df.loc[idx, 'p_ki'] = df.loc[idx, 'call']
    #     elif ki < k0:
    #         df.loc[idx, 'p_ki'] = df.loc[idx, 'put']
    #     else:
    #         df.loc[idx, 'p_ki'] = (df.loc[idx, 'put'] + df.loc[idx, 'call']) / 2
    # Change to VEC
    df['p_ki'] = df['CALL']
    # call_cols = df['strike_price'] > k0
    put_cols = df['strike_price'] < k0
    mid_col = df['strike_price'] == k0
    # df.loc[call_cols, 'p_ki'] = df.loc[call_cols, 'CALL']
    df.loc[put_cols, 'p_ki'] = df.loc[put_cols, 'PUT']
    # Performance  
    # df.loc[mid_col, 'p_ki'] = (df.loc[mid_col, 'PUT'] + df.loc[mid_col, 'CALL']) / 2
    s = pd.Series(mid_col)
    mid = s[s].index[0]
    df['p_ki'].iat[mid] = (df['PUT'].iat[mid] + df['CALL'].iat[mid]) / 2


    # Do not need this part in calculation, good for table-viz
    # df['optionType'] = ""
    # for ki in df['strike_price']:
    #     idx = df['strike_price'] == ki
    #     if ki > k0:
    #         df.loc[idx, 'optionType'] = 'call' 
    #     elif ki < k0:
    #         df.loc[idx, 'optionType'] = 'put' 
    #     else:
    #         df.loc[idx, 'optionType'] = 'call/put avg' 

    # comment out for performence
    # df = df.drop(columns=['call', 'put', 'callPutDiff'])

    # Contribution by Strike
    ert = calculate_expRT(tte_near_frac)
    # Iterrows() has SERIOUS performence issue, should AVOID
    # df['contributionByStrike'] = ""
    # contributionByStrike = []
    # for idx, row in df.iterrows():
    #     diff = 0
    #     curr = 0
    #     ki = df.loc[idx, 'strike_price']
    #     if idx == 0:
    #         diff = abs(ki - df.loc[idx+1, 'strike_price'])
    #     elif idx == len(df.index) - 1:
    #         diff = abs(ki - df.loc[idx-1, 'strike_price'])
    #     else:
    #         diff = abs(df.loc[idx+1, 'strike_price'] - df.loc[idx-1, 'strike_price']) / 2
    #     curr = diff / ki ** 2 * ert * df['p_ki'].iat[idx]
    #     contributionByStrike.append(curr)
    first_diff = df['strike_price'].diff().fillna(0)
    second_diff = first_diff.shift(-1).fillna(0)
    diff = (first_diff + second_diff) / 2
    diff.iat[0] = diff.iat[0] * 2
    diff.iat[-1] = diff.iat[-1] * 2
    # comment for performance
    # df['contributionByStrike'] = df['p_ki'].mul(ert).mul(diff) / (df['strike_price'].pow(2))

    sec_term = 1 / tte_near_frac * (forwardPrice / k0 - 1) ** 2
    first_term = (df['p_ki'].mul(ert).mul(diff) / df['strike_price'].pow(2)).sum() * 2 / tte_near_frac
    sig_2 = first_term - sec_term
    sigma = math.sqrt(sig_2)
    return sigma, tte_near

def calculate_BSM_implied_volatility(eval_dttm, mat_dttm, strike_, type_, opt_, spot_, q_=DIVIDEND_YIELD, rf_=RISK_FREE_RATE, ivol_=1):
    """
    Calculate implied volatility given [curr_date, maturity_date, strike, type, opt_price, spot]
    using QuantLib BSM European Options
    """
    # setup evaluation datetime
    calendar = ql.China()
    ql.Settings.instance().evaluationDate = eval_dttm

    # option parameters
    exercise = ql.EuropeanExercise(mat_dttm)
    payoff = ql.PlainVanillaPayoff(type_, strike_)
    option = ql.VanillaOption(payoff, exercise)

    # market data
    underlying = ql.SimpleQuote(spot_)
    dividend_yield = ql.FlatForward(eval_dttm, q_, ql.ActualActual())
    risk_free_rate = ql.FlatForward(eval_dttm, rf_, ql.ActualActual())
    start_vol = ql.BlackConstantVol(eval_dttm, calendar, ivol_, ql.ActualActual())

    # process
    process = ql.BlackScholesMertonProcess(ql.QuoteHandle(underlying), \
                                        ql.YieldTermStructureHandle(dividend_yield), \
                                        ql.YieldTermStructureHandle(risk_free_rate), \
                                        ql.BlackVolTermStructureHandle(start_vol))

    # method: analytic European
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    try:
        ivol_ = option.impliedVolatility(process=process, targetValue=opt_)
    except:
        # print("Option pricing engine cannot back-solve implied vol due to bisection constraint")
        ivol_ = 0

    # print(f"Implied Volatility is {ivol_}")
    return ivol_

def calculate_VIX_from_sigmas(sig1, sig2, t1, t2):
    """
    Calculate VIX using sigmas, TTEs from near- and next- term contracts. <float>
    """
    var1 = sig1 ** 2
    var2 = sig2 ** 2
    vix = 0
    vix_component = 0
    if t1 >= 30:
        vix_component = var1
    else: 
        vix_component = (var1 * t1 * (t2 - 30) / (t2 - t1) / 365 \
                    + var2 * t2 * (30 - t1) / (t2 - t1) / 365) * 365 / 30
    vix = 100 * math.sqrt(vix_component)
    return vix


def calculate_IV_curve(data, term, plot=False, avg=True):
    """
    IV_curve calculates implied volatility for a single day
    :param raw: <str[][]>  2d-arrary that stores option information
                [curr_date, maturity_date, strike, type, opt_price, spot]
    :param term(optional): <int> there are 4 terms in total 
    :param plot(optional): default to False
    :param avg(optional): default to True: calculate the average of the ivol
    :return: average ivol or array of ivols
    :return: Time to Expiration
    """
    # load opt data in to df
    df = pd.DataFrame(data, columns=["date", "maturity_date", "strike_price", "call_put", "opt_price", "spot_price"])
    df['maturity_group'] = pd.factorize(df["maturity_date"], sort=True)[0] + 1
    df.replace([708001000, 708002000], ["call", "put"], inplace=True) 
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], format="%Y%m%d")
    df["ivol"] = ""

    # calculate the ivol for each day
    for idx, row in df.iterrows():
        date_today = df["date"].loc[idx]
        eval_dttm = ql.Date(date_today.day, date_today.month, date_today.year)
        date_mat = df["maturity_date"].loc[idx]
        mat_dttm = ql.Date(date_mat.day, date_mat.month, date_mat.year)
        type = ql.Option.Call if row.call_put == 'call' else ql.Option.Put
        ivol = calculate_BSM_implied_volatility(eval_dttm, mat_dttm, row.strike_price, type, row.opt_price, row.spot_price)
        df.loc[idx, "ivol"] = ivol

    # Construct OTM contracts
    spot = df["spot_price"].iloc[0]
    k0 = nearest_below(df["strike_price"].copy().to_numpy(), spot)
    rows_to_drop = []
    for idx, row in df.iterrows():
        if row["strike_price"] < k0 and row["call_put"] == "call":
            rows_to_drop.append(idx)
        if row["strike_price"] >= k0 and row["call_put"] == "put":
            rows_to_drop.append(idx)
    df.drop(rows_to_drop, inplace=True)
    df.sort_values(by=["strike_price"], inplace=True)

    # seperate maturities
    df_grouped = df.groupby(df["maturity_group"])
    df_near = df_grouped.get_group(term)
    # calcualte TTE
    tte = (df_near["maturity_date"] - df_near["date"]).iloc[0].days
    if plot:
        df_near.plot(x="strike_price", y="ivol")
    ivols = df_near["ivol"].to_numpy()
    if avg:
        # potentially change it to weighted average instead of simple average
        return statistics.mean(ivols), tte
    return ivols, tte


#%% General Utilities
# used in VIX calculation -- variance swap method
# calculate e^RT
def calculate_expRT(time_to_expire, rf=DIVIDEND_YIELD):
    """
    Calculate e^{RT} given TTE and rf
    """
    ert = math.exp(rf * time_to_expire)
    return ert

# calculate Forward Price of the option
def calculate_forward_price(strike, call, put, tte, rf=RISK_FREE_RATE):
    """
    Calculate Forward Price (F) given strike, call, put, TTE, rf (<float>)
    """
    f = strike + calculate_expRT(tte, rf) * (call - put)
    return f

# convert object type to int
def obj2int(obj):
    """
    convert obj to int
    """
    return obj.astype(str).astype(int)


# find the nearest but smaller number of the input
# rows <int[]>
def nearest_below(rows, num):
    """
    Get the nearest number in <rows> that is smaller than <num>
    :param rows: <float[]>
    :param num: <float>
    """
    rows.sort()
    prev = rows[0] 
    for row in rows:
        if row > num:
            return prev
        else:
            prev = row
    return prev

def calculate_rolling_std(data, window_size):
    """
    Calculate Rolling standard deviation with a rolling window
    """
    s = pd.Series(data)
    ret = s.pct_change(1) # calculate return
    return ret.rolling(window_size).std().to_numpy() * math.sqrt(NUM_TRADING_DAYS) # annulize

def calculate_rolling_average(data, window_size):
    """
    Calculate Rolling average with a rolling window
    """
    s = pd.Series(data)
    ret = s.pct_change(1) # calculate return
    return (1 + ret.rolling(window_size).mean().to_numpy()) ** 252 - 1

def dttm_formatter(str):
    """
    20210101 to 2021-01-01
    """
    return '-'.join([str[:4], str[4:6], str[6:]])

def get_trading_days(from_='2021-01-01', to_='2021-05-30'):
    if len(from_) == 8 and len(to_) == 8:
        from_ = dttm_formatter(from_)
        to_ = dttm_formatter(to_)
    if len(from_) != 10 or len(to_) != 10:
        raise ValueError("dates should be in format yyyy-mm-dd")
    cal_china = mcal.get_calendar('SSE') # Shanghai Stock Exchange
    days = cal_china.valid_days(start_date=from_, end_date=to_)
    return days.strftime('%Y%m%d')

def col2col(key_col, val_col):
    zip_iter = zip(key_col, val_col)
    res_dict = dict(zip_iter)
    return res_dict

#%% Database APIs
def get_opt_infos(underlier=UNDERLIER):
    """
    get option info from Database
    """
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt, b.S_INFO_MATURITYDATE, b.S_INFO_STRIKEPRICE, b.S_INFO_CALLPUT, a.S_DQ_SETTLE, d.S_DQ_CLOSE
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT = d.TRADE_DT
    and a.S_INFO_WINDCODE = b.S_INFO_WINDCODE
    and b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE in ('{underlier}')
    and c.S_INFO_WINDCODE = d.S_INFO_WINDCODE
    order by a.TRADE_DT
    """
    data = db_connection.executeSQL(query)
    return data

def get_opt_infos_single_day(date='20150420', underlier=UNDERLIER):
    """
    get single-day option info from Database
    :param date: str eg. '20210130'
    """
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt, b.S_INFO_MATURITYDATE, b.S_INFO_STRIKEPRICE, b.S_INFO_CALLPUT, a.S_DQ_CLOSE, d.S_DQ_CLOSE
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT = '{date}'
    and a.TRADE_DT = d.TRADE_DT
    and a.S_INFO_WINDCODE = b.S_INFO_WINDCODE
    and b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE in ('{underlier}')
    and c.S_INFO_WINDCODE = d.S_INFO_WINDCODE
    order by a.TRADE_DT
    """
    data = db_connection.executeSQL(query)
    return data

def get_opt_infos_from_to(from_, to_, underlier=UNDERLIER):
    """
    get single-day option info from Database
    :param date: str eg. '20210130'
    """
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt, b.S_INFO_MATURITYDATE, b.S_INFO_STRIKEPRICE, b.S_INFO_CALLPUT, a.S_DQ_CLOSE, d.S_DQ_CLOSE
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT >= '{from_}' and a.TRADE_DT <= '{to_}'
    and a.TRADE_DT = d.TRADE_DT
    and a.S_INFO_WINDCODE = b.S_INFO_WINDCODE
    and b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE in ('{underlier}')
    and c.S_INFO_WINDCODE = d.S_INFO_WINDCODE
    order by a.TRADE_DT
    """
    data = db_connection.executeSQL(query)
    return data

def get_dates(underlier=UNDERLIER):
    """
    get all dates from Database
    """
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT = d.TRADE_DT
    and a.S_INFO_WINDCODE = b.S_INFO_WINDCODE
    and b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE in ('{underlier}')
    and c.S_INFO_WINDCODE = d.S_INFO_WINDCODE
    order by a.TRADE_DT
    """
    data = db_connection.executeSQL(query)
    res = []
    for row in data:
        res.append(row[0])
    return res 

def get_dates_from_to(from_, to_, underlier=UNDERLIER):
    """
    get all dates from Database
    """
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT >= '{from_}' and a.TRADE_DT <= '{to_}'
    and a.TRADE_DT = d.TRADE_DT
    and a.S_INFO_WINDCODE = b.S_INFO_WINDCODE
    and b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE in ('{underlier}')
    and c.S_INFO_WINDCODE = d.S_INFO_WINDCODE
    order by a.TRADE_DT
    """
    data = db_connection.executeSQL(query)
    res = []
    for row in data:
        res.append(row[0])
    return res 

def get_opt_underlier_spot_price_from_to(from_, to_, underlier=UNDERLIER):
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt, d.S_DQ_CLOSE
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT >= '{from_}' and a.TRADE_DT <= '{to_}'
    and a.TRADE_DT = d.TRADE_DT
    and a.S_INFO_WINDCODE = b.S_INFO_WINDCODE
    and b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE in ('{underlier}')
    and c.S_INFO_WINDCODE = d.S_INFO_WINDCODE
    order by a.TRADE_DT
    """
    data = db_connection.executeSQL(query)
    res = []
    for row in data:
        res.append(row[1])
    return res

def get_opt_underlier_spot_price(underlier=UNDERLIER):
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt, d.S_DQ_CLOSE
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT = d.TRADE_DT
    and a.S_INFO_WINDCODE = b.S_INFO_WINDCODE
    and b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE in ('{underlier}')
    and c.S_INFO_WINDCODE = d.S_INFO_WINDCODE
    order by a.TRADE_DT
    """
    data = db_connection.executeSQL(query)
    res = []
    for row in data:
        res.append(row[1])
    return res

def get_contracts(underlier=UNDERLIER):
    db_connection = DB_connect()
    query = f"""
    SELECT b.S_INFO_WINDCODE, b.S_INFO_MATURITYDATE, b.S_INFO_STRIKEPRICE, b.S_INFO_CALLPUT 
    FROM wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c 
    WHERE b.S_INFO_SCCODE = c.S_INFO_CODE
    and c.S_INFO_WINDCODE = '{underlier}'
    """
    data = db_connection.executeSQL(query)
    return data

#%% Read CSV APIs
def csv_get_underlier_spot_at(date=DEFAULT_DATE, underlier=UNDERLIER):
    path = f'/Users/ruochen/HFData/fund_min_insight/Bar_1Min/2021/{date}.csv'
    df = pd.read_csv(path)
    df = df[df['windcode'] == underlier]
    return df


def csv_get_opt_price_at(date=DEFAULT_DATE, underlier=UNDERLIER):
    path = f'/Users/ruochen/HFData/option_min_insight/Bar_1Min/2021/{date}.csv'
    df = pd.read_csv(path)
    return df


def csv_get_opt_desc(underlier=UNDERLIER):
    path = './option_description.csv'
    df = pd.read_csv(path)
    df = df[df['underlier'] == underlier]
    df.drop(columns=['purpose', 'code', 'full_name', 'settle_month', 'exer_type', 'exer_date', 'dlvy_date', 'ex_code', 'ex_name', 'sec_type'], inplace=True)
    return df

def csv_get_opt_at(date=DEFAULT_DATE, underlier=UNDERLIER):
    # merge opt_desc(static table) with opt_price(time-series table)
    try:
        df_opt_price = csv_get_opt_price_at(date, underlier)
        df_opt_desc = csv_get_opt_desc(underlier)
        df_spot = csv_get_underlier_spot_at(date, underlier)
        windcode_list = df_opt_desc['wind_code']
        df_opt_price = df_opt_price[df_opt_price['windcode'].isin(windcode_list)]
        df_merged = df_opt_price.merge(df_opt_desc, left_on='windcode', right_on='wind_code')
        df_merged = df_merged[(df_merged['ft_date'] <= int(date)) & (df_merged['lt_date'] >= int(date))] # make sure the contract is trading on current date
        # Now just need to merge spot
        df_merged = df_merged.merge(df_spot, on='time', suffixes=('_opt', '_spot'))
        df_merged['mat_date'] = pd.to_datetime(df_merged['mat_date'], format='%Y%m%d')
        df = df_merged[OPT_COLs_MIN]
        return df 
    except FileNotFoundError:
        print(f"File at {date} not found.")
        return None


