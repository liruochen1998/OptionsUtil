#%% Imports
import QuantLib as ql
import numpy as np
import pandas as pd
import math
import statistics
from db_connect import DB_connect


#%% Global Constants
DIVIDEND_YIELD = 0.01
RISK_FREE_RATE = 0.03
OPT_COLs = ["date", "maturity_date", "strike_price", "call_put", "opt_price", "spot_price"]
UNDERLIER = '510050.SH'

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
    
    
#%% Options Utilities
def calculate_VIX_series(data, method):
    """
    Calculate iVIX series feeding all options data
    :param data: <raw data pulled from get_opt_infos()
    :param method: <str> 'implied_vol' or 'var_swap'
    """
    if method != 'implied_vol' and method != 'var_swap': raise Exception(f"Do not have method {method}.")
    df = pd.DataFrame(data, columns=OPT_COLs)
    vix = []
    for idx, group in df.groupby(['date']):
        df_sub = group
        terms = implied_volatility_method(df_sub) if method == 'implied_vol' else variance_swap_method(df_sub)
        near_term = terms[0]
        next_term = terms[1]
        vix_curr = calculate_VIX_from_sigmas(near_term[0], next_term[0], near_term[1], next_term[1])
        vix.append(vix_curr)
    return vix  

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

def calculate_variance_swap_vol(data, term):
    """
    Calculate Sigma of a given unit of time, and Time-to-expiration
    :param data 
    :param term
    :return sigma
    :return TTE
    """
    df = pd.DataFrame(data, columns=OPT_COLs)
    df['maturity_group'] = pd.factorize(df["maturity_date"], sort=True)[0] + 1
    df = df.replace([708001000, 708002000], ['call', 'put']) # code -> text
    df_grouped = df.groupby(df["maturity_group"]) # split by expiration date 1->near, 2->next, ignore 3&4
    df_near = df_grouped.get_group(term)

    # calcualte time-to-expiration
    t_curr = pd.to_datetime(df_near["date"])
    t_exp = pd.to_datetime(df_near["maturity_date"])
    tte_near = abs(t_exp - t_curr).dt.days.iloc[0] / 365    
    if tte_near * 365 <= 7:
        # print("Warning: Less than 7 days, omit!")
        return None
    df_near = df_near.pivot(index="strike_price", columns="call_put")["opt_price"]
    df_near["callPutDiff"] = abs(df_near["call"] - df_near["put"]) # find the diff between call&put

    strikePrice = df_near["callPutDiff"].idxmin() # find the strike price of the minimum difference
    callPrice = df_near["call"][strikePrice]
    putPrice = df_near["put"][strikePrice]

    forwardPrice = calculate_forward_price(strikePrice, callPrice, putPrice, tte_near)

    # Calculate Q(Ki)
    df = df_near.reset_index() # colapse columns after pivoting
    # K0 - the strike price equal to or immediately below F (for near- next- term)
    k0 = nearest_below(df['strike_price'].copy().to_numpy(), forwardPrice)
    # Ki - if Ki > K0: use put prices; if Ki < K0: use call prices; if Ki = K0, use average of call/put
    df['p_ki'] = ""
    for ki in df['strike_price']:
        idx = df['strike_price'] == ki
        if ki > k0:
            df.loc[idx, 'p_ki'] = df.loc[idx, 'call']
        elif ki < k0:
            df.loc[idx, 'p_ki'] = df.loc[idx, 'put']
        else:
            df.loc[idx, 'p_ki'] = (df.loc[idx, 'put'] + df.loc[idx, 'call']) / 2

    df['optionType'] = ""
    for ki in df['strike_price']:
        idx = df['strike_price'] == ki
        if ki > k0:
            df.loc[idx, 'optionType'] = 'call' 
        elif ki < k0:
            df.loc[idx, 'optionType'] = 'put' 
        else:
            df.loc[idx, 'optionType'] = 'call/put avg' 

    df.drop(columns=['call', 'put', 'callPutDiff'], inplace=True)
    df

    # Contribution by Strike
    df['contributionByStrike'] = ""
    ert = calculate_expRT(tte_near)
    contributionByStrike = []
    for idx, row in df.iterrows():
        diff = 0
        curr = 0
        ki = df.loc[idx, 'strike_price']
        if idx == 0:
            diff = abs(ki - df.loc[idx+1, 'strike_price'])
        elif idx == len(df.index) - 1:
            diff = abs(ki - df.loc[idx-1, 'strike_price'])
        else:
            diff = abs(df.loc[idx+1, 'strike_price'] - df.loc[idx-1, 'strike_price']) / 2
        curr = diff / ki ** 2 * ert * df.iloc[idx]['p_ki']
        contributionByStrike.append(curr)
    df['contributionByStrike'] = contributionByStrike

    sec_term = 1 / tte_near * (forwardPrice / k0 - 1) ** 2
    first_term = df['contributionByStrike'].sum() * 2 / tte_near
    sig_2 = first_term - sec_term
    sigma = math.sqrt(sig_2)
    return sigma, int(tte_near * 365) 

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

        

#%% Database APIs
def get_opt_infos(underlier=UNDERLIER):
    """
    get option info from Database
    """
    db_connection = DB_connect()
    query = f"""
    select DISTINCT a.trade_dt, b.S_INFO_MATURITYDATE, b.S_INFO_STRIKEPRICE, b.S_INFO_CALLPUT, a.S_DQ_SETTLE, d.S_DQ_CLOSE
    from wind.CHINAOPTIONEODPRICES a, wind.CHINAOPTIONDESCRIPTION b, wind.CHINAOPTIONCONTPRO c， wind.CHINACLOSEDFUNDEODPRICE d
    where a.TRADE_DT >= '20150209' and a.TRADE_DT < '20190101'
    and a.TRADE_DT = d.TRADE_DT
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
    select DISTINCT a.trade_dt, b.S_INFO_MATURITYDATE, b.S_INFO_STRIKEPRICE, b.S_INFO_CALLPUT, a.S_DQ_SETTLE, d.S_DQ_CLOSE
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
    select DISTINCT a.trade_dt, b.S_INFO_MATURITYDATE, b.S_INFO_STRIKEPRICE, b.S_INFO_CALLPUT, a.S_DQ_SETTLE, d.S_DQ_CLOSE
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
    where a.TRADE_DT >= '20150209' and a.TRADE_DT < '20190101'
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