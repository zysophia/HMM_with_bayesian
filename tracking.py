import numpy as np
import pandas as pd


def zqmaxdown(series):
    '''
    calculate the maxdown value
    '''
    i = 0
    flag = 0
    downdegree = [0]
    
    while i<len(series):    
        if series[i]<max(series[flag:i+1]):
            for j in range(i,len(series)):
                if series[j]>max(series[flag:i+1]) or j==len(series)-1:                    
                    downdegree.append(min(series[i:j+1])/max(series[flag:i+1])-1)
                    flag = j
                    i = j
                    break
        i += 1
    return min(downdegree)


def rtn_series(close, signal, expense = (0,0)):
    '''
    claculate indexes which judge the performance of a strategy. 
    '''
    rtn_series = np.ones(len(close))
    c_in, c_out = 0,0
    rtn_rec = []
    info = {}
    full = False
    for i in range(1, len(close)):
        if not full and signal[i]:
            rtn_series[i] = rtn_series[i-1]*(1-expense[0])
            full = True
            point = i
            c_in += 1
        elif full and not signal[i]:
            rtn_series[i] = rtn_series[i-1]*(1-expense[1])
            full = False
            rtn_rec.append(close[i]/close[point]-1)
            c_out += 1
        elif not full:
            rtn_series[i] = rtn_series[i-1]
        else:
            rtn_series[i] = rtn_series[i-1]*close[i]/close[i-1]
    if full:
        rtn_rec.append(close[-1]/close[point]-1)
    
    rtn_rec = np.array(rtn_rec)        
    win_rate = sum(rtn_rec>0)/(len(rtn_rec)+1e-10)
    avg_rtn = np.mean(rtn_rec)
    
    info["rtn_annual"] = rtn_series[-1]**(1/len(rtn_series)*250)-1
    info["c_in"] = c_in
    info["c_out"] = c_out
    info["win_rate"] = win_rate
    info["avg_rtn"] = avg_rtn
    info["maxdown"] = zqmaxdown(rtn_series)
    info["volatility"] = np.std(np.diff(rtn_series)/rtn_series[:-1])*np.sqrt(250)
    info["sharpe"] = (info["rtn_annual"]-0.03)/info["volatility"]
    info["annual_excess"] = (rtn_series[-1] - close[-1]/close[0])*(1/len(rtn_series)*250)
    info["trade_freq"] = (c_in+c_out)/2/len(rtn_series)*250
    info["score"] = (info["rtn_annual"]*0.3 + info["maxdown"]*0.25 + info["sharpe"]*0.2\
                    + info["win_rate"]*0.15 - info["trade_freq"]/100*0.1)*100
        
    return rtn_series, info
        
        

'''
def state_params(hidden_states,track_len,close_v):
    # get performance params for each hidden_state in a fixed period 
    track_return = {}
    for i in range(int(max(hidden_states))+1):
        track_return[i] = []
    
    for i in range(len(hidden_states)-track_len):
        rtn = close_v[i+track_len-1]/close_v[i]-1
        close = close_v[i:i+track_len]
        effi = np.abs(close[-1]-close[0])/np.abs(np.diff(close)).sum()
        track_return[int(hidden_states[i])] += [rtn,effi],
        
    state_params = [{}for i in range(len(track_return.keys()))]
    for k,v in track_return.items():
        v = np.array(v)
        state_params[k]["state"] = k
        state_params[k]["len"] = len(v)
        state_params[k]["return"] = np.mean(v,axis = 0)[0] if len(v)>0 else np.nan
        state_params[k]["effi"] = np.mean(v,axis = 0)[1] if len(v)>0 else np.nan
        state_params[k]["win"] = np.mean(v[:,0]>0) if len(v)>0 else np.nan
        
    return state_params

def judge_state(state_params,p1 = (0.01,0.01),p2=(0.15,0.15),p3=(0.7,0.5)):
    # judge if a hidden_state has good or bad prospect
    good = []; bad = []
    for i in state_params:
        if i["len"]>10 and i["return"]>p1[0] and i["effi"]>p2[0] and i["win"]>p3[0]:
            good.append(i["state"])
        if i["len"]>10 and i["return"]<p1[1] and i["effi"]>p2[1] and i["win"]<p3[1]:
            bad.append(i["state"])
    return good,bad
    
def back_track(dates,close,buy,sell,expense = (0,0),start = 0):
    
    dates, close = dates[start:], close[start:]
    buy = [b - start for b in buy]
    sell = [s - start for s in sell]
    close = close/close[0]
    holding = np.ones(len(close))
    full = False
    for i in range(1, len(close)):
        if not full and i in buy:
            holding[i] = holding[i-1]*(1-expense[0])
            full = True
        elif full and i in sell:
            holding[i] = holding[i-1]*(1-expense[1])
            full = False
        elif not full:
            holding[i] = holding[i-1]
        else:
            holding[i] = holding[i-1]*close[i]/close[i-1]
            
    return dates, close, holding

def calcHurst2(ts):#https://blog.csdn.net/xiaodongxiexie/article/details/70800038 

    n_min, n_max = 2, len(ts)//2
    RSlist = []
    for cut in range(n_min, n_max+1):
        children = len(ts) // cut
        children_list = [ts[i*children:(i+1)*children] for i in range(cut)]
        L = []
        for a_children in children_list:
            Ma = np.mean(a_children)
            Xta = pd.Series(map(lambda x: x-Ma, a_children)).cumsum()
            Ra = max(Xta) - min(Xta)
            Sa = np.std(a_children)
            rs = Ra / Sa
            L.append(rs)
        RS = np.mean(L)
        RSlist.append(RS)
    return np.polyfit(np.log(range(2+len(RSlist),2,-1)), np.log(RSlist), 1)[0]
'''