import backtrack_on_index as btrk
import pandas as pd
import datetime

idx_names = ["AR_26","volume"]#,"ATR_14"]
tstates_set = range(3,4)
info_set = ["idx_names", "tstates", "c_in", "c_out", "rtn_annual", "annual_excess", \
            "win_rate", "maxdown", "sharpe", "trade_freq", "score"]

info_table = pd.DataFrame(index = range(len(tstates_set)), columns = info_set)

i_d = 0
for states in tstates_set:
    print(states)
    
    ind_info = btrk.backtrack_on_index(stk_name = "000300", \
                                       start = datetime.date(2011,1,1), \
                                       split = datetime.date(2018,10,1), \
                                       end = datetime.date(2019,3,1), \
                                       tstates = states, \
                                       idx_names = idx_names)
    for info_n in info_set:
        info_table[info_n][i_d] = ind_info[info_n]
        
    i_d += 1

