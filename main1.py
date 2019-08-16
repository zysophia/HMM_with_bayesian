import backtrack_on_index as btrk
import pandas as pd
import datetime

idx_set = ["return_1","ma_20","volatility_20","bias_20","ma_5", "volatility_5"]#, \
          # "bias_5","ATR_14","ma_diff_5_20","macd","RSI_6",\
          # "RSI_12","BR_26", "AR_26","volume"]

tstates_set = range(3,5)
info_set = ["idx_names", "tstates", "c_in", "c_out", "rtn_annual", "annual_excess", \
            "win_rate", "maxdown", "sharpe", "trade_freq", "score"]

info_table = pd.DataFrame(index = range(len(idx_set)*len(tstates_set)), columns = info_set)

i_d = 0
for idx in idx_set:
    for states in tstates_set:
        print(idx,states)
        
        ind_info = btrk.backtrack_on_index(stk_name = "000300", \
                                           start = datetime.date(2006,1,1), \
                                           split = datetime.date(2015,1,1), \
                                           end = datetime.date(2018,1,1), \
                                           tstates = states, \
                                           idx_names = [idx])
        for info_n in info_set:
            info_table[info_n][i_d] = ind_info[info_n]
            
        i_d += 1