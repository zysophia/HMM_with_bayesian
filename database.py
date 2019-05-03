import sqlite3
import datetime
import numpy as np
import tracking


class StockData():
    def __init__(self):
        
        self.conn = sqlite3.connect('stocks.db')
        self.cursor = self.conn.cursor()
        
        
        
    def fill_data(self, stkname = "000300"):
        
        self.cursor.execute("CREATE TABLE IF NOT EXISTS {0} \
                            (date date PRIMARY KEY, \
                             open float, high float, low float, close float, volume float)"\
                             .format("stk"+stkname))
        
        lastday = self.cursor.execute("select max(date) from {0}".format("stk"+stkname)).fetchone()[0]
        if lastday:
            lastday = datetime.datetime.strptime(lastday, "%Y-%m-%d").date()
        
        filename = 'C:/Users/lenovo/Desktop/毕业设计/'+ stkname +'.txt'
        with open(filename, 'r') as f:
            
            lines = reversed(f.readlines())
            for line in lines:
                if not line:
                    continue
                sp = line.split()
                if len(sp)<6 : continue # skip lines without data
                if sp[0] == "时间": break
                d_tmp = datetime.datetime.strptime(sp[0], "%Y/%m/%d").date()
                if lastday and d_tmp < lastday: break
                
                o, h, l, c, v = sp[1:6]
                self.cursor.execute("REPLACE INTO {0}(date,open,high,low,close,volume) VALUES {1}" \
                                    .format("stk"+stkname, (str(d_tmp), o, h, l, c, v)))
                
        self.conn.commit()
        
        
        
    def cal_index(self, stkname = "000300", idx = ["return_1"]):
        
        
        tdata = self.cursor.execute("SELECT * FROM {0} ORDER BY date".format("stk"+stkname)).fetchall()
        tdate, tclose, thigh, tlow, topen = [], [], [], [], []
        for d in tdata:
            tdate.append(d[0])
            topen.append(d[1])
            thigh.append(d[2])
            tlow.append(d[3])
            tclose.append(d[4])
        
        if "return_1" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN return_1 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[1:]
            close = np.array(tclose)
            return1 = np.diff(close)/close[:-1]
            
            for i in range(len(return1)):
                self.cursor.execute("UPDATE {0} SET return_1 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, return1[i], str(date[i])))
                
            self.conn.commit()
            
            
        if "hurst_120" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN hurst_120 float".format("stk"+stkname))
            except:
                pass
            
            hurst120 = np.zeros(len(date))
            for i in range(119,len(tclose)):
                print(i)
                ts = tclose[i-119:i+1]
                hurst120 = tracking.calcHurst2(ts)
                
                self.cursor.execute("UPDATE {0} SET hurst_120 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, hurst120, str(tdate[i])))
                if i%50==0:
                    self.conn.commit()
                    
            self.conn.commit()
            
            
        if "ma_20" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN ma_20 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[20:]
            ma20 = (np.cumsum(tclose)[20:] - np.cumsum(tclose)[:-20])/20
            
            for i in range(len(ma20)):
                self.cursor.execute("UPDATE {0} SET ma_20 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, ma20[i], str(date[i])))
                
            self.conn.commit()    
            
            
        if "volatility_20" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN volatility_20 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[1:]
            return1 = np.diff(close)/close[:-1]
            volatility20 = [np.std(return1[i:i+20]) for i in range(len(return1)-20)]
            date = date[-len(volatility20):]
            
            for i in range(len(volatility20)):
                self.cursor.execute("UPDATE {0} SET volatility_20 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, volatility20[i], str(date[i])))
                
            self.conn.commit() 
            
        if "bias_20" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN bias_20 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[20:]
            ma20 = (np.cumsum(tclose)[20:] - np.cumsum(tclose)[:-20])/20
            close = tclose[20:]
            bias20 = close/ma20 - 1
            
            for i in range(len(bias20)):
                self.cursor.execute("UPDATE {0} SET bias_20 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, bias20[i], str(date[i])))
                
            self.conn.commit() 
            
            
        if "ma_5" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN ma_5 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[5:]
            ma5 = (np.cumsum(tclose)[5:] - np.cumsum(tclose)[:-5])/5
            
            for i in range(len(ma5)):
                self.cursor.execute("UPDATE {0} SET ma_5 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, ma5[i], str(date[i])))
                
            self.conn.commit()    
            
            
        if "volatility_5" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN volatility_5 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[1:]
            return1 = np.diff(close)/close[:-1]
            volatility5 = [np.std(return1[i:i+5]) for i in range(len(return1)-5)]
            date = date[-len(volatility5):]
            
            for i in range(len(volatility5)):
                self.cursor.execute("UPDATE {0} SET volatility_5 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, volatility5[i], str(date[i])))
                
            self.conn.commit() 
            
        if "bias_5" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN bias_5 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[5:]
            ma5 = (np.cumsum(tclose)[5:] - np.cumsum(tclose)[:-5])/5
            close = tclose[5:]
            bias5 = close/ma5 - 1
            
            for i in range(len(bias5)):
                self.cursor.execute("UPDATE {0} SET bias_5 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, bias5[i], str(date[i])))
                
            self.conn.commit() 
            
        if "ATR_14" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN ATR_14 float".format("stk"+stkname))
            except:
                pass
            
            date = tdate[1:]
            tr = np.zeros(len(date))
            for i in range(1,len(date)+1):
                tr[i-1] = max(max(abs(thigh[i]-tlow[i]),abs(thigh[i]-tclose[i-1])),abs(tclose[i-1]-tlow[i]))
            atr14 = (np.cumsum(tr)[14:] - np.cumsum(tr)[:-14])/14
            date = tdate[-len(atr14):]
            
            for i in range(len(atr14)):
                self.cursor.execute("UPDATE {0} SET ATR_14 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, atr14[i], str(date[i])))
                
            self.conn.commit()  
            
        if "ma_diff_5_20" in idx:
            
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN ma_diff_5_20 float".format("stk"+stkname))
            except:
                pass
            
            
            ma5 = (np.cumsum(tclose)[5:] - np.cumsum(tclose)[:-5])/5
            ma20 = (np.cumsum(tclose)[20:] - np.cumsum(tclose)[:-20])/20
            diff = ma5[-len(ma20):] - ma20
            date = tdate[-len(diff):]
            
            for i in range(len(diff)):
                self.cursor.execute("UPDATE {0} SET ma_diff_5_20 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, diff[i], str(date[i])))
                
            self.conn.commit()  
           
        if "macd" in idx:
             
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN macd float".format("stk"+stkname))
            except:
                pass
            
            ma12 = (np.cumsum(tclose)[12:] - np.cumsum(tclose)[:-12])/12
            ma26 = (np.cumsum(tclose)[26:] - np.cumsum(tclose)[:-26])/26
            diff = ma12[-len(ma26):] - ma26
            macd = (np.cumsum(diff)[9:] - np.cumsum(diff)[:-9])/9
            date = tdate[-len(macd):]
            
            for i in range(len(macd)):
                self.cursor.execute("UPDATE {0} SET macd = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, macd[i], str(date[i])))
                
            self.conn.commit()             
            
        if "RSI_6" in idx:
             
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN RSI_6 float".format("stk"+stkname))
            except:
                pass
            
            rsi = np.zeros(len(tclose)-6)
            for i in range(6,len(tclose)):
                d = np.diff(tclose[i-6:i+1])
                rsi[i-6] = np.sum([j for j in d if j>0])/np.sum(np.abs(d))*100
            date = tdate[-len(rsi):]
            
            for i in range(len(rsi)):
                self.cursor.execute("UPDATE {0} SET RSI_6 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, rsi[i], str(date[i])))
                
            self.conn.commit()     
            
        if "RSI_12" in idx:
             
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN RSI_12 float".format("stk"+stkname))
            except:
                pass
            
            rsi = np.zeros(len(tclose)-12)
            for i in range(12,len(tclose)):
                d = np.diff(tclose[i-12:i+1])
                rsi[i-12] = np.sum([j for j in d if j>0])/np.sum(np.abs(d))*100
            date = tdate[-len(rsi):]
            
            for i in range(len(rsi)):
                self.cursor.execute("UPDATE {0} SET RSI_12 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, rsi[i], str(date[i])))
                
            self.conn.commit()     
            
        if "AR_26" in idx:
             
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN AR_26 float".format("stk"+stkname))
            except:
                pass
            
            ar = np.zeros(len(tclose)-25)
            for i in range(25,len(tclose)):
                d1 = np.array(thigh[i-25:i+1]) - np.array(topen[i-25:i+1])
                d2 = np.array(topen[i-25:i+1]) - np.array(tlow[i-25:i+1])
                ar[i-25] = sum(d1)/sum(d2)*100
                date = tdate[-len(ar):]
            
            for i in range(len(ar)):
                self.cursor.execute("UPDATE {0} SET AR_26 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, ar[i], str(date[i])))
                
            self.conn.commit() 
          
        if "BR_26" in idx:
             
            try:
                self.cursor.execute("ALTER TABLE {0} ADD COLUMN BR_26 float".format("stk"+stkname))
            except:
                pass
            
            br = np.zeros(len(tclose)-26)
            for i in range(26,len(tclose)):
                d1 = np.array(thigh[i-25:i+1]) - np.array(tclose[i-26:i])
                d2 = np.array(tclose[i-26:i]) - np.array(tlow[i-25:i+1])
                br[i-26] = sum(d1)/sum(d2)
                date = tdate[-len(br):]
            
            for i in range(len(br)):
                self.cursor.execute("UPDATE {0} SET BR_26 = {1} WHERE date = '{2}'" \
                                    .format("stk"+stkname, br[i], str(date[i])))
                
            self.conn.commit() 
          
    def get_data(self, stkname = "000300", idx = "close",  \
                 stdate = datetime.date(2018,1,1), endate = datetime.date(2019,1,1)):
        
        selected = self.cursor.execute("SELECT {3} FROM {0} WHERE date >= '{1}' AND date <= '{2}' ORDER BY date" \
                            .format("stk"+stkname, str(stdate), str(endate), idx)).fetchall()
            
        data = []
        for i in selected:
            row = []
            for j in i:
                row.append(j)
            data.append(row)
        
        # multiple data is available, but recommend fetch one at a time
        return np.array(data)
                


if __name__ == "__main__":
    stockdata = StockData()
    stockdata.fill_data("000300")
    stockdata.cal_index("000300",["return_1","ma_20","volatility_20","bias_20",\
                                  "ma_5", "volatility_5", "bias_5","ATR_14",\
                                  "ma_diff_5_20","macd","RSI_6","RSI_12","BR_26","AR_26"])
    #close = stockdata.get_data("000300","volume",datetime.date(2018,1,1),datetime.date(2019,1,1))
    stockdata.conn.close()