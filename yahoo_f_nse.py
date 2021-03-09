import requests
from datetime import datetime
# create two dates with year, month, day, hour, minute, and second
date1 = datetime(2010,12, 10, 5,30)
date2 = datetime(2020, 12, 8, 5, 30)
p1=str(int(date1.timestamp()))
p2=str(int(date2.timestamp()))
url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-historical-data"

querystring = {"period1":p1,"period2":p2,"symbol":"^NSEI","frequency":"1h","filter":"history"}

headers = {
    'x-rapidapi-key': "********************************",
    'x-rapidapi-host': "*********************"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

data=response.json()
date=list(map(lambda x : datetime.fromtimestamp(x['date']).strftime("%m/%d/%Y, %H:%M:%S"),data['prices']))
open_=list(map(lambda x :x['open'],data['prices']))
high_=list(map(lambda x :x['high'],data['prices']))
close=list(map(lambda x :x['close'],data['prices']))
df=pd.DataFrame()
df['Date']=date
df['open']=open_
df['high']=high_
df['close']=close
df