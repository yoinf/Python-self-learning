import plotly.express as px
import pandas as pd
#import openpyxl as xl


### parameters
# 1(east: 7.5°N 135°E), 2(south: 7°N 134.5°E), 3(west: 7.5°N 134°E), 4(north: 8°N 134.5°E)
dataRoute = 'Hsinchu.txt'
graphTitle = f'Wind Speed Distribution near Hsinchu (120.5°E 25°N), Feb 2005 ~ Jul 2018'
ys = [2005,2019]    # year range, 2005/03/01/09~2019/05/31/21
ms = [1,12]         # month range
ds = [1,31]         # day range
hs = [0,21]         # hour range, range(0,24,3)
plotRangeR = [0,50] # wind freq. swcales
wsS = 2 # wind speed scale
wsN = 8 # number of wind speed scales, Reds: 9 colors a cycle, # Plasma_r: 10 colors

### import txt data
with open(dataRoute, 'r') as f:
    fread = f.read()
linelen = 45
ls = int((len(fread)+1)/linelen) # line number, data number
data = []
maxAvg = [{1:0.0,2:0.0,3:0.0,4:0.0,5:0.0,6:0.0,7:0.0,8:0.0,9:0.0,10:0.0,11:0.0,12:0.0},
          {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0}]
daysPerM = [0]*13

### arrange data
for i in range(ls): # (1-indexed) 12, 20, 29, 37, 45 are comas. \n is the 54th char for every line.
    j = i*linelen
    wsp = float(fread[j+28:j+35])
    data.append([wsp,float(fread[j+36:j+44])]) # for the plot
    '''
    y = int(fread[j:j+4])          # year
    if ys[1]<y:
        break
    '''
    m = int(fread[j+4:j+6])        # month
    maxAvg[0][m] += wsp
    daysPerM[m] += 1
    maxAvg[1][m] = max(maxAvg[1][m],wsp)
    '''
    d = int(fread[j+6:j+8])        # day
    h = int(fread[j+8:j+10])       # hour
    if ys[0]<=y and ms[0]<=m and m<=ms[1] and ds[0]<=d and d<=ds[1] and hs[0]<=h and h<=hs[1]:
        data.append([float(fread[j+12:j+19]),float(fread[j+20:j+28])])    # wind speed (m/s) and wind direction
        float(fread[j+20:j+28])    # wind direction (rad, N=0, E=90)
        float(fread[j+29:j+36])    # significant wave (m)
        float(fread[j+37:j+44])    # TP, Primary wave mean period (s)
        float(fread[j+45:j+53])    # wave direction
    '''
for i in range(1,13):
    maxAvg[0][i] /= daysPerM[i]
maxAvg = pd.DataFrame(maxAvg,index=['Average wind speed (m/s)', 'Max wind speed (m/s)'])    
print(maxAvg)
maxAvg.to_excel("Hsinchu_wind_speed.xlsx")

### build pandas data for plot
names = ["direction","wind speed (m/s)","frequency (%)"]
df = {names[0]:['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSE','W','WNW','NW','NNW']*(wsN+1),
      names[1]:[],
      names[2]:[0.0]*16*(wsN+1)}
for i in range(wsN):
    df[names[1]].extend(['-'.join([str(i*wsS),str((i+1)*wsS)])]*16)
df[names[1]].extend([str(wsS*wsN)+'+']*16)
for ws,wd in data: # calculate freq.
    ws2 = int(ws/wsS) if ws<wsS*wsN else wsN
    wd2 = int((wd+11.25)/22.5) if wd<348.75 else 0
    df[names[2]][ws2*16+wd2] += 100.0/len(data)
df = pd.DataFrame(df)

### express figure
#df = px.data.wind()
fig = px.bar_polar(df,
                   range_r = plotRangeR,
                   theta = names[0],
                   color = names[1],
                   r = names[2],
                   template = "plotly",
                   color_discrete_sequence = px.colors.sequential.Reds, 
                   title = graphTitle)
fig.update_layout(font_size=16,
                  legend_font_size=16,
                  polar_radialaxis_ticksuffix='%')
fig.show()

### graph_object. the angle of plotly is counterclockwise
'''import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Barpolar(
    r=[20.0, 7.5, 15.0, 22.5, 2.5, 2.5, 12.5, 22.5],
    name='< 5 m/s',
    marker_color='rgb(242,240,247)'
))
fig.add_trace(go.Barpolar(
    r=[40.0, 30.0, 30.0, 35.0, 7.5, 7.5, 32.5, 40.0],
    name='5-8 m/s',
    marker_color='rgb(203,201,226)'
))
fig.add_trace(go.Barpolar(
    r=[57.5, 50.0, 45.0, 35.0, 20.0, 22.5, 37.5, 55.0],
    name='8-11 m/s',
    marker_color='rgb(158,154,200)'
))
fig.add_trace(go.Barpolar(
    r=[77.5, 72.5, 70.0, 45.0, 22.5, 42.5, 40.0, 62.5],
    name='11-14 m/s',
    marker_color='rgb(106,81,163)'
))

fig.update_traces(text=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
fig.update_layout(
    title=dict(text='Wind Speed Distribution in Peleliu, Palau'),
    font_size=16,
    legend_font_size=16,
    polar_radialaxis_ticksuffix='%',
    polar_angularaxis_rotation=90,
)
fig.show()'''
