from numpy import *
import os
import datetime
import pandas as pd
from dateutil.parser import parse
import matplotlib.pyplot as plt
from astropy.io import fits
import pylab
import astropy.units as u
plt.ion()
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pylab
import astropy.units as u
import astropy

#extract eq widths

path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/1')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/1")

eq1 = []
#eq cal dates
eqdate1 = []

for filename in os.listdir(path):
	if filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq1.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate1.append(fd)
#eq w julian dates
fixed1 = Time(eqdate1, format="datetime", scale="utc")
eqdate1st = [str(x) for x in eqdate1]

time1 = []
for time in eqdate1:
    time1 = np.append(time1, time.hour * 60 + time.minute)


####


path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/2')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/2")

eq2 = []
#eq cal dates
eqdate2 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq2.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate2.append(fd)
#eq w julian dates
fixed2 = Time(eqdate2, format="datetime", scale="utc")
eqdate2st = [str(x) for x in eqdate2]

time2 = []
for time in eqdate2:
    time2 = np.append(time2, time.hour * 60 + time.minute)



path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/3')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/3")

eq3 = []
#eq cal dates
eqdate3 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq3.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate3.append(fd)
#eq w julian dates
fixed3 = Time(eqdate3, format="datetime", scale="utc")
eqdate3st = [str(x) for x in eqdate3]

time3 = []
for time in eqdate3:
    time3 = np.append(time3, time.hour * 60 + time.minute)




path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/4')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/4")

eq4 = []
#eq cal dates
eqdate4 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq4.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate4.append(fd)
#eq w julian dates
fixed4 = Time(eqdate4, format="datetime", scale="utc")
eqdate4st = [str(x) for x in eqdate4]

time4 = []
for time in eqdate4:
    time4 = np.append(time4, time.hour * 60 + time.minute)




path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/5')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/5")

eq5 = []
#eq cal dates
eqdate5 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq5.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate5.append(fd)
#eq w julian dates
fixed5 = Time(eqdate5, format="datetime", scale="utc")
eqdate5st = [str(x) for x in eqdate5]

time5 = []
for time in eqdate5:
    time5 = np.append(time5, time.hour * 60 + time.minute)




path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/6')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/6")

eq6 = []
#eq cal dates
eqdate6 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq6.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate6.append(fd)
#eq w julian dates
fixed6 = Time(eqdate6, format="datetime", scale="utc")
eqdate6st = [str(x) for x in eqdate6]

time6 = []
for time in eqdate6:
    time6 = np.append(time6, time.hour * 60 + time.minute)



path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/7')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/7")

eq7 = []
#eq cal dates
eqdate7 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq7.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate7.append(fd)
#eq w julian dates
fixed7 = Time(eqdate7, format="datetime", scale="utc")
eqdate7st = [str(x) for x in eqdate7]

time7 = []
for time in eqdate7:
    time7 = np.append(time7, time.hour * 60 + time.minute)



path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/8')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/8")

eq8 = []
#eq cal dates
eqdate8 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq8.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate8.append(fd)
#eq w julian dates
fixed8 = Time(eqdate8, format="datetime", scale="utc")
eqdate8st = [str(x) for x in eqdate8]

time8 = []
for time in eqdate8:
    time8 = np.append(time8, time.hour * 60 + time.minute)



path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/9')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/9")

eq9 = []
#eq cal dates
eqdate9 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq9.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate9.append(fd)
#eq w julian dates
fixed9 = Time(eqdate9, format="datetime", scale="utc")
eqdate9st = [str(x) for x in eqdate9]

time9 = []
for time in eqdate9:
    time9 = np.append(time9, time.hour * 60 + time.minute)




path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/10')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/10")

eq10 = []
#eq cal dates
eqdate10 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq10.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate10.append(fd)
#eq w julian dates
fixed10 = Time(eqdate10, format="datetime", scale="utc")
eqdate10st = [str(x) for x in eqdate10]

time10 = []
for time in eqdate10:
    time10 = np.append(time10, time.hour * 60 + time.minute)





path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/11')

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs/11")

eq11 = []
#eq cal dates
eqdate11 = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq11.append(float(obs[1]))
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate11.append(fd)
#eq w julian dates
fixed11 = Time(eqdate11, format="datetime", scale="utc")
eqdate11st = [str(x) for x in eqdate11]

time11 = []
for time in eqdate11:
    time11 = np.append(time11, time.hour * 60 + time.minute)

os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs")

"""e1 = pd.Series(eq1)
e1 = e1.pct_change()
t1=time1-time1[0]
N = isnan(e1)
e1[N]=0


#plt.plot(t3, e3, ".-")
#plt.plot(t4, e4+.5, ".-")
#plt.plot(t5, e5+1, ".-")
#plt.plot(t6, e6+1.5, ".-")
#plt.plot(t8, e8+2, ".-")
#plt.plot(t9, e9+2.5, ".-")
#plt.xlabel("Time (min)")
#plt.ylabel("Percent Change")
#plt.show()

#plt.plot(t11, e11, ".-")
#plt.xlabel("Time (min)")
#plt.ylabel("Percent Change")
#plt.show()


#plt.plot(t1, e1, ".-")
#plt.plot(t2, e2+.5,".-")
#plt.plot(t7, e7+1, ".-")
#plt.plot(t10, e10+1.5, ".-")
#plt.xlabel("Time (min)")
#plt.ylabel("Percent Change")
#plt.show()
yerr=0.05
#plot1"""

#e1 = np.array(eq1)
e1 = np.array(eq1)/eq1[0] -1
e2 = np.array(eq2)/eq2[0] -1
e3 = np.array(eq3)/eq3[0] -1
e4 = np.array(eq4)/eq4[0] -1
e5 = np.array(eq5)/eq5[0] -1
e6 = np.array(eq6)/eq6[0] -1
e7 = np.array(eq7)/eq7[0] -1
e8 = np.array(eq8)/eq8[0] -1
e9 = np.array(eq9)/eq9[0] -1
e10 = np.array(eq10)/eq10[0] -1

t1=time1-time1[0]
t2=time2-time2[0]
t3=time3-time3[0]
t4=time4-time4[0]
t5=time5-time5[0]
t6=time6-time6[0]
t7=time7-time7[0]
t8=time8-time8[0]
t9=time9-time9[0]
t10=time10-time10[0]












fig, (ax1,ax2,ax3) = plt.subplots(3, sharex= True)
fig.suptitle(" Eq Width Short Time Scale Variations (Percentage)")
ax1.plot(t4,e4,".-g", label= "070531")
ax1.set_ylim([-0.4,0.4])
ax1.legend(loc=2)
ax2.plot(t8,e8,".-g", label="070817")
ax2.set_ylim([-0.4,0.4])
ax2.legend(loc=2)
ax3.plot(t9,e9,".-r", label= "070818")
ax3.set_ylim([-0.4,0.4])
ax3.legend(loc=2)
ax1.axhline(y=0,xmin=0,c="grey",linewidth=0.5,zorder=0)
ax2.axhline(y=0,xmin=0,c="grey",linewidth=0.5,zorder=0)
ax3.axhline(y=0,xmin=0,c="grey",linewidth=0.5,zorder=0)



#plot2
fig, (ax1,ax3,ax4) = plt.subplots(3, sharex= True)
fig.suptitle("Eq Width Short Time Scale Variations (Absolute)")
ax1.plot(t3,eq3,".-b",label="070529")
ax1.set_ylim([0,.0015])
ax1.legend(loc=2)
#ax2.plot(t2,eq2,".-r", label="070528")
#ax2.set_ylim([0,.0015])
#ax2.legend(loc=3)
ax3.plot(t6,eq6,".-g", label="070619")
ax3.set_ylim([0,.0015])
ax3.legend(loc=3)
ax4.plot(t5,eq5,".-m", label="070614")
ax4.set_ylim([.0005,.0015])
ax4.legend(loc=3)
ax1.axhline(y=eq3[0],xmin=0,c="grey",linewidth=0.5,zorder=0)
#ax2.axhline(y=eq2[0],xmin=0,c="grey",linewidth=0.5,zorder=0)
ax3.axhline(y=eq6[0],xmin=0,c="grey",linewidth=0.5,zorder=0)
ax4.axhline(y=eq5[0],xmin=0,c="grey",linewidth=0.5,zorder=0)

#plot3

fig, (ax2,ax3) = plt.subplots(2, sharex=True)
fig.suptitle("Eq Width Short Time Scale Variations (Percentage)")
#ax1.plot(t1,e1,".-b", label="070519")
#ax1.set_ylim([-0.4,0.4])
#ax1.legend(loc=3)
#ax1.axhline(y=0,xmin=0,c="grey",linewidth=0.5,zorder=0)
ax2.plot(t7,e7,".-g", label="070628")
ax2.set_ylim([-0.4,0.4])
ax2.legend(loc=3)
ax2.axhline(y=0,xmin=0,c="grey",linewidth=0.5,zorder=0)
ax3.plot(t10,e10,".-r", label="070921")
ax3.set_ylim([-0.4,0.4])
ax3.legend(loc=3)
ax3.axhline(y=0,xmin=0,c="grey",linewidth=0.5,zorder=0)