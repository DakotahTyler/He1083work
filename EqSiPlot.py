
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

path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/eq_widths')

eq = []
#eq cal dates
eqdate = []

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		eq.append(float(obs[1]))
		if filename.startswith("s"):
			find_date = filename
			dat = find_date.split("_")[3][2:]+find_date.split("_")[4].split(".")[0]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate.append(fd)
		if filename.startswith("k"):
			find_date = filename
			dat = find_date.split("k4i22")[1].split("t")[0]+find_date.split("t")[1].split(".")[0][:-2]
			fdat = iter(dat)
			fixed_date = "-".join(a+b for a,b in zip(fdat,fdat))
			fd = datetime.datetime.strptime(fixed_date, '%y-%m-%d-%H-%M')
			eqdate.append(fd)
#eq w julian dates
fixed = Time(eqdate, format="datetime", scale="utc")
eqjdate = fixed.jd



#plt.plot(time,eq, 'bo')
#plt.locator_params(axis="eq",nbins=3)
#plt.title("Eq Width over time")
#plt.show()


# get s index into si, sidate 

sdata = np.loadtxt("/home/dtyler/Desktop/DocumentsDT/Analysis/solars.dat", usecols=[0,1,2])
#jd,si
jd, si = np.split(sdata,[-1], axis=1)
jd, pc = np.split(sdata,[-2], axis=1)
jud = jd +2400000

sdate = Time(jud, format='jd',out_subfmt="date_hm", scale='utc')
#julian to calendar
tempdate = np.array(sdate.iso).ravel()
#2007-03 - present..... 4800:
sipdate=tempdate
sip=si
sidate = tempdate[4800:]
si=si[4800:-45]
sijdate=jud[4800:-45]

#plt.plot(sipdate,sip)
#plt.xticks([])
#plt.show()

ts = np.loadtxt("/home/dtyler/Desktop/Milbourne_etal_2019_Table_3.dat", usecols=[0,3])
#jd,si
jd, tsi = np.split(ts,[-1], axis=1)
jud = jd +2457222.5

ts = Time(jud, format='jd',out_subfmt="date_hm", scale='utc')


#plt.plot(sidate,si)
#plt.title("S Index")
#plt.xlabel("Date")
#plt.ylabel("intensity")
#plt.locator_params(axis="Date",nbins=8)
#plt.show()

#fig, ax1 = plt.subplots()

#red = "tab:red"
#ax1.set_xlabel("Julian Dates")
#ax1.set_ylabel("Eq Width", color=red)
#ax1.plot(eqjdate, eq,".",linewidth=1, color=red)
#ax1.tick_params(axis="y", labelcolor=red)
#ax1.locator_params(nbins = 8)
#ax2 = ax1.twinx()

#blue = "tab:blue"
#ax2.set_ylabel("S Index", color=blue)
#ax2.plot(sijdate, si,"-",alpha=.4, linewidth=3, color=blue)
#ax2.tick_params(axis="y", labelcolor=blue)

#fig.tight_layout()
#plt.show()

fig, ax1 = plt.subplots()
blue= "tab:blue"
red = "tab:red"
ax1.set_xlabel("Julian Dates")
ax1.set_ylabel("Eq Width", color=red)
ax1.plot(eqjdate,eq,".", linewidth=1,color=red)
#ax1.plot(sijdate, si,"-",alpha=.4, linewidth=3, color=blue)
ax1.tick_params(axis="y", labelcolor=red)
#ax1.set_yticks([])
ax1.locator_params(nbins = 8)
ax2 = ax1.twinx()
plt.title("Eq Width & S Index Vs Time")

blue = "tab:blue"
ax2.set_ylabel("S Index", color=blue)
ax2.plot(jud, tsi,"-",alpha=.4,linewidth=3, color=blue)

#jud, tsi,"-",alpha=.4,linewidth=3, color=blue
ax2.tick_params(axis="y", labelcolor=blue)
ax2.set_yticks([])

ax3 = ax2.twinx()

ax3.plot(sijdate, si,"-",alpha=.4, linewidth=3, color=blue)
#jud, tsi,"-",alpha=.4,linewidth=3, color=blue
ax3.tick_params(axis="y", labelcolor=blue)
#ax3.set_yticks([])

fig.tight_layout()
plt.show()

