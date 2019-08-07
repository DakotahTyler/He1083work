#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 00:33:52 2019

@author: dtyler
"""

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

sdata = np.loadtxt("/home/dtyler/Desktop/DocumentsDT/Analysis/solars.dat", usecols=[0,1,2])
#jd,si
jd, si = np.split(sdata,[-1], axis=1)
jd, pc = np.split(sdata,[-2], axis=1)
jud = jd +2400000

sdate = Time(jud, format='jd',out_subfmt="date_hm", scale='utc')
#julian to calendar
tempdate = np.array(sdate.iso).ravel()
#2007-03 - present..... 4800:

sidate = tempdate
sijdate=jud
date=[]
yr,mon,day,oldeq = np.loadtxt("/home/dtyler/Desktop/he.dat", usecols=(0,1,2,3),unpack=True, dtype=np.int)
fmt = "[%y %m %d]"
yr = yr[:5474]
mon = mon[:5474]
day = day[:5474]
oldeq = oldeq[:5474]
d = np.column_stack((yr,mon,day))
for k in d:
	p = datetime.datetime.strptime(str(k),fmt)
	date.append(p)
fd = Time(date, format="datetime", scale="utc")
oejdate = fd.jd





fig, ax1 = plt.subplots()
blue= "tab:blue"
red = "tab:red"
ax1.set_xlabel("Julian Dates")
ax1.set_ylabel("S Index", color=blue)
ax1.plot(sijdate, si,"-",alpha=.4,linewidth=3, color=blue, label="S Index")
ax1.tick_params(axis="y", labelcolor=blue)
ax1.locator_params(nbins = 8)
ax2 = ax1.twinx()
plt.title("Old Eq Widths vs S Index")
ax1.set_ylim(.145,.20)


blue = "tab:blue"
ax2.set_ylabel("Eq Width (mA)", color=red)
ax2.plot(oejdate, oldeq,".", ms=1, color=red, label = "Eq Width")
ax2.tick_params(axis="y", labelcolor=red)
ax2.set_ylim(25,100)
