#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from astropy.time import Time
import datetime
import julian
import pandas as pd
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pylab
import astropy.units as u
import astropy
plt.ion()



# get s index into si, sidate 

sdata = np.loadtxt("/home/dtyler/Desktop/DocumentsDT/Analysis/solars.dat", usecols=[0,1,2])
#jd,si
jd, si = np.split(sdata,[-1], axis=1)
jd, pc = np.split(sdata,[-2], axis=1)
jud = jd +2400000

sdate = Time(jud, format='jd',out_subfmt="date_hm", scale='utc')
tempdate = np.array(sdate.iso).ravel()
#2007 - present
sidate = tempdate[4800:]
si=si[4800:]

#plt.plot(sidate,si)
#plt.title("S Index")
#plt.xlabel("Date")
#plt.ylabel("intensity")
#plt.locator_params(axis="Date",nbins=8)
#plt.show()