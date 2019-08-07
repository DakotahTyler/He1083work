#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:35:14 2019

@author: dtyler
"""
import collections
import os
import datetime
import sys
import numpy as np
import hashlib
import matplotlib.pyplot as plt


path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/multiple_obs')


seen = {}


for filename in os.listdir(path):
	if filename.startswith("k"):
		name = filename
		b = name.split("k4i22")[1].split(".")[0].split("t")[0]
		if b not in seen:
			seen[b] = 1
		else:
			seen[b] +=1
				

for i,j in seen.items():
	if (j>=2):
		
		fil= open(path+'/'+'multi_EWs_'+i+'.txt','w')
		for filename in os.listdir(path):
			if filename.startswith("k"):
				name = filename
				b = name.split("k4i22")[1].split(".")[0].split("t")[0]
				time_str = name.split("t")[1].split(".")[0].split("txt")[0]
				
				time_aux = int(time_str)
				time_hr = 3600*time_aux/10000
				time_min = 60*(time_aux - (time_aux/10000)*10000)/100
				time_sec = (time_aux - (time_hr/3600)*10000 -(time_min/60)*100)
				time = time_hr + time_min + time_sec
			
				if (b==i):
					coll2 = np.loadtxt(path+'/'+name,  usecols=(1), unpack=True)
					fil.write(str(coll2[0])+'\t'+str(time)+'\n')	
		fil.close()


for filename in os.listdir(path):
	if filename.startswith("multi_EWs"):
		ew, time = np.loadtxt(path+'/'+filename,  usecols=(0,1), unpack=True)
		plt.plot((time - min(time))/60.0, 100*(ew-ew[0])/ew[0], 'o-')

plt.xlabel('time [min]')
plt.ylabel(r'$\Delta$ EW [%]')
plt.show()
