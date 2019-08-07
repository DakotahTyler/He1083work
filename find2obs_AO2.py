#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

bdate=0
path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/eq_widths')
#path = ('/home/antonija/Documents/Projects/sun1083/solar_reu/ews')

seen = {}












for filename in os.listdir(path):
	if filename.startswith("k"):
		name = filename
		b = name.split("k4i22")[1].split(".")[0].split("t")[0]
		if b not in seen:
			seen[b] = 1
		else:
			seen[b] +=1
				

k=0
print(seen)
for i,j in seen.items():
	if (j>=3):
		print(i,j)
		fil= open(path+'/'+'multi_EWs_'+i+'.txt','w')
		for filename in os.listdir(path):
			if filename.startswith("k"):
				name = filename
				b = name.split("k4i22")[1].split(".")[0].split("t")[0]
				time_str = name.split("t")[1].split(".")[0].split("txt")[0]
				print(time_str)
				time_aux = int(time_str)
				time_hr = 3600*time_aux/10000
				time_min = 60*(time_aux - (time_aux/10000)*10000)/100
				time_sec = (time_aux - (time_hr/3600)*10000 -(time_min/60)*100)
				time = time_hr + time_min + time_sec
				print(time)
				if (b==i):
					coll2 = np.loadtxt(path+'/'+name,  usecols=(1), unpack=True)
					print(coll2[0])
					fil.write(b+'\t'+ str(coll2[0])+'\t'+str(time)+'\n')
		fil.close()
	for filename in os.listdir(path):
		if filename.startswith("multi_EWs"):
			ew = np.loadtxt(path+'/'+filename,  usecols=(1), unpack=True)
			bd = x[0]
			ew_mean = np.average(ew)
			ew_min = ew_mean-min(ew)
			ew_max = max(ew)-ew_mean
			print(ew_min, ew_max)
			plt.errorbar((k,k), (ew_mean,ew_mean), yerr=((ew_min,ew_min),(ew_max,ew_max)),ms=3,elinewidth=2)
			k+=1
plt.show()
