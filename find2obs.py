#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:35:14 2019

@author: dtyler
"""

import os

dates = []

path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/eq_widths')

for filename in os.listdir(path):
		if filename.startswith("k"):
			name = filename
			b = name.split("k4i22")[1].split(".")[0].split("t")[0]
			dates.append(b)
			seen = {}
			dupes = []
			for x in dates:
				if x not in seen:
					seen[x] = 1
				if seen[x] == 2:
					dupes.append(x)
					seen[x] +=1



