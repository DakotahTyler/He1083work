#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:03:43 2019

@author: dtyler
"""

import os
import datetime
import sys
import numpy as np


path = ('/home/dtyler/Desktop/DocumentsDT/Analysis/eq_widths')

for filename in os.listdir(path):
	if filename.startswith("k"):
		obs = open(filename).read().split()
		ob = str(obs[1])
		if float(obs[1]) > .0035:
			print(filename)