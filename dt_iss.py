#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pylab
from sherpa.fit import Fit
from sherpa.plot import DataPlot, ModelPlot
from sherpa.data import Data1D
from sherpa.stats import LeastSq
from sherpa.plot import FitPlot
from sherpa.optmethods import MonCar
from sherpa.models.basic import Gauss1D
import astropy.units as u
from specutils import Spectrum1D
from specutils.analysis import equivalent_width

path = ('/home/dtyler/Desktop/DocumentsDT/Programs')
#fits_image_filename = ("k4i22120102t185130.fts")

for filename in os.listdir(path):
	if filename.startswith("s") or filename.startswith("k"):
		fits_image_filename = (filename)
		
		
		hdul = fits.open(fits_image_filename, ignore_missing_end=True)
		#hdul.info()
		
		hdr = hdul[0].header
		ref_value = hdr['CRVAL1']/10.0 - 1082.712
		#print(hdr['CRVAL1'])
		if 'AVGSDEV' in hdul[0].header: 
		   err_true  = hdr['AVGSDEV']
		else:
		  err_true = .005
		
		#These are just my raw X and Y data 
		w = hdul[0].data[0]
		p = hdul[0].data[1]
		
		#Create empty arrays for the slices that I want to fit
		w_temp = []
		p_temp = []
		
		#Loop to take only range of Helium line (1082.96 to 1083.09)
		for i in range(0,len(w)):
		     if w[i] > (1082.95+ref_value*5) and w[i]< (1083.1+ref_value*5): ### change back to the old wavelength range
		         w_temp.append(w[i])
		         p_temp.append(p[i])
		
		x = np.array(w_temp)
		y_raw = np.array(p_temp)
		
		
		#standard deviation and scaling factor applied
		y_min = min(y_raw)
		norm_factor = (max(y_raw) - y_min)
		err_scaled =  err_true/norm_factor 
		sd = np.ones(x.size) * err_scaled
		
		hdul.close()
		
		#normalize spliced intensities & invert
		y = -1.0*(y_raw-min(y_raw))/norm_factor +1.0
		#Set data and model for fits
		icorr=0
		
		G1 = Gauss1D('G1')
		d = Data1D('He 1083',x,y,staterror = sd)
		
		#guess parameters, this is important or sherpa won't know where to start looking 
		G1.fwhm = .05
		G1.pos = 1083.03 + ref_value*5
		mdl = G1 
		
		mplot = ModelPlot()
		mplot.prepare(d, mdl)
		dplot = DataPlot()
		dplot.prepare(d)
		mplot.overplot()
		
		#set error methods, ChiSq() or LeastSq()
		#Chi square is a way to compare which profile best describes data, ie: is it more gaussian or lorentzian
		#Least Square says how good the data fits the particular model instance
		#opt - optimizers improve the fit. Monte Carlo is what I used, it is slow but it is most robust. Many options on sherpas site
		ustat = LeastSq()
		opt =  MonCar() #LevMar() #NelderMead() #
		
		
		#apply actual Fit
		f = Fit(d, mdl, stat=ustat, method=opt)
		res = f.fit()
		fplot = FitPlot()
		mplot.prepare(d, mdl)
		fplot.prepare(dplot,mplot)
		fplot.plot()
		
		#param_errors = f.est_errors()
		
		
		#plotting routine
		plt.plot(d.x, d.y, "c.", label="Data")
		plt.plot(d.x, mdl(d.x), linewidth=2, label="Gaussian")
		plt.legend(loc=2)
		plt.title(fits_image_filename)
		plt.xlabel("Wavelength nm")
		plt.ylabel("Normalized Intensity")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/outputs")
		os.mkdir("images_"+fits_image_filename)
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/outputs/images_"+fits_image_filename)
		plt.savefig(fits_image_filename[:-4]+"_model"+str(icorr+1)+".png")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs")
		plt.close()
		
		############################################
		# Loop to Add more Gaussians if outliers > tolerance
		# I maxed out the iterations at 3, otherwise it may get stuck looping indefinitely
		tolerance = err_scaled*0.4 ### choose tolerance
		outliers = np.where(abs(mdl(d.x)-y) > tolerance)[0]
		if (len(outliers)>0):
			max_outlier = int(np.argmin(abs(mdl(outliers)-y[outliers])))
			for i in range(len(outliers)):
				if (abs(mdl(outliers[i])-y[outliers[i]]) > abs(mdl(max_outlier)-y[max_outlier])) and (outliers[i]>10) and (outliers[i] < len(x)-10):
					max_outlier = outliers[i]
		
			### define best model variables
			best_res = res.statval
			best_fit = f
			bmdl = mdl
			bres = best_fit.fit()
		
		################################################################# v and (max_outlier>10)        and (max_outlier < len(x)-10))
			while ((abs(mdl(x[max_outlier]) - y[max_outlier]) > tolerance) and (icorr < 4)):
				icorr+=1
				
				G2 = Gauss1D('G'+str(icorr+1))
				G2.fwhm = .02/(2*icorr+1)
				G2.pos = x[max_outlier]
				#I like 1083.11 for no smaller bump
				G2.pos.max = 1083.2
				#I like 1082.94 for no smaller bump
				G2.pos.min = 1082.7
				G2.ampl =  abs(mdl(x[max_outlier]) - y[max_outlier])
				mdl = mdl + G2 
		
				#print(mdl)
				mplot = ModelPlot()
				mplot.prepare(d, mdl)
				dplot = DataPlot()
				dplot.prepare(d)
				mplot.overplot()
			
				f = Fit(d, mdl, stat=ustat, method=opt)
				res = f.fit()
				fplot = FitPlot()
				mplot.prepare(d, mdl)
				fplot.prepare(dplot, mplot)
				fplot.plot()
		
				#print('residuals', res.statval)
						
				plt.plot(d.x, d.y, "c.", label="Data")
				plt.plot(d.x, mdl(d.x), linewidth=2, label="Gaussian")
				plt.legend(loc=2)
				plt.title(fits_image_filename)
				plt.xlabel("Wavelength nm")
				plt.ylabel("Normalized Intensity")
				os.chdir(r"/home/dtyler/Desktop/DocumentsDT/outputs/images_"+fits_image_filename)
				plt.savefig(fits_image_filename[:-4]+"_model"+str(icorr+1)+".png")
				os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs")
				plt.close()
				max_outlier = int(np.argmax(abs(mdl(d.x)-y)))
		#Break loop if residual is ever larger than previous iteration#Save best versions
		
		
				if (res.statval > best_res):
					f = best_fit
					mdl = bmdl
					res = bres
					break
				else:
					best_res = res.statval
					best_fit = f
					bmdl = mdl
					bres = best_fit.fit()
				
		#############################################
		
		
		mdl2= -1.0*(bmdl-1.0)*norm_factor+min(y_raw)
		plt.plot(w,p)
		plt.plot(x, mdl2(d.x))
		pylab.title("He 1083 nm")
		pylab.xlabel("Wavelength nm")
		pylab.ylabel("Intensity")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/outputs/images_"+fits_image_filename)
		plt.savefig(fits_image_filename[:-4]+"_fit.png")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs")
		
		
		d2 = Data1D('He 1083 nm',x,y_raw,staterror = sd*norm_factor)
		mplot2 = ModelPlot()
		mplot2.prepare(d2, mdl2)
		dplot2 = DataPlot()
		dplot2.prepare(d2)
		mplot2.overplot()
		
		#apply actual Fit
		f2 = Fit(d2, mdl2, stat=ustat, method=opt)
		res2 = f2.fit()
		fplot2 = FitPlot()
		mplot2.prepare(d2, mdl2)
		fplot2.prepare(dplot2,mplot2)
		fplot2.plot()
		pylab.xlabel("Wavelength nm")
		pylab.ylabel("Intensity")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/outputs/images_"+fits_image_filename)
		plt.savefig(fits_image_filename[:-4]+"_fit2.png")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs")
		
		# Fit continuum
		"""xc = np.array([mplot2.x[1], mplot2.x[-1]])
		yc = np.array([mplot2.y[0], mplot2.y[-1]])
		dc = Data1D("continuum", xc,yc)
		
		cont = Polynom1D()
		cont.c2.thaw()
		cf = Fit(dc,cont, stat=LeastSq(), method = NelderMead())
		cres = cf.fit()
		cfplot = FitPlot()
		cmplot = ModelPlot()
		cdplot = DataPlot()
		cmplot.prepare(dc,cont)
		cfplot = FitPlot()
		cfplot.prepare(cmplot,mplot2)
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs/outputs/images_"+fits_image_filename)
		plt.savefig(fits_image_filename[:-4]+"_continuum.png")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs")"""
		
		#Normalize model to continuum
		leftrange = len(mplot2.y)/(2)
		left = int(round(leftrange))
		rightrange = len(mplot2.y)/(2)
		right = int(round(rightrange))
		
		
		### test to see if eq calc knows only to do filled in area
		
		
		
		y_cont = np.linspace(max(mplot2.y[:left]), max(mplot2.y[right:]), num= len(mplot2.x))
		y_model = mplot2.y
		y_cn = y_model/y_cont
		dcontplot = DataPlot()
		dcont = Data1D("normalized", mplot.x, y_cn)
		dcontplot.prepare(dcont)
		dcontplot.plot()
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/outputs/images_"+fits_image_filename)
		plt.savefig(fits_image_filename[:-4]+"norm_continuum.png")
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs")
		
		
		
		
		
		###################################
		###  calculate equivalent width
		spectrum = Spectrum1D(flux= y_cn*u.dimensionless_unscaled, spectral_axis=mplot.x*u.AA)
		eqw = equivalent_width(spectrum)
		
		
		"""print("EW", eqw.value) 
		print('norm factor', norm_factor)
		for par in bmdl.pars:
			print(par.fullname, par.val)"""
		
		### print output to file
		
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Analysis/eq_widths")
		
		fil = open(fits_image_filename[:-4]+".txt",'w')
		fil.write('eqw'+'\t'+str(eqw.value)+'\n')	
		fil.write('norm'+'\t'+str(norm_factor)+'\n')
		for par in bmdl.pars:
			fil.write(par.fullname +'\t'+ str(par.val)+'\n')	
		fil.close()
		os.chdir(r"/home/dtyler/Desktop/DocumentsDT/Programs")
