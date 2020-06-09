import argparse
import numpy as np
import os
import pandas as pd
import datetime
from datetime import timedelta
from itertools import islice
from scipy.signal import correlate
import matplotlib.pyplot as plt
import time
import h5py
from gwpy.segments import DataQualityFlag


def dataframe_clean(dataframe,column,snr=8.0,conf=0.9):
	dataframe.drop_duplicates(str(column),inplace=True)
	dataframe.sort_values(by=str(column),inplace=True)
	dataframe = dataframe[dataframe.confidence>=conf]
	dataframe = dataframe[dataframe.snr>=snr]
	dataframe = dataframe.reset_index(drop=True)
	
	return dataframe

def segs_and_dur(starttime,endtime,ifo):
	segs =  DataQualityFlag.query('{0}:DMT-ANALYSIS_READY:1'.format(ifo),starttime,endtime).active
	dur =   int(np.sum([i.end-i.start for i in segs]))

	return segs, dur




def remove_highblrms(motion_file,value,num):
    
	ar1 = motion_file[(motion_file.value>value) & (motion_file.value<value+100)].times.value
	ar2 = motion_file.times.value
	ar1a = ar1[num:]
    
	indices = np.nonzero([x in ar1a for x in ar2])
	motion_file.value[indices] = np.nan
    
	if num==0:
		ar1a = ar1[:]
		indices = np.nonzero([x in ar1a for x in ar2])
		motion_file.value[indices] = np.nan
    
	return list(ar1a)

def removed_times(motion_file,motion_file_list):
	times=[]
	for i in motion_file_list:
		times.append(remove_highblrms(motion_file,i[0],i[1]))

	times_flat = [i for j in times for i in j]
	times_segs = SegmentList()
	for i in times_flat:
    		times_segs.append(Segment(i-60,i))

	times_segsdur = np.sum([i.end - i.start for i in times_segs])
	times_segs = times_segs.coalesce()
	
	return times_segs



def compare_gds_l2(starttime_gds,starttime_l2,dur_gds,fft_gds,dur_l2,fft_l2,correction_factor=1.35,ref_c1=0.001,ref_c2=9e-5,ref_c3=9e-6):
	
	overlapgds = fft_gds/2
	windowgds = dur_gds/2
	overlapl2 = fft_l2/2
	fac = 12.56/1.064
	
	t = TimeSeries.fetch('L1:GDS-CALIB_STRAIN_CLEAN',starttime_gds-windowgds,starttime_gds+windowgds).detrend().asd(fft_gds,overlapgds)	   
	t_etmx = TimeSeries.fetch('L1:SUS-ETMX_L2_WIT_L_DQ',starttime_l2,starttime_l2+dur_l2).detrend()*correction_factor
	

	t_etmxsinsubzero = TimeSeries(np.sin((t_etmx*0.5*fac*t_etmx.unit).value), t0 = t_etmx.t0.value, dt = t_etmx.dt.value).detrend()
	t_etmxsin = TimeSeries(np.sin((t_etmx*1*fac*t_etmx.unit).value), t0 = t_etmx.t0.value, dt = t_etmx.dt.value).detrend()
	t_etmxsin1 = TimeSeries(np.sin((t_etmx*2*fac*t_etmx.unit).value), t0 = t_etmx.t0.value, dt = t_etmx.dt.value).detrend()
	t_etmxsin2 = TimeSeries(np.sin((t_etmx*fac*3*t_etmx.unit).value), t0 = t_etmx.t0.value, dt = t_etmx.dt.value).detrend()


	### Constants
	Tr=4e-6    # transmissivity
	lamda=1e-6   # wavelength
	L = 4000    # arm length
	#ref_c = 1e-3   # fraction of light incident on and reflected back from ESD's
	finesse_fac = np.sqrt(2*443*np.pi)   ### finnesse factor

	### calculating the noise in darm due to sus_motion
	h_tetmx1 = (1/np.pi)*Tr*lamda*ref_c1*(0.125/L)*t_etmxsin.asd(fft_l2,overlapl2)
	h_tetmx2 = (1/np.pi)*Tr*lamda*ref_c2*(0.125/L)*t_etmxsin1.asd(fft_l2,overlapl2)
	h_tetmx3 = (1/np.pi)*Tr*lamda*ref_c3*(0.125/L)*t_etmxsin2.asd(fft_l2,overlapl2)

	h_tetmxtot = h_tetmx1+h_tetmx2+h_tetmx3

	### Comparing the noise in darm to noise in darm due to sus_motion
	plt.figure(figsize=(16,8))
	plt.plot(t,label='h(t)'.format(starttime_gds))
	plt.plot(h_tetmxtot,label='from l2 motion'.format(starttime_l2))
	plt.xlim(5,150)
	plt.xticks(list(np.arange(10,150,10)),list(np.arange(10,150,10)),fontsize=21)
	plt.yscale("log")
	plt.ylim(0.2e-23,1e-17)
	plt.yticks(fontsize=21)
	plt.ylabel('GW Amplitude Spectral Density [strain / $\sqrt{Hz}$]',fontsize=20)
	plt.xlabel('Frequency [Hz]',fontsize=20)
	plt.legend(loc='upper right',fontsize=20)
	plt.title("L2 stage motion overlaid on h(t) spectra for scattering at {0}".format(starttime_gds),fontsize=22)
	plt.show()


	return


def plot_asdgds(gpstime1,dur1,gpstime2,dur2,fft):
    
    t1 = to_gps(gpstime1)
    t2 = to_gps(gpstime2)
    
    overlap = fft/2
    
    gds = TimeSeries.fetch('L1:GDS-CALIB_STRAIN_CLEAN',t1,t1+dur1).detrend().asd(fft,overlap)
    gds_ref = TimeSeries.fetch('L1:GDS-CALIB_STRAIN_CLEAN',t2,t2+dur2).detrend().asd(fft,overlap)
    
    plt.figure(figsize=(16,8))
    plt.plot(gds,label='{0}'.format(gpstime1))
    plt.plot(gds_ref,label='{0}'.format(gpstime2))
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(5,1000)
    plt.grid(True,which='both')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(10e-25,1e-17)
    
    #plt.xticks()
    plt.ylabel('ASD  [1/$\sqrt{Hz}$]',fontsize=20)
    plt.xlabel('Frequency (Hz)',fontsize=20)
    plt.title("Spectrum:L1:GDS-CALIB_STRAIN_CLEAN",fontsize=20)
    plt.legend(fontsize=18)
    plt.show()


def compare_transmons(gpstime,dur,fft):
    
    windowgps = dur/2
    overlap = fft/2
    
    
    
    t = TimeSeries.fetch('L1:ASC-X_TR_A_NSUM_OUT_DQ',gpstime-windowgps,gpstime+windowgps)
    tgds = TimeSeries.fetch('L1:GDS-CALIB_STRAIN_CLEAN',gpstime-10,gpstime+10).detrend()
    tnonscat = TimeSeries.fetch('L1:ASC-X_TR_A_NSUM_OUT_DQ',to_gps('2019-06-23 03:00:00')-windowgps,to_gps('2019-06-23 03:00:00')+windowgps)
    tgdsnonscat = TimeSeries.fetch('L1:GDS-CALIB_STRAIN_CLEAN',to_gps('2019-06-23 03:00:00')-10,to_gps('2019-06-23 03:00:00')+10).detrend()
    
    tasd = t.asd(fft,overlap)/t.mean()
    tgdsasd = tgds.asd(5,2.5)
    tnonasd = tnonscat.asd(fft,overlap)/tnonscat.mean()
    tgdsnonscatasd = tgdsnonscat.asd(5,2.5)
    
   # plt.figure(figsize=(16,8))
    plt.plot(tasd,label='scattering')
    plt.xlim(5,50)
    plt.plot(tnonasd,label='non scattering June 23')
    plt.xlim(5,50)
    plt.xticks([i for i in [5,10,20,30,40,50]],[i for i in [5,10,20,30,40,50]],fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.ylabel('RIN')
    plt.title("X end PD  gpstime {1}".format(fft,gpstime))
    plt.legend(fontsize=12)
    plt.show()
    
   # plt.figure(figsize=(16,8))
    plt.plot(tgdsasd,label='scattering')
    plt.xlim(5,50)
    plt.plot(tgdsnonscatasd,label='non scattering June 23')
    plt.xlim(5,50)
    plt.xticks([i for i in [5,10,20,30,40,50]],[i for i in [5,10,20,30,40,50]],fontsize=12)
    plt.yticks(fontsize=12)
    #plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.2e-23,1e-17)
    plt.ylabel('GW ASD [strain / $\sqrt{Hz}$]')
    plt.xlabel('Frequency')
    plt.title(" h(t) gpstime {3}".format(5,2.5,20,gpstime))
    plt.legend(fontsize=12)
    plt.show()
    
    
    
    return


def trigsinosems(starttime,cutofft,date):
    """For a given start and endtime, this function finds the osem scattering segments and then vetoes the h(t) triggers with snr>15 and peakf between 10 and 60.
    It calculates the deadtime, efficiency of the scattering veto by osem scattering segments.."""
    
    gpstime = to_gps(date)
    filename = 'files/L1-SCATTERING_SEGMENTS_15_HZ-'+str(gpstime)+'-86400.h5'
    f = h5py.File(filename,'r')
    ### Reading all the segments for this day.
    a = []
    b = []
    c = []
    for i in list(f['segments']):
        if f['segments'][i]['active'].shape[0] !=0:
            a.append(f['segments'][i]['active'].value)
    for i in range(len(a)):
        b+= list(a[i])
    for i in range(len(b)):
        c.append(Segment(b[i][0]+10**-9*b[i][1],b[i][2]+10**-9*b[i][3]))
    len(c)
    
    ### here we only take those segs which are within the cutoff time.
    csegs = []
    for i in range(len(c)):
        if c[i][0] < cutofft and c[i][1] < cutofft:
            csegs.append(Segment(c[i][0],c[i][1]))
            

    
   # print(len(c)) # number of segments for the whole day.
   # print(len(csegs)) # number of segments within the cutoff time.
    
    ### Getting triggers
    cache = find_trigger_files('L1:GDS-CALIB_STRAIN', 'omicron',starttime,cutofft,ext = "h5")
    t = EventTable.read(cache,format = 'hdf5', path = 'triggers', 
                        columns = ['time','frequency','snr']).filter('snr>15','snr<200','frequency>10','frequency<60')
    
    ### Finding the triggers that are within the osems.
    trigsinosem = [i for i in t['time'] for j in csegs if i in j ]
    
    ## Efficiency
    eff = round((len(trigsinosem)*100)/len(t),2)
    
    ## Total duration of the segments.
    dur = []
    for i in csegs:
        dur.append(i[1]-i[0])
    totdur = np.sum(dur)
    
   # print(totdur)
    
    ### Deadtime
    deadtime = round((totdur*100)/(cutofft-starttime),2)
    
   # print("Total deadtime is {0}".format(deadtime))
   # print("Efficiency is {0}".format(eff))
    
    ### Efficiency over Deadtime
    eod = round(eff/deadtime,2)
    print("Efficiency over deadtime for {1} is {0}".format(eod,date))
    
    return eod


def trigsinsegs2(starttime,endtime,filename,thres):
    """For a given start and endtime, this function calculates the whitened blrms of the X end transmons and then vetoes the h(t) triggers with snr>15 and peakf between 10 and 60.
    It also calculates, the number of triggers vetoed by blrms that match with scattering triggers identified by gravityspy."""
    
    XTR = TimeSeries.fetch('L1:ASC-X_TR_B_NSUM_OUT_DQ',starttime,endtime)
    XTR = XTR.whiten(4,2).bandpass(4,10).rms(1)
    
    highxtr = XTR > np.mean(XTR) + thres*np.std(XTR)
    flag = highxtr.to_dqflag(round=True)
    
    dur = []
    for i in range(len(flag.active)):
        dur.append(flag.active[i].end-flag.active[i].start)
        
    totaldur = np.sum(dur)
    
    cache = find_trigger_files('L1:GDS-CALIB_STRAIN', 'omicron',starttime,endtime,ext = "h5")
    t = EventTable.read(cache,format = 'hdf5', path = 'triggers', 
                        columns = ['time','frequency','snr']).filter('snr>15','snr<200','frequency>10','frequency<60')
    
    df = pd.read_csv(filename)
    df.drop_duplicates(["time"],inplace=True)
    df = df[(df.time>starttime) & (df.time<=endtime)]
    
    trigsinsegs=[i for i in t['time'] for j in flag.active if i in j]
    
    trigsingspy = []
    for i in trigsinsegs:
        for j in df["time"]:
            if abs(i-j)<0.5:
                trigsingspy.append(i)
    
    
    eff = round((100*len(trigsinsegs))/len(t),2)
    
    deadtime = round((totaldur*100)/(int(endtime)-int(starttime)),2)
    
    eff2 = round((100*len(trigsingspy))/len(trigsinsegs),2)
    
    eod = eff/deadtime
    
    
    return eod
