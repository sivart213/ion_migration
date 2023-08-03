# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:42:38 2021

@author: j2cle
"""
#%%
import numpy as np
import pandas as pd
import General_functions as gf
import matplotlib.pyplot as plt
# import inspect
from matplotlib import ticker
from scipy.special import erfc
from scipy import optimize
# from scipy.io import savemat
# import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
# from matplotlib.ticker import ScalarFormatter
# import matplotlib as mpl
from scipy.optimize import curve_fit,fsolve,minimize
from functools import partial

def arrh(T,pre_fac,E_A):
    return pre_fac*np.exp(E_A/(gf.KB_EV*T))

def c_ratio_np(depth,thick,temp,e_app,time,diff,mob):
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (1/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))

                    
def find_tauc_np_cratio_3d(time,depth,temp,e_app,d_0,e_a, target=0.08, thick = 450e-4):
    diff = d_0*np.exp(-e_a/(gf.KB_EV*temp))
    mob = diff/(gf.KB_EV*temp)
    if depth == 0:
        depth = (2*np.sqrt(diff * time)) + mob * e_app * time
    if thick == 0:
        thick = (2*np.sqrt(diff * time)) + mob * e_app * time
    
    ratio = c_ratio_np(depth,thick,temp,e_app,time,diff,mob)
    if isinstance(ratio,float):
        if ratio <  1e-10:
            ratio = 1
    else:
        for n in range(len(ratio)):
            if ratio[n] <  1e-10:
                ratio[n] = 1
            
    return (ratio-target)**2

def find_tauc_np_cratio_3d_alt(time,depth,temp,e_app,d_0,e_a, target=0.08, thick = 450e-4):
    diff = d_0*np.exp(-e_a/(gf.KB_EV*temp))
    mob = diff/(gf.KB_EV*temp)
    if depth == 0:
        depth = (2*np.sqrt(diff * time)) + mob * e_app * time
    if thick == 0:
        thick = (2*np.sqrt(diff * time)) + mob * e_app * time
    
    ratio = c_ratio_np(depth,thick,temp,e_app,time,diff,mob)
            
    return ratio
    # return abs(c_ratio_np(depth,thick,temp,e_app,time,diff,mob)-target)



def limitcontour(x,y,z, xlim=None, ylim=None, **kwargs):
    
    if len(x.shape)==1:
        xx,yy = np.meshgrid(x,y)
    else:
        xx,yy = x,y
        
    xmask = np.ones(x.shape).astype(bool)
    ymask = np.ones(y.shape).astype(bool)
    zmask = np.ones(z.shape).astype(bool)
    
    if xlim:
        xmask = xmask & (x>=xlim[0]) & (x<=xlim[1])
        zmask = zmask & (xx>=xlim[0]) & (xx<=xlim[1])

    if ylim:
        ymask = ymask & (y>=ylim[0]) & (y<=ylim[1])
        zmask = zmask & (yy>=ylim[0]) & (yy<=ylim[1])

    xm = np.ma.masked_where(~xmask , x)
    ym = np.ma.masked_where(~ymask , y)
    zm = np.ma.masked_where(~zmask , z)
        
    return xm,ym,zm


def plotcont6(x,y,z,rho1,rho2,v1,v2,v3,name='',x_corr=0,y_corr=1,xname='Temp',yname='E',ind=0,levels=50,templim=[25,85]):
    
    z=z/(3600*24)
    
    if x_corr == 273.15:
        xunit = ' [C]'
    else:
        xunit = ' [K]'
    if y_corr == 1e6:
        yunit = ' [MV/cm]'
    else:
        yunit = ' [V/cm]'
        
    if yname == 'V app':
        yunit = ' [V]'
    

    
    fig, ax = plt.subplots()
    csa = ax.contourf(x-x_corr,y/y_corr,z,np.logspace(np.log10(z.min()),np.log10(z.max()), levels), locator = ticker.LogLocator(),  cmap='gist_heat')
    
    ax.set_xlabel(xname+xunit)
    ax.set_xlim(templim[0],templim[1])
    ax.set_ylabel(yname+yunit)
    ax.set_yscale('log')
    ax.set_title(name)
    
    cbar = fig.colorbar(csa)
    cbar.locator = ticker.LogLocator(10)
    cbar.set_ticks(cbar.locator.tick_values(z.min(), z.max()))
    cbar.minorticks_off()
    cbar.set_label('Breakthrough [Days]')
    
   
    xx,yy,zz = limitcontour(x-x_corr,y/y_corr,z,xlim=templim)
    
    visual_levels = [1, 4, 7, 30, 365, 365*10]
    lv_lbls = ['1 d', '4 d', '1 w', '1 mo', '1 yr', '10 yr']
    ax = plt.gca()
    csb = ax.contour(xx,yy,zz,visual_levels, colors='w',locator=ticker.LogLocator(),linestyles='--',norm=LogNorm(),linewidths=1.25)
    csb.levels = lv_lbls

    ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=False)
        
    b1 = rho1*v1/y_corr
    b2 = rho1*v2/y_corr
    b3 = rho1*v3/y_corr
    
    points = [(30,b1[gf.find_nearest(x-x_corr, 30)]),(45,b2[gf.find_nearest(x-x_corr, 45)]),(60,b3[gf.find_nearest(x-x_corr, 60)])]
    v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # ax = plt.gca()
    ax.plot(x-x_corr,b1,x-x_corr,b2,x-x_corr,b3, color='k',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],textcoords='offset points',xytext=(0,40),ha='center',arrowprops=dict(facecolor='black', shrink=0.001),
    #         horizontalalignment='right', verticalalignment='bottom')
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="gray"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
    # c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr
    
    # points = [(35,c1[gf.find_nearest(x-x_corr, 35)]),(50,c2[gf.find_nearest(x-x_corr, 50)]),(65,c3[gf.find_nearest(x-x_corr, 65)])]
    # v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # # ax = plt.gca()
    # ax.plot(x-x_corr,c1,x-x_corr,c2,x-x_corr,c3, color='b',linestyle='--',linewidth=1.25)
    # for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
    #         bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
    #         xytext=(0,-25), textcoords='offset points', ha='center',
    #         arrowprops=dict(arrowstyle="->"))
    c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr
    
    points = [(35,c1[gf.find_nearest(x-x_corr, 35)])]
    v_lbls = [str(v1)+' V (Boro)']
    # ax = plt.gca()
    ax.plot(x-x_corr,c1, color='b',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
    return

def plotcont8(x,y,z,rho1,rho2,v1,v2,v3,name='',x_corr=0,y_corr=1,xname='Temperature',yname='E',ind=0,levels=50,templim=[25,85]):
    
    z=z/(3600*24)
    
    if x_corr == 273.15:
        xunit = ' ($^o$C)'
    else:
        xunit = ' (K)'
    if y_corr == 1e6:
        yunit = ' (MV/cm)'
    else:
        yunit = ' (V/cm)'
        
    if yname == 'V app':
        yunit = ' (V)'
    
    
    
    fig, ax = plt.subplots()
    csa = ax.contourf(x-x_corr,y/y_corr,z,np.logspace(np.log10(z.min()),np.log10(z.max()), levels), locator = ticker.LogLocator(),  cmap='gist_heat')
    
    ax.set_xlabel(xname+xunit, fontname='Arial', fontsize=18, fontweight='bold')
    ax.set_xlim(templim[0],templim[1])
    ax.set_ylabel(yname+yunit, fontname='Arial', fontsize=18, fontweight='bold')
    ax.set_yscale('log')
    # ax.set_title(name)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)
    
    cbar = fig.colorbar(csa)
    cbar.locator = ticker.LogLocator(10)
    cbar.set_ticks(cbar.locator.tick_values(z.min(), z.max()))
    cbar.minorticks_off()
    cbar.set_label('Breakthrough (Days)', fontname='Arial', fontsize=18, fontweight='bold')
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontweight('bold')
        tick.set_fontsize(12)
   
    xx,yy,zz = limitcontour(x-x_corr,y/y_corr,z,xlim=templim)
    
    visual_levels = [1, 4, 7, 30, 365, 365*10]
    lv_lbls = ['1 d', '4 d', '1 w', '1 mo', '1 yr', '10 yr']
    ax = plt.gca()
    csb = ax.contour(xx,yy,zz,visual_levels, colors='w',locator=ticker.LogLocator(),linestyles='--',norm=LogNorm(),linewidths=1.25)
    csb.levels = lv_lbls

    ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=False)
    # ax.clabeltext(fontname='Arial')
        
    b1 = rho1*v1/y_corr
    b2 = rho1*v2/y_corr
    b3 = rho1*v3/y_corr
    
    points = [(35,b1[gf.find_nearest(x-x_corr, 35)]),(50,b2[gf.find_nearest(x-x_corr, 50)]),(65,b3[gf.find_nearest(x-x_corr, 65)])]
    v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # ax = plt.gca()
    ax.plot(x-x_corr,b1,x-x_corr,b2,x-x_corr,b3, color='k',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],textcoords='offset points',xytext=(0,40),ha='center',arrowprops=dict(facecolor='black', shrink=0.001),
    #         horizontalalignment='right', verticalalignment='bottom')
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="gray"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"), fontname='Arial', fontsize=12, fontweight='bold')
    # c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr
    
    # points = [(35,c1[gf.find_nearest(x-x_corr, 35)]),(50,c2[gf.find_nearest(x-x_corr, 50)]),(65,c3[gf.find_nearest(x-x_corr, 65)])]
    # v_lbls = [str(v1)+' V',str(v2)+' V',str(v3)+' V']
    # # ax = plt.gca()
    # ax.plot(x-x_corr,c1,x-x_corr,c2,x-x_corr,c3, color='b',linestyle='--',linewidth=1.25)
    # for i in range(len(v_lbls)):
    #     ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
    #         bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
    #         xytext=(0,-25), textcoords='offset points', ha='center',
    #         arrowprops=dict(arrowstyle="->"))
    c1 = rho2*v1/y_corr
    # c2 = rho2*v2/y_corr
    # c3 = rho2*v3/y_corr
    
    points = [(35,c1[gf.find_nearest(x-x_corr, 35)])]
    v_lbls = [str(v1)+' V (Boro)']
    # ax = plt.gca()
    ax.plot(x-x_corr,c1, color='b',linestyle='--',linewidth=1.25)
    for i in range(len(v_lbls)):
        ax.annotate(v_lbls[i],xy=points[i],  xycoords='data',
            bbox=dict(boxstyle="round", fc="0.9", ec="blue"),
            xytext=(0,25), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))
    return

class Module:
    def __init__(self):
        
        self.glass = 'Soda'
        self.glass_t = gf.tocm(0.125,'in')
        self.database(self.glass)
        
        self.enc = 'EVA'
        self.enc_t = gf.tocm(450,'um')
        self.database(self.enc)
        
        self.diff_range = 'Ave'
        self.database(self.diff_range)
        
        self.cell_t = 180e-4
        self.sinx_t = gf.tocm(80,'nm')
        self.rho_sinx = 1e13
         
        self.array_size = 100
        self.T_1D = np.linspace(20+273.15,100+273.15,self.array_size)
        self.E_1D = np.logspace(3,6,self.array_size)
        self.T_2D,self.E_2D = np.meshgrid(self.T_1D, self.E_1D)
        
        self.resistivity()
        
        self.bbt_arr = {}
        self.bbt_df = {}
        
    def load_data(self,params):
        self.glass = params['Glass']
        self.glass_t = params['Glass Thickness']
        self.database(self.glass)

        self.enc = params['Encap']
        self.enc_t = params['Encap Thickness']
        self.database(self.enc)
        
        self.cell_t=params['Cell Thickness']
        
        self.sinx_t=params['SiNx Thickness']
        self.rho_sinx=params['SiNx Resistivity']
        
        self.diff_range = params['Diffusivity']
        self.database(self.diff_range)
        
        self.resistivity()
        
    def database(self,material):
        if material.lower() == 'boro':
            self.glass_pre = 0.04495210144020945 # std data
            self.glass_ea = 0.9702835437735396 # std data
        if material.lower() == 'soda':
            self.glass_pre = 5644.5501772317775 # std data
            self.glass_ea = 0.3590103059601377 # std data
        if material.lower() == 'boro alt':
            self.glass_pre = 0.6580761914650776 # my data
            self.glass_ea = 0.8507365332956724 # my data
        if material.lower() == 'soda alt':
            self.glass_pre = 0.023546001890093236 # my data
            self.glass_ea = 0.9471251012618027 # my data
        if material.lower() == 'eva':
            self.enc_pre = 19863.639619529386
            self.enc_ea = 0.6319537614631568
        if material.lower() == 'ave':
            self.d_0 = 5.75e-7 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.578 # eV # mims=0.809, TOF=0.28
        if material.lower() == 'max':
            self.d_0 = 2.902e-5 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.623 # eV # mims=0.809, TOF=0.28
        if material.lower() == 'min':
            self.d_0 = 1.140e-8 # cm/s # mims=0.021, TOF=3.41e-11
            self.e_a = 0.533 # eV # mims=0.809, TOF=0.2800
    
    def resistivity(self):
        self.rho_glass = arrh(self.T_1D,self.glass_pre,self.glass_ea)
        self.rho_enc = arrh(self.T_1D,self.enc_pre,self.enc_ea)
        
        self.rho_system = self.rho_enc/(self.rho_glass*self.glass_t+self.rho_enc*self.enc_t+self.rho_sinx*self.sinx_t)
        
    def bbt(self,ident,target=1e16/5e21,diff_depth=0):
        if diff_depth == 0:
            diff_depth = self.enc_t

        time_bound_low = np.full((self.array_size,self.array_size),0.0)
        time_bound_high = np.full((self.array_size,self.array_size),0.0)
        cont = 0
        changed_logic=np.full((self.array_size,self.array_size),True)
        
        test1 = find_tauc_np_cratio_3d_alt(np.full((self.array_size,self.array_size),1),diff_depth,self.T_2D,self.E_2D,self.d_0,self.e_a,target,self.enc_t)
        secs = 0
        while cont == 0:
            if secs <= 60: # if less than a min
                delta=1 # inc by sec
            elif secs <= 3600*24: # if less than a hour
                delta=float(60) # inc by min
            elif secs <= 3600*24*30: # if less than a month
                delta=float(3600) # inc by hour
            # elif secs <= 3600*24*7: # if less than a week
            #     delta=float(3600*24) # inc by day
            elif secs <= 3600*24*365*.5: # if less than half a year
                delta=float(3600*24) # inc by day
            elif secs <= 3600*24*365: # if less than a year
                delta=float(3600*24*7) # inc by week
            elif secs <= 3600*24*365*100: # if less than 100 year
                delta=float(3600*24*14) # inc by 2 week
            elif secs > 3600*24*365*250: # if less than 250 year
                delta=float(3600*24*30) # inc by month
            elif secs > 3600*24*365*500: # if less than 500 year
                delta=float(3600*24*365) # inc by year
            elif secs > 3600*24*365*1000:
                delta=float(secs*2) # double each round
            
            secs+=delta 
            
            # print(secs)
            test2 = find_tauc_np_cratio_3d_alt(np.full((self.array_size,self.array_size),secs),diff_depth,self.T_2D,self.E_2D,self.d_0,self.e_a,target,self.enc_t)
            time_bound_low[test1==0]=float(secs)
            
            comb_logic = (test2-test1)<=0.1
            test_logic = test2>=0.5
            changed_logic[comb_logic*test_logic]=False
            
            time_bound_high[((test2-test1)>0)*changed_logic]=float(secs)
            time_bound_low[time_bound_low>=time_bound_high] = secs-delta
            
            test1=test2

            if np.max(time_bound_high) < secs and np.min(time_bound_high) != 0:
                cont=1

        self.bbt_arr[ident] = np.array([[optimize.minimize_scalar(find_tauc_np_cratio_3d,bounds=(time_bound_low[l_E][l_T],time_bound_high[l_E][l_T]),args=(diff_depth,self.T_1D[l_T],self.E_1D[l_E],self.d_0,self.e_a,target,self.enc_t),method='bounded').x 
                                                                        for l_T in range(self.array_size)] for l_E in range(self.array_size)])

        self.bbt_df[ident] = pd.DataFrame(self.bbt_arr[ident],index=self.E_1D,columns=self.T_1D)
        
    def find_info(self,ident,temp,volt):
        if ident not in self.bbt_arr:
            self.bbt(ident,1)
        
        temp_ind = gf.find_nearest(self.T_1D,temp+273.15)
        self.rho_found = self.rho_system[temp_ind]
        self.E_found = self.rho_found*volt
        E_ind = gf.find_nearest(self.E_1D,self.rho_found*volt) 

        self.btt_found = self.bbt_arr[ident][E_ind,temp_ind]


#%%
mod_soda=Module()

mod_boro=Module()
mod_boro.glass = 'Boro'
mod_boro.database('Boro')
mod_boro.resistivity()

#%%
mod_soda.bbt('run1',1e16/5e21)

#%%
plotcont6(mod_soda.T_1D,mod_soda.E_1D,mod_soda.bbt_arr['run1'],mod_soda.rho_system,mod_boro.rho_system,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
plotcont8(mod_soda.T_1D,mod_soda.E_1D,mod_soda.bbt_arr['run1'],mod_soda.rho_system,mod_boro.rho_system,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])


# plotcont6(arr_T_1D,arr_E_1D,arr_bt_e_vs_t,arr_rho_system_1D_soda,arr_rho_system_1D_boro,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])
# plotcont8(arr_T_1D,arr_E_1D,arr_bt_e_vs_t,arr_rho_system_1D_soda,arr_rho_system_1D_boro,1500,1000,600,'Variance in Breakthrough Time for E & T',273.15,1e6,levels=25,templim=[25,100])


# %%

var_temp_ind = gf.find_nearest(arr_T_1D,85+273.15)
var_v = 1500
var_rho = arr_rho_system_1D_soda[var_temp_ind]
var_E_ind= gf.find_nearest(arr_E_1D,var_rho*var_v) 

var_btt = arr_bt_e_vs_t[var_E_ind,var_temp_ind]

print('Time = ',var_btt/(3600*24))
#%%

var_rho_boro = arrh(25+273.15,boro_prefac,boro_ae)
var_rho_soda = arrh(25+273.15,soda_prefac,soda_ae)
var_rho_EVA = arrh(25+273.15,EVA_prefac,EVA_ae)

#%%

mod_soda_mims = Module()
mod_soda_mims.sinx_t = 1e-30
mod_soda_mims.resistivity()



mod_soda_mims.bbt(ident='run1',target=1e16/5e19,diff_depth=0.0002)

#%%
totals=0
mod_soda_mims.find_info('run1',60,600)
totals+=mod_soda_mims.btt_found
print('60 Low', mod_soda_mims.btt_found/3600)
mod_soda_mims.find_info('run1',70,600)
totals+=mod_soda_mims.btt_found
print('70 Low', mod_soda_mims.btt_found/3600)
mod_soda_mims.find_info('run1',80,600)
totals+=mod_soda_mims.btt_found
print('80 Low', mod_soda_mims.btt_found/3600)
mod_soda_mims.find_info('run1',90,600)
totals+=mod_soda_mims.btt_found
print('90 Low', mod_soda_mims.btt_found/3600)
mod_soda_mims.find_info('run1',60,1500)
totals+=mod_soda_mims.btt_found
print('60 High', mod_soda_mims.btt_found/3600)
mod_soda_mims.find_info('run1',70,1500)
totals+=mod_soda_mims.btt_found
print('70 High', mod_soda_mims.btt_found/3600)
mod_soda_mims.find_info('run1',80,1500)
totals+=mod_soda_mims.btt_found
print('80 High', mod_soda_mims.btt_found/3600)
mod_soda_mims.find_info('run1',90,1500)
totals+=mod_soda_mims.btt_found
print('90 High', mod_soda_mims.btt_found/3600)
print('Totals',totals/(3600*24))
