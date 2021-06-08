#! /usr/bin/env python3

import matplotlib 
import numpy as np
import pandas as pd
import os, sys, time, random, pickle
#import ipywidgets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection    
 
# use LaTeX, choose nice some looking fonts and tweak some settings
matplotlib.rc('font', family='serif')
matplotlib.rc('font', size=16)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('legend', numpoints=1)
matplotlib.rc('legend', handlelength=1.5)
matplotlib.rc('legend', frameon=True)
matplotlib.rc('xtick.major', pad=7)
matplotlib.rc('xtick', direction="in")
matplotlib.rc('ytick', direction="in")
matplotlib.rc('xtick', top = True)
matplotlib.rc('ytick', right =True )
matplotlib.rc('xtick.minor', pad=7)
matplotlib.rc('text', usetex=True)
# matplotlib.rc('text.latex', 
#               preamble=[r'\usepackage[T1]{fontenc}',
#                         r'\usepackage{amsmath}',
#                         r'\usepackage{txfonts}',
#                         r'\usepackage{textcomp}'])

matplotlib.rc('figure', figsize=(12, 9))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda:0')


def ptrac_analysis1(fn):
    """
    Carbon, Proton, Photoelectron, Compton
    """
    l_carbon=[]
    l_proton=[]
    l_photoelectric=[]
    l_compton = []
    h=1
    status = {"carbon" : False, "proton" : False, "electron" : False, "compton" : False}
    with open(os.path.join(rawdatafolder, fn)) as ff:
        for ln, line1 in enumerate(ff):
#             if ln > 2000:
#                 break
            if ln < 14:
                continue
            if ln == 14 :
                print(line1) 
            
            l1 = line1.split()
            if len(l1)< 7 :
                # start line of history
                if l1[1] == '1000' :
                    h+=1
                    continue
            else :
                line2 = next(ff)
                l2 = line2.split()
            
            ### carbon scattering
            if l1[2] == '6000' and l1[4] == '1':  
                l_carbon.append([float(l2[8]), int(l1[5]), h])
                
            ### proton production
            if status["proton"] and l1[4] == '9':  
                l_proton.append([float(l2[8]), int(l1[5]), h])     
                            
            if l1[0] == '2030' :
                status["proton"] = True
            else :
                status["proton"] = False
            
            ### electron production
            
            if status["electron"] and l1[4] == '3': 
                l_photoelectric.append([float(l2[8]), int(l1[5]), h])
 
            if l1[0] == '2011' and l1[4] == '2' :
                status["electron"] = True
            else :
                status["electron"] = False
            
            ### compton scattering
 
            if status["compton"] and l1[4] == '3':
                l_compton.append([float(l2[8]) , int(l1[5]), int(h)])       
            
            if l1[0] == '2012' and l1[4] == '2':
                status["compton"] = True
            else :
                status["compton"] = False
            
                
        df_carbon = pd.DataFrame({"time": [ll[0] for ll in l_carbon], 
                               "cell" : [ll[1] for ll in l_carbon], 
                               "Nhist": [ll[2] for ll in l_carbon]})
        df_proton = pd.DataFrame({"time": [ll[0] for ll in l_proton], 
                               "cell" : [ll[1] for ll in l_proton], 
                               "Nhist": [ll[2] for ll in l_proton]})
        df_photoelectric = pd.DataFrame({"time": [ll[0] for ll in l_photoelectric], 
                               "cell" : [ll[1] for ll in l_photoelectric], 
                               "Nhist": [ll[2] for ll in l_photoelectric]})
        df_compton = pd.DataFrame({"time": [ll[0] for ll in l_compton], 
                               "cell" : [ll[1] for ll in l_compton], 
                               "Nhist": [ll[2] for ll in l_compton]})
        
        
        
        return df_carbon, df_proton, df_photoelectric, df_compton

def ptrac_analysis2(rawdatafolder, fn):
    """
    Neutron, photon(gamma)
    """
    
    l_neutron=[]
    l_photon=[]

    h=1
    status = {"carbon" : False, "proton" : False, "electron" : False, "compton" : False}
    with open(os.path.join(rawdatafolder, fn)) as ff:
        for ln, line1 in enumerate(ff):
#             if ln > 2000:
#                 break
            if ln < 14:
                continue
            if ln == 14 :
                print(line1) 
            
            l1 = line1.split()
            if len(l1)< 7 :
                # start line of history
                if l1[1] == '1000' :
                    h+=1
                    continue
            else :
                line2 = next(ff)
                l2 = line2.split()
            
            ### carbon scattering
            if l1[2] == '6000' and l1[4] == '1':  
                l_neutron.append([float(l2[8]), int(l1[5]), h, "C"])
                
            ### proton production
            if status["proton"] and l1[4] == '9':  
                l_neutron.append([float(l2[8]), int(l1[5]), h, "P"])     
                            
            if l1[0] == '2030' :
                status["proton"] = True
            else :
                status["proton"] = False
            
            ### electron production
            
            if status["electron"] and l1[4] == '3': 
                l_photon.append([float(l2[8]), int(l1[5]), h, "E"])
 
            if l1[0] == '2011' and l1[4] == '2' :
                status["electron"] = True
            else :
                status["electron"] = False
            
            ### compton scattering
 
            if status["compton"] and l1[4] == '3':
                l_photon.append([float(l2[8]) , int(l1[5]), int(h), "O"])       
            
            if l1[0] == '2012' and l1[4] == '2':
                status["compton"] = True
            else :
                status["compton"] = False
            
                
        df_neutron = pd.DataFrame({"time": [ll[0] for ll in l_neutron], 
                               "cell" : [ll[1] for ll in l_neutron], 
                               "Nhist": [ll[2] for ll in l_neutron], 
                               "origin" : [ll[3] for ll in l_neutron]})
        df_photon = pd.DataFrame({"time": [ll[0] for ll in l_photon], 
                               "cell" : [ll[1] for ll in l_photon], 
                               "Nhist": [ll[2] for ll in l_photon],
                               "origin" : [ll[3] for ll in l_photon]})
        
        
        return df_neutron, df_photon

def ptrac_analysis_all_save(rawdatafolder, inid):
    analysisdata={}
    prefix0 ="C"+str(inid)
    for tid in range(1, 20):
        prefix = prefix0+"_"+str(tid)
        #print(prefix+".p")
        dfn, dfp = ptrac_analysis2(rawdatafolder, prefix+".p")
        analysisdata[prefix+"_neutron"]=dfn
        analysisdata[prefix+"_photon"]=dfp

    with open("./data/"+prefix0+"_analysis.pkl", 'wb') as f:
        pickle.dump(analysisdata, f)
    
def read_ptrac_analysis(pklfilepath):
    return pickle.load(open(pklfilepath, "rb"))

def ptrac_reduce(analysisdict):
    newdict = {}
    for key, df in analysisdict.items():
        newdict[key] = df.drop_duplicates(subset=["cell", "Nhist"], keep="first")
    return newdict

# def ptrac_analysys_reduce(rawdatafolder, inid, tid):
#     prefix="C"+str(inid)+"_"+str(tid)
#     fn = prefix+".p"
#     ln, lp = ptrac_analysis2(rawdatafolder, fn)
#     lnr=ptrac_reduce(ln)
#     lpr=ptrac_reduce(lp)
    
#     return (prefix, lnr, lpr)

# def ptrac_reduce_save(inid, tid):
#     prefix, lnr, lpr = ptrac_analysys_reduce(inid, tid)
#     lnr.to_pickle(os.path.join(pkldatafolder, prefix+"_neutron.pkl"))
#     lpr.to_pickle(os.path.join(pkldatafolder, prefix+"_photon.pkl"))

# def ptrac_analysis_reduce_all(rawdatafolder, num=1):
#     prefix="C"+str(num)+"_"
#     finalresult = {}

#     for i in range(1, 11):
#         fn = prefix+str(i)+".p"
#         print(fn+" start.")
#         ln, lp = ptrac_analysis2(rawdatafolder, fn)
#         finalresult[prefix+str(i)+"_neutron"]=ptrac_reduce(ln)
#         finalresult[prefix+str(i)+"_photon"]=ptrac_reduce(lp)

#     return finalresult


def ptrac_combine(reduceddict, do_sort = True, timeprocess = True, ):
    """
    전지혜씨의 process 중 combine + order를 한번에 처리.
    """
    neutronlist = []
    photonlist = []

    valid_cell = {121, 122, 123, 124, 125, 126, 127, 128, 129, 141, 142, 143, 
                221, 222, 223, 224, 225, 226, 227, 228, 229, 241, 242, 243, 
                321, 322, 323, 324, 325, 326, 327, 328, 329, 341, 342, 343} 

    for key, value in reduceddict.items():
        tt = key.split("_")
        tid, ptl = int(tt[-2]), tt[-1]

        newdf = value.copy()
        
        if timeprocess:
            newdf["time"]=value["time"]+100000*(tid-1)
        if ptl ==  "neutron":
            neutronlist.append(newdf)
        elif ptl == "photon":
            photonlist.append(newdf)
    
    dfn = pd.concat(neutronlist)
    dfp = pd.concat(photonlist)

    n_cell = set(dfn["cell"].unique())
    p_cell = set(dfp["cell"].unique())
    print(n_cell)
    print(p_cell)
    nn0, pp0 = n_cell- valid_cell, p_cell-valid_cell
    print(nn0)
    print(pp0)
    if len(nn0)>0 :
        for nn in nn0:
            dfn = dfn[dfn["cell"]!=nn]
    if len(pp0)>0 :
        for pp in pp0 :
            dfp = dfp[dfp["cell"]!=pp]

    if do_sort:
        dfn.sort_values("time", inplace=True, ignore_index=True)
        dfp.sort_values("time", inplace=True, ignore_index=True)
    return dfn, dfp

def ptrac_combine2(reduceddict):
    fnprefix = "C"+str(inid)
    _, _, filelist= list(os.walk(pkldatafolder))[0]
    files1 = [l for l in filelist if l.startswith("C"+str(inid))]
    files = [l for l in filelist if l.endswith("pkl")]
    neutrons={}
    photons={}
    for filename in files:
        tdf = pd.read_pickle(os.path.join(pkldatafolder, filename))
        kid = int(filename.split("_")[1])
        tdf["time"]=tdf["time"]+100000*(kid-1)
        if filename[:-4].endswith("neutron"):
            neutrons[filename[:-4]]=tdf
        elif filename[:-4].endswith("photon"):
            photons[filename[:-4]]=tdf
    
    dfn = pd.concat(list(neutrons.values()))
    dfp = pd.concat(list(photons.values()))
    
    valid_cell = {121, 122, 123, 124, 125, 126, 127, 128, 129, 141, 142, 143, 
                 221, 222, 223, 224, 225, 226, 227, 228, 229, 241, 242, 243, 
                 321, 322, 323, 324, 325, 326, 327, 328, 329, 341, 342, 343} 
    
    n_cell = set(dfn["cell"].unique())
    p_cell = set(dfn["cell"].unique())
    
    nn0, pp0 = n_cell- valid_cell, p_cell-valid_cell
    if len(nn0)>0 :
        for nn in nn0:
            dfn = dfn[dfn["cell"]!=nn]
    if len(pp0)>0 :
        for pp in pp0 :
            dfp = dfp[dfp["cell"]!=pp]
    
    dfn.sort_values("time", inplace=True, ignore_index=True)
    dfp.sort_values("time", inplace=True, ignore_index=True)
    return dfn, dfp


def ptrac_rossi(df):
    tdata = (df["time"].to_numpy()).copy()
    #tdata = tdata.astype(np.int16)
    print(tdata)
    result = []
    i=1
    while True :
        ts = tdata[i:]-tdata[:-i]
        if (ts<=100.).any():
            result.append(ts[ts<=100.])
            i+=1
        else :
            break
    return np.concatenate(result)
        
def jjh_rossi(df):
    tdata = (df["time"].to_numpy()).copy()
    tdata = tdata.astype(np.int32)
    #return tdata
    ll = tdata.shape[0]
    print(ll)
    result = []
    for i in range(0, ll-1):
        if i % 100 == 100 :
            print(i, "-th")
        for j in range(0, ll-i):
            try:
                rr = tdata[i+j]-tdata[i]
            except r :
                print(i+j, i, rr)
                print(r)
            
            if rr<=100:
                result.append(rr)

    return np.array(result)



def read_jjh(jjhdatafolder, inid=1, tid=None, stage=0):
    if stage == 0:
        strstg = "analysis"
    elif stage == 1:
        strstg = "reduced"
    if tid != None :
        fn0 = "C"+str(inid)+"_"+str(tid)
        fname_n = fn0+"_neutron_"+strstg+".o"
        fname_p = fn0+"_photon_"+strstg+".o"
        df_n = pd.read_csv(os.path.join(jjhdatafolder, fname_n), delimiter="\s+", header=None)

        df_n.columns = ["time", "cell", "Nhist", "origin", "origin2"]
        df_n.origin = df_n.origin +" " +df_n.origin2
        df_n.drop(columns = ["origin2"], inplace=True)

        df_p = pd.read_csv(os.path.join(jjhdatafolder, fname_p), delimiter="\s+", header=None)
        df_p.columns = ["time", "cell", "Nhist", "origin"]
        return df_n, df_p
    if tid == None :
        dfs_n = {}
        dfs_p = {}
        for tt in range(1, 21):
            fn0 = "C"+str(inid)+"_"+str(tt)
            fname_n = fn0+"_neutron_"+strstg+".o"
            fname_p = fn0+"_photon_"+strstg+".o"
            
            df_n, df_p = read_jjh(inid, tt)
            dfs_n[fname_n[:-2]]=df_n
            dfs_p[fname_p[:-2]]=df_p
        return dfs_n, dfs_p
    
    
def jjh_combine(jjhdatafolder, inid=1):
    dfsn, dfsp = read_jjh(jjhdatafolder, inid)
    
    for key, dfn in dfsn.items():
        tid = int(key.split("_")[1])
        dfn["time"]=dfn["time"] + 100000*(tid-1)
    
    for key, dfp in dfsp.items():
        tid = int(key.split("_")[1])
        dfp["time"]=dfp["time"] + 100000*(tid-1)
    
    combined_dfn = pd.concat(list(dfsn.values()))
    combined_dfp = pd.concat(list(dfsp.values()))
    return combined_dfn, combined_dfp


def jjh_combine(jjhdatafolderpath, targetpath, inid=3):
    A=['neutron', 'photon']
    for z in range(0, 2):
        pref = "C"+str(inid)+"_"+A[z]
        f_new11 = open(os.path.join(targetpath, pref+"_combined.o"), "w")
        for y in range(1, 21):
            f_new1 =  open(os.path.join(jjhdatafolderpath, "C"+str(inid)+"_"+str(y)+"_"+A[z]+"_reduced.o"), "r")
            line1=f_new1.readlines()

            for i in range(0,len(line1)):
                a = line1[i].split()[1]
                b = int(float(line1[i].split()[0]))
                if int(a) > 120 and int(a) < 130:
                        f_new11.write(str(100000*(y-1)+b)+"\n")
                elif int(a) > 140 and int(a) < 144:
                        f_new11.write(str(100000*(y-1)+b)+"\n")
                elif int(a) > 220 and int(a) < 230:
                        f_new11.write(str(100000*(y-1)+b)+"\n")
                elif int(a) > 240 and int(a) < 244:
                        f_new11.write(str(100000*(y-1)+b)+"\n")
                elif int(a) > 320 and int(a) < 330:
                        f_new11.write(str(100000*(y-1)+b)+"\n")
                elif int(a) > 340 and int(a) < 344:
                        f_new11.write(str(100000*(y-1)+b)+"\n")

            f_new1.close()
        f_new11.close()


            