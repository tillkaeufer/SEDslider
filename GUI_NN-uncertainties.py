#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pickle 

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
# jupyters notebook Befehl zum direkten Anzeigen von Matplotlib Diagrammen
plt.rcParams['figure.figsize'] = (9, 6)
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
colormap={0:'red',1:'green'}
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[2]:


name='single_17_epo2000' # what network to use
path_data='./data' #where is the downloaded data
knn_switch=True # using Knn
Knn_factor=2 #parameter to adjust the uncertainties


# # loading data

print('Loading data')
# In[3]:


scaler=joblib.load(f'{path_data}/scaler/{name}_para_scaler.save')
y_scaler=joblib.load(f'{path_data}/scaler/{name}_sed_scaler.save')


# In[4]:


model_saved=load_model(f'{path_data}/NeuralNets/{name}.h5')


# In[5]:


header=np.load(f'{path_data}/header.npy')
header=np.concatenate((header,['incl']),axis=0)
#print(header)


# In[6]:


txt=str()
with open(f'{path_data}/wavelength.out','r') as f:
    lines=f.readlines()
for line in lines[1:]:
    
    txt=txt+line.strip()+' '  
txt=txt[1:-2].split()
wavelength=np.array(txt,'float64')
  


# In[10]:



if knn_switch:
    n_neighbors=20
    weights='distance'
    with open(f'{path_data}/Knns/{name}_{n_neighbors}_{weights}', 'br') as file_pi:
        knn=pickle.load(file_pi)
    dat='test'
    mean_sample=np.load(f'{path_data}/Knns/sample_values/{name}_{n_neighbors}_{weights}_{dat}_mean.npy')
    min_sample=np.load(f'{path_data}/Knns/sample_values/{name}_{n_neighbors}_{weights}_{dat}_min.npy')
    max_sample=np.load(f'{path_data}/Knns/sample_values/{name}_{n_neighbors}_{weights}_{dat}_max.npy')


# # preparing handling of the data

# In[11]:

print('preparing handling of the data')

def transform_parameter(name,val):
    dummy=np.zeros((1,len(header)))
##    if name in name_log:
#        val=np.log10(val)
    pos=np.where(header==name)[0][0]
    dummy[0,pos]=val
    val=scaler.transform(dummy)[0,pos]
    return val,pos


# In[12]:


slider_dict={
    'Mstar':{
        'label':r'$log(M_{star}) [M_{sun}]$',
        'lims':[-0.69, 0.39],
        'x0':0.06,
        'priority':1}
        ,
    
    'Teff':{
        'label':r'$log(T_{eff})$',
        'lims':[3.5, 4.0], 
        'x0':3.69,
        'priority':1},
    
    'Lstar':{
        'label':r'$log(L_{star})$',
        'lims':[-1.3, 1.7],
        'x0':0.79,
        'priority':1}, 
    'fUV':{
        'label':r'$log(fUV)$',
        'lims':[-3, -1],
        'x0':-0.57, 
        'priority':1},
    
    'pUV':{
        'label':r'$log(pUV)$',
        'lims':[-0.3, 0.39],
        'x0':-0.02, 
        'priority':1},
    
    'Mdisk':{
        'label':r'$log(Mass_{disk})$',
        'lims':[-5, 0],
        'x0':-1.367, 
        'priority':2},
    
    'incl':{
        'label':r'$incl [Deg]$',
        'lims':[0, 9],
        'x0':2,
        'priority':2},
    
    'Rin':{
        'label':r'$log(R_{in}[AU])$',
        'lims':[-2.00, 2.00], 
        'x0':-1.34,
        'priority':2},
   
     'Rtaper':{
        'label':r'$log(R_{taper}[AU])$',
        'lims':[0.7, 2.5],
         'x0':1.95, 
        'priority':2},
    
    'Rout':{
        'label':r'$log(R_{out}[AU])$',
        'lims':[1.3, 3.14],
        'x0':2.556, 
        'priority':2},
    
    'epsilon':{
        'label':r'$\epsilon$',
        'lims':[0, 2.5],
        'x0':1, 
        'priority':2},
    
    'MCFOST_BETA':{
        'label':r'$\beta$',
        'lims':[0.9, 1.4],
        'x0':1.15, 
        'priority':2},
    
    'MCFOST_H0':{
        'label':'MCFOST_H0[AU]',
        'lims':[3, 35],
        'x0':12, 
        'priority':2},    
    
    'a_settle':{
        'label':r'$log(a_{settle})$',
        'lims':[-5, -1],
        'x0':-3, 
        'priority':3},
    
    'amin':{
        'label':r'$log(a_{min})$',
        'lims':[-3, -1],
        'x0':-1.5, 
        'priority':3},
    
    
    'amax':{
        'label':r'$log(a_{max})$',
        'lims':[2.48, 4],
        'x0':3.6, 
        'priority':3},
    
    'apow':{
        'label':r'$a_{pow}$',
        'lims':[3, 5],
        'x0':3.6, 
        'priority':3},
    
    'Mg0.7Fe0.3SiO3[s]':{
        'label':r'Mg0.7Fe0.3SiO3[s]',
        'lims':[0.45, 0.7],
        'x0':0.57, 
        'priority':3},
    
    'amC-Zubko[s]':{
        'label':r'amC-Zubko[s]',
        'lims':[0.05, 0.3],
        'x0':0.18, 
        'priority':3},
    
    'fPAH':{
        'label':r'$log(fPAH)$',
        'lims':[-3.5, 0],
        'x0':-1.5, 
        'priority':3},
    
    'PAH_charged':{
        'label':r'PAH_charged',
        'lims':[0, 1], 
        'priority':3},
}


# In[13]:


log_dict={'Mstar': 'log', 'Lstar': 'log', 'Teff': 'log', 'fUV': 'log', 'pUV': 'log', 'amin': 'log', 'amax': 'log',
          'apow': 'linear', 'a_settle': 'log', 'Mg0.7Fe0.3SiO3[s]': 'linear', 'amC-Zubko[s]': 'linear', 'fPAH': 'log',
       'PAH_charged': 'linear', 'Mdisk': 'log', 'Rin': 'log', 'Rtaper': 'log', 'Rout': 'log', 'epsilon': 'linear',
       'MCFOST_H0': 'linear', 'MCFOST_BETA': 'linear', 'incl': 'linear'}


# In[14]:


for key in log_dict:
    slider_dict[key]['scale']=log_dict[key]


# In[15]:


for key in slider_dict:
    if slider_dict[key]['scale']=='log':
        if 'log' in slider_dict[key]['label']:
            print(slider_dict[key]['label']+': fine')
        else:
            slider_dict[key]['label']='$log('+slider_dict[key]['label'][1:-1]+')$'
            low=slider_dict[key]['lims'][0]
            high=slider_dict[key]['lims'][1]
            slider_dict[key]['lims']=[np.log10(low),np.log10(high)]            


# # plotting

# In[18]:
print('Lets go')

color_list=['bisque','lightsteelblue', 'lightgreen','lightgoldenrodyellow']

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2, bottom=0.5)

features=np.zeros((1,len(header)))
for key in slider_dict:
    #print(key)
    [down,up]=slider_dict[key]['lims']
    try:
        middle=slider_dict[key]['x0']

    except:
        middle=(up-down)/2+down
    
    val_trans, pos=transform_parameter(key,middle)
    #print(val_trans)
    features[0,pos]=val_trans  
#print(features)
data=10**(y_scaler.inverse_transform(model_saved.predict(features)))[0]
if knn_switch:
    error_knn=knn.predict(features)
    higher, =plt.plot(wavelength,10**(np.log10(data)+Knn_factor*error_knn[0]),color='grey',alpha=1)
    lower, =plt.plot(wavelength,10**(np.log10(data)-Knn_factor*error_knn[0]),color='grey',alpha=1)
t=wavelength
s = data
l, = plt.plot(t, s,marker='+',linestyle='none')

plt.axis([np.min(wavelength), np.max(wavelength), 10**(-18), 10**(-6)])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$ \lambda \, [\mu m]$')
plt.ylabel(r'$ \nu F_\nu [erg/cm^2/s]$')

placed=[]
l_head=len(header)
i=0
l_1=0
r=0
for key in slider_dict:
    if i<l_head/2:
        frame=[0.1,np.round(0.1+l_1*0.03,2),0.3,0.02]
        l_1+=1
    else:
        frame=[0.6,np.round(0.1+r*0.03,2),0.3,0.02]
        r+=1

   
    plot=plt.axes(frame, facecolor=color_list[slider_dict[key]['priority']-1])
    label=slider_dict[key]['label']
    down=slider_dict[key]['lims'][0]
    up=slider_dict[key]['lims'][1]
    try:
        middle=slider_dict[key]['x0']
    except:
        middle=(up-down)/2+down
    slider=Slider(plot, label, down, up, valinit=middle)
    slider_dict[key]['slider']=slider
    i+=1
#print(placed)

features=np.zeros((1,len(header)))


def update(val):
    #features=np.zeros((1,len(header)))
    for key in slider_dict:
        val_trans, pos=transform_parameter(key,slider_dict[key]['slider'].val)
        features[0,pos]=val_trans
    #print(features)
    data=10**(y_scaler.inverse_transform(model_saved.predict(features)))[0]
    if knn_switch:  
        dist,neighbor_ar=knn.kneighbors(features)
        min_dist,mean_dist,max_dist=np.min(dist)/min_sample,np.mean(dist)/mean_sample,np.max(dist)/max_sample
        txt=''
        if mean_dist<=1.0:
            colortitle='tab:green'
        elif 1.0<=mean_dist<=2.0:
            colortitle='tab:orange'
        elif 2.0<=mean_dist:
            colortitle='tab:red'
            txt='Warning!! Few models! '
        ax.set_title(txt+'Distance to neighbors (average=1): Minimum %4.2f, Mean %4.2f, Maximum %4.2f' %(min_dist,mean_dist,max_dist),color=colortitle)
        l.set_ydata(data)
        lower.set_ydata(10**(np.log10(data)-Knn_factor*error_knn[0]))
        higher.set_ydata(10**(np.log10(data)+Knn_factor*error_knn[0]))
        fig.canvas.draw_idle()

    
for key in slider_dict:
    slider_dict[key]['slider'].on_changed(update)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
#slider.on_changed(update)

def reset(event):
    for key in slider_dict:
        slider_dict[key]['slider'].reset()
button.on_clicked(reset)


plt.show()


# In[ ]:





# In[28]:




