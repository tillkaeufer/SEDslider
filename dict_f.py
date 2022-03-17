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
        'x0':-1.57, 
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


two_slider_dict={
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
        'label':r'$R_{in}[AU]$',
        'lims':[1, 70], 
        'x0':18,
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
    'M2disk':{
        'label':r'log(M2disk)',
        'lims':[-11, -1],
        'x0':-6,
        'priority':4},
    'e2psilon':{
        'label':r'e2psilon',
        'lims':[-1, 2],
        'x0':0.7,
        'priority':4},
    'MCFOST_H2':{
            'label':r'MCFOST_H2[AU]',
        'lims':[0.02, 0.5],
        'x0':0.1,
        'priority':4},
    'MCFOST_B2':{
        'label':r'MCFOST_B2',
        'lims':[0.05, 1.7],
        'x0':0.27,
        'priority':4},
    'R2in':{
        'label':r'$log(R2_{in}[AU])$',
        'lims':[-2, 2],
        'x0':-0.5,
        'priority':4},
    'R2out':{
        'label':r'$log(R2_{out}[AU])$',
        'lims':[-2, 1.5],
        'x0':0.9,
        'priority':4},
    'a2max':{
        'label':r'$log(a2_{max})$',
        'lims':[-1, 4],
        'x0':2.5,
        'priority':4},
    'f2PAH':{
            'label':r'log(fPAH)',
        'lims':[-6, -1],
        'x0':-3,
        'priority':4}
}
#changed Rin and added two zone paras
log_dict={'Mstar': 'log', 'Lstar': 'log', 'Teff': 'log', 'fUV': 'log', 'pUV': 'log', 'amin': 'log', 'amax': 'log',
          'apow': 'linear', 'a_settle': 'log', 'Mg0.7Fe0.3SiO3[s]': 'linear', 'amC-Zubko[s]': 'linear', 'fPAH': 'log',
       'PAH_charged': 'linear', 'Mdisk': 'log', 'Rin': 'linear', 'Rtaper': 'log', 'Rout': 'log', 'epsilon': 'linear',
       'MCFOST_H0': 'linear', 'MCFOST_BETA': 'linear', 'incl': 'linear',
         'M2disk':'log','e2psilon':'linear','MCFOST_H2':'linear','MCFOST_B2':'linear','R2in':'log','R2out':'log', 'a2max':'log', 'f2PAH':'log'}