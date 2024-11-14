import os
import subprocess
import sys
import uuid
import scipy as sp
import numpy as np
import scipy.special as special
from scipy import integrate
from scipy.optimize import minimize_scalar



def main(A,ipi,fMod,ITD,audDep):

    # time parameters
    tStart = 0; 
    delay = 10;
    stimDuration = 200
    dt = 0.002
    tEnd = 2*delay + stimDuration; 
    t = np.arange(tStart,tEnd,dt)

    # input paramters
    # ipi = interpulse interval [ms]
    # A = pulse amplitude [pA]
    # ITD = time difference [ms]
    # fMod = modulation frequency [Hz].  0=unmodulated
    # audDep = [0=control model, 1=auditory deprived model]

    # run AN model
    nNeuron = 10
    iterNum = 0
    sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)

    # save to AN file
    ANfileName = 'ANfile.txt'
    np.savetxt(ANfileName,gSum,fmt="%f")

    # run MSO model
    writeData = 1
    synStrength = 21
    MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)
    MSOfileName = 'MSOVoltage.txt'
    
    # get voltage data
    t,v1,v2 = getMSOvoltage(MSOfileName)
    # ax[0].plot(t,gSum)
    # ax[1].plot(t,v1,t,v2)

    return t,v1,v2,gSum





def ANmodel(nPulse,Delta,A,fMod):

    # pulse train parameters
    # input: nPulse      # number of pulses
    # input: A           # pulse amplitude (pA) [constant amplitude pulse trains only]
    # input: Delta       # interpulse interval (ms)
    delta  = 0.05   # pulse duration (ms)
    
    # ANF parameters
    Cm     = 0.0714 # capacitance (pF)
    taum   = 0.1395 # membrane time constants (ms) 
    th0    = 30     # initial threshold 
    rs0    = 0.05   # initial relative spread
    sigJ   = 0.1    # st. dev. of spike time jitter (ms) 

    # refractory parameters
    tauThetar = 0.411 # threshold relative refractory period (milliseconds)
    tauRSr = 0.2 # relative spread refractory time constant(ms)
    tAbs = 0.332 # absolute refractory period (milliseconds)

    # adaptation parameters
    rho = 0.04 # Threshold SRA increment & Relative Spread SRA increment
    tauSRA = 50. # time constants (milliseconds) SRA

    # facilitation parameters
    aVthF = -0.15 # threshold strength (1/ms)
    aRSF = 0.75 # relative spread strength (1/ms)
    tauVthF = 0.5 # threshold time constant (fast, milliseconds)
    tauRSF = 0.3 # relative spread time constant (milliseconds)

    # accommodation parameters
    # Quick
    aVthQuick = 0.5 # threshold strength (1/ms) 
    aRSQuick = 0.75 # relative spread strength(1/ms)
    tauVthQuick = 1.5 # threshold time constant (quick, milliseconds)
    tauRSQuick = 0.3 # relative spread time constant (quick, milliseconds)
    # Slow
    aVthSlow = 0.01 # threshold strength (1/ms)
    aRSSlow = 0 # relative spread strength (1/ms)
    tauVthSlow = 50 # threshold time constant (slow, milliseconds)
    tauRSSlow = 50 # relative spread time constant (slow, milliseconds)

    # spike probability function
    def pSpike(V,th,rs):
        sig = th*rs
        arg = (V - th) / (np.sqrt(2)*sig)
        return 0.5*(1 + special.erf( arg ) )

    # function for updating facilitation and accommodation multipliers
    def multiplierUpdate(previous,a,tau,Vonset,Voffset,A):

        tauTerm = (taum*tau)/(taum-tau)

        term1 = ((A*taum*tau)/Cm)*(1-np.exp(-delta/tau)) 
        term2 = (Vonset-A*taum/Cm)*tauTerm*np.exp(-delta/taum)*(1-np.exp(-delta/tauTerm))

        val = previous*np.exp(-delta/tau)
        val += (a/th0)*(term1 + term2)
        
        out = val*np.exp(-(Delta-delta)/tau) + (a/th0)*Voffset*tauTerm*np.exp(-(Delta-delta)/taum)*(1 - np.exp(-(Delta-delta)/tauTerm) )

        return out


    # loop across current pulses
    V = np.zeros(nPulse)
    spikeTimes = []
    r = sp.random.rand(nPulse) # draw random numbers for stochastic spiking
    Voffset = 0
    tOffset = delta-Delta
    tSpike = -999. # initialize time of previous spike to distant past
    xad = 1.  # initialize adaptation multiplier
    xadLastSpike = 1.  # initialize adaptation multiplier
    xxfac = 0 # initialize facilitation multiplier
    yyfac = 0 # initialize facilitation multiplier
    xxaccQ = 0 # initialize accommodation multiplier
    yyaccQ = 0 # initialize accommodation multiplier
    xxaccS = 0 # initialize accommodation multiplier
    yyaccS = 0 # initialize accommodation multiplier
    xfac =  1
    yfac =  1
    xacc = 1 
    yacc = 1
    for i in range(nPulse):


        # update time at pulse offset
        tOffset += Delta
        tOnset = tOffset - delta

        # for amplitude modulated pulses
        if fMod == 0:
            Apulse = A
        else: 
            Abottom = 70
            Apulse = Abottom + (A-Abottom)*(1. + np.sin(2*np.pi*fMod*tOnset/1000 - np.pi/2) ) / 2.
        # update voltage
        Vonset  = Voffset*np.exp(-(Delta-delta)/taum)
        Voffset = Vonset*np.exp(-delta/taum) + (Apulse*taum/Cm)*(1.-np.exp(-delta/taum) )
        
        # multipliers for threshold and RS values
        T = tOffset - tSpike # time since last spike
        xref = 1./ (1. - np.exp(-(T-tAbs)/tauThetar) )
        yref = 1. + np.exp(-(T-tAbs)/tauRSr)
        xad  = 1. + (xadLastSpike - 1.)*np.exp(-T/tauSRA)
        yad  = xad # per Boulet Eq 3.23 pg 54 in dissertation

        # spike probablity for this pulse
        if (tOffset-tSpike)<tAbs: # absolute refractory period
            p = 0
        else: # calculate spike probability
            th = th0*xref*xfac*xad*xacc
            rs = rs0*yref*yfac*yad*yacc
            p = pSpike(Voffset,th,rs)
        
        # calculate multipliers for facilitation and accommodation for next pulse 
        xxfac  = multiplierUpdate(0,aVthF, tauVthF,Vonset,Voffset, Apulse) # facilitation resets after each pulse
        yyfac  = multiplierUpdate(0, aRSF, tauRSF, Vonset,Voffset, Apulse)
        xxaccQ = multiplierUpdate(xxaccQ, aVthQuick, tauVthQuick,Vonset,Voffset, Apulse)
        yyaccQ = multiplierUpdate(yyaccQ, aRSQuick, tauRSQuick,Vonset,Voffset, Apulse)
        xxaccS = multiplierUpdate(xxaccS, aVthSlow, tauVthSlow,Vonset,Voffset, Apulse)
        
        xfac = 1. + xxfac
        yfac = 1. + yyfac
        xacc = 1. + xxaccQ + xxaccS
        yacc = 1. + yyaccQ + yyaccS

        # stochastic spike generation
        if (r[i]<p): # spike occurs
            tSpike = tOffset + sigJ*np.random.randn() # add spike time jitter
            spikeTimes.append(tSpike) # record spike time
            Voffset = 0 # reset voltage
            xadLastSpike = xad + rho # increment for adaptation multiplier
            xxfac = 0 # reset facilitation
            yyfac = 0 # reset facilitation

    return np.asarray(spikeTimes)




# make synaptic EPSG signal
def makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron):

    # inputs: 
    # stimDuration = stimulus duration [ms]
    # dt = time step [ms]
    # ipi = interpulse interval [ms]
    # A = pulse amplitude [pA]
    # ITD = time difference [ms]
    # fMod = modulation frequency [Hz].  0=unmodulated
    # nNeuron = # neurons (per side)

    # time parameters
    tStart = 0; 
    delay = 10; # pad onset and offset
    tEnd = 2*delay + stimDuration; 
    t = np.arange(tStart,tEnd,dt)
    nt = np.size(t)

    # parameters
    synStrength = 21. 

    # initialized conductance arrays
    gSynL = np.zeros_like(t) # an array of synaptic conductance, same size t
    gSynR = np.zeros_like(t)
    spike_arrL = np.zeros_like(t) # an array 0s and 1s for spike counts
    spike_arrR = np.zeros_like(t)

    # call AN model
    nPulse = int(np.floor(stimDuration/ipi))
    for i in range(nNeuron):

        # loop through time bins to create 0s and 1s in small time bins
        s = ANmodel(nPulse,ipi,A,fMod) # create spike train
        s = s-ITD/2+delay # ITD shift
        if len(s) != 0: # ONLY START THE FOR LOOP IF THE SPIKE TIME ARRAY IS NOT EMPTY
            j = 0 # a counter to move through the spikeTime array
            for i in np.arange(nt)-1:
                if (s[j]>=t[i]) and (s[j]<t[i+1]):
                    spike_arrL[i] += 1.
                    if j < (len(s)-1):
                        j += 1
                    else:
                        break

        s = ANmodel(nPulse,ipi,A,fMod) # create spike train
        s = s+ITD/2+delay # ITD shift
        if len(s) != 0: # ONLY START THE FOR LOOP IF THE SPIKE TIME ARRAY IS NOT EMPTY
            j = 0 # a counter to move through the spikeTime array
            for i in np.arange(nt)-1:
                if (s[j]>=t[i]) and (s[j]<t[i+1]):
                    spike_arrR[i] += 1.
                    if j < (len(s)-1):
                        j += 1
                    else:
                        break

        ng = int(np.floor(5/dt)) # change this later
        gFunc = (1./0.21317) * (np.exp(-t[:ng]/.18) - np.exp(-t[:ng]/.1) )  # EPSG unitary waveform
    
        cL = np.convolve(gFunc,spike_arrL)  # convolve the EPSG with the spike events
        cR = np.convolve(gFunc,spike_arrR)  # convolve the EPSG with the spike events

        gSynL = cL[:nt]
        gSynR = cR[:nt]
        gSum = gSynL + gSynR

    return spike_arrR,spike_arrL,gSum



def MSOmodel(tEnd,dt,ipi, ITD, A, synStrength, nNeuron, iterNum, writeData, audDep):

    ############################################# FROM GLOBAL VAR ##############
    couple12 = .8
    couple21 = .5
    GNA = 2000

    # couple12 = .9
    # couple21 = .3
    # GNA = 1000

    # KLT gating
    EK = -106;

    # H gating
    Eh = -47; #-37; # had to change this to get -58 resting potl
    kr = 0.65;

    # Na gating
    ENa  = 55;     # Na reversal potential [mV]

    # KHT gating
    phi = 0.85;

    # resting potential [mV] 
    Vrest = -58; 

    # ORIGINAL KR-parameters
    c1 = 25; c2 = 12; # capacitance [pF]
    gAx = 50;# axial conductance [nS]
    gKLT = 190; # [nS]
    gh = 70; # [nS]
    gNa = 3000; # [nS]
    gKHT = 150; # [nS]
    glk1 = 15;
    gTot2 = 24;
    #################################################################################

    ### KLT gating ###
    def winf(V):
        return (1. + np.exp(-(V+57.3)/11.7) )**-1;
    def zinf(V):
        return (1.-.22) / (1.+np.exp((V+57.)/5.44)) + .22;
    def tauw(V):
        return .46*(100. / (6.*np.exp((V+75.)/12.15) + 24.*np.exp(-(V+75.)/25.) + .55));
    def tauz(V):
        return .24*(1000. / (np.exp((V+60.)/20.) + np.exp(-(V+60.)/8.)) + 50.);

    # H gating
    def rinf(V):
        return (1 + np.exp((V+60.3)/7.3))**(-1);
    def taurf(V):
        return 10**4 / ( (-7.4*(V+60))/(np.exp(-(V+60)/0.8)-1) + 65*np.exp(-(V+56)/23) );
    def taurs(V):
        return 10**6 / ( (-56*(V+59))/(np.exp(-(V+59)/0.8)-1) + 0.24*np.exp(-(V-68)/16) );

    # Na gating Rothman Manis with 35C temp adjustment
    def minf(V):
        return (1.+np.exp(-(V+38.)/7.))**-1.;
    def hinf(V):
        return (1.+np.exp((V+65.)/6.))**-1.;
    def taum(V):
        return .24*((10 / (5*np.exp((V+60) / 18) + 36*np.exp(-(V+60) / 25))) + 0.04);
    def tauh(V):
        return .24* (100. / (7.*np.exp((V+60)/11.) + 10.*np.exp(-(V+60.)/25.)) + 0.6);

    # KHT gating Rothman Manis with 35C temp adjustment
    def ninf(V):
        return (1 + np.exp(-(V + 15) / 5))**-0.5;
    def pinf(V):
        return 1 / (1 + np.exp(-(V + 23) / 6));
    def taun(V):
        return 0.24* ((100 / (11*np.exp((V+60) / 24) + 21*np.exp(-(V+60) / 23))) + 0.7);
    def taup(V):
        return 0.24* ((100 / (4*np.exp((V+60) / 32) + 5*np.exp(-(V+60) / 22))) + 5);

    # M gating [modeldb: https://senselab.med.yale.edu/ModelDB/ShowModel?model=114394&file=/NN_kole/Km.mod#tabs-2, Kuba 2015]
    tha = -30
    Ra = 0.001
    Rb = 0.001
    qa = 9.
    tadj = 2.3**( (35-23) / 10) # temperature adjustm to 35 with q10=2.3
    def aM(V):
        return Ra * (V - tha) / (1 - np.exp(-(V - tha)/qa))
    def bM(V):
        return -Rb * (V - tha) / (1 - np.exp((V - tha)/qa))
    def tauMM(V):
        return (1/tadj) * (1 /( aM(V)+bM(V) ))
    def MMinf(V):
        return aM(V) /( aM(V)+bM(V) )

    # conductances
    gh0 = gh*(kr*rinf(Vrest) + (1-kr)*rinf(Vrest));
    gKLT0 = gKLT * winf(Vrest)**4*zinf(Vrest);
    gNa0 = gNa * minf(Vrest)**3*hinf(Vrest);
    gKHT0 =  gKHT*(phi*ninf(Vrest)**2 + (1-phi)*pinf(Vrest));

    # total compartment conductance . (frozen, at rest) [nS]
    gTot1 = glk1 + gh0 + gKLT0;  # cpt1
    gTot2 = gTot2 + gNa0 + gKHT0; # cpt2

    KRcouple =  [ 1/(1+gTot2/gAx) , 1/(1+gTot1/gAx)] ;
    
    # fraction of active conductances
    fKLT = gKLT0/gTot1;
    fh = gh0/gTot1;
    fNa = gNa0/gTot2;
    fKHT = gKHT0/gTot2;

    R1 = (gTot2+gAx) / ((gTot1+gAx)*(gTot2+gAx) - gAx**2 ); # Input resistance to CPT1 [10**9 Ohm]
    
    areaRatio = c2/c1;

    # Numerical fit to find time constant #
    M = np.array([ [ (gTot1 + gAx)/c1 ,  -gAx/c1] , [-gAx/c2 , (gTot2 + gAx)/c2]]); # steady state matrix;
    # eigen-calculations. eigenvectors both have 1 in first entry
    evl,evc = np.linalg.eig(M)

    eval1 = evl[0]
    eval2 = evl[1]
    evec1 = evc[:,0]/evc[0,0]
    evec2 = evc[:,1]/evc[0,1]

    # coordinates of initial values [1;stoa] in eigenbasis
    w = np.linalg.solve( np.array( [evec1 , evec2]).transpose() , np.array([1,KRcouple[0]]) );

    # minimize L2 error with numerical integration
    def fobj(tau):
        I = integrate.quad( lambda t: (np.exp(-t/tau) -  np.exp(-eval1*t)*w[0] - np.exp(-eval2*t)*w[1] )**2, 0,5. ) ;
        return I[0]
    minSolve = minimize_scalar(fobj, bounds=(0, 1), method='bounded') # estimated single-exponential membrane time constant
    tauEst = minSolve.x

    # Fixed parameters used to fit model #
    # areaRatio = [ calculated above] 0.01
    # R1      =  [calculated above to be 14.8] # 8.5 * 1e-3;    #Input resistance to CPT1 [10**9 Ohm]
    # tauEst  = tauEst; #estimated to be .445 ms;     # "estimated time constant" [ms]
    # Vrest   = Vrest; #-58;    # resting membrane potential [mV]
    Elk     = Vrest; #-58;    # leak reversal potential [mV]

    # Passive parameters #
    gAx    = (1/R1) * couple21 / (1-couple12*couple21); # coupling conductance [nS]
    gTot1  = gAx * (1/couple21 - 1); # CPT1 leak conductance [nS]
    gTot2  = gAx * (1/couple12 - 1); # CPT2 leak conductance [nS]

    # Passive parameters that require separation of time scales assumption #
    tau1  = tauEst * (1-couple12*couple21);          # CPT1 time constant [ms]
    tau2  = tau1 * areaRatio * (couple12/couple21);  # CPT2 time constant [ms]
    cap1  = tau1 * (gTot1 + gAx); # CPT1 capacitance [pF]
    cap2  = tau2 * (gTot2 + gAx); # CPT2 capacitance [pF]

    # active conducance - cpt1
    gKLT = (fKLT*gTot1) / (winf(Vrest)**4*zinf(Vrest));
    gh =  (fh*gTot1)/ ((kr*rinf(Vrest) + (1-kr)*rinf(Vrest)));

    # active conducance - cpt2
    gKLT2 = (fKLT*gTot2) / (winf(Vrest)**4*zinf(Vrest));
    gh2   =  (fh*gTot2)/ ((kr*rinf(Vrest) + (1-kr)*rinf(Vrest)));
    gKHT  = (gKHT/gNa)*GNA;
    gM    = 0 # changed below for sound deprived model

    # passive conductance
    glk1 = (1 - (fKLT + fh))*gTot1;
    glk2 = (1 - (fKLT+fh))*gTot2 - (gNa*(minf(Vrest)**3*hinf(Vrest))  + gKHT* ((phi*ninf(Vrest)**2 + (1-phi)*pinf(Vrest))));

    # adjust leak for -58 mV resting potl
    Elk1 =  Vrest + ( gKLT*winf(Vrest)**4*zinf(Vrest)*(Vrest-EK)  + gh*rinf(Vrest)*(Vrest-Eh) ) / glk1;
    Elk2 =  Vrest + ( gKHT*(phi*ninf(Vrest)**2 + (1-phi)*pinf(Vrest) ) * (Vrest-EK)  +  gNa*(minf(Vrest)**3)*hinf(Vrest)*(Vrest-ENa) + gh2*rinf(Vrest)*(Vrest-Eh) + gKLT2*winf(Vrest)**4*zinf(Vrest)*(Vrest-EK)    ) / glk2;

    if audDep==1:

        # increase size of cpt2: (Kuba 2010, 2012 review)
        cap2 *= 1.5
        gNa *= 1.5
        gKHT *= 1.5
        glk2 *= 1.5 

        # reduce KLT in soma (Kuba 2015)
        gKLT *= .5
        gh *= .5 # also reduce to maintain resting potl (cf. Khurana)

        # replace KLT with M current (cf. Kuba et al 2015)
        gM =  .5*fKLT*gTot2 / MMinf(Vrest) # 1.7*fKLT*gTot2 / MMinf(Vrest)
        gKLT2 = 0

    elif audDep==2: # change AIS only

        # increase size of cpt2: (Kuba 2010, 2012 review)
        cap2 *= 1.5
        gNa *= 1.5
        gKHT *= 1.5
        glk2 *= 1.5 

        # DO NOT reduce KLT in soma (Kuba 2015)

        # replace KLT with M current (cf. Kuba et al 2015)
        gM =  .5*fKLT*gTot2 / MMinf(Vrest) # 1.7*fKLT*gTot2 / MMinf(Vrest)
        gKLT2 = 0


    elif audDep==3: # change soma only

        # DO NOT CHANGE AIS

        # reduce KLT in soma (Kuba 2015)
        gKLT *= .5
        gh *= .5 # also reduce to maintain resting potl (cf. Khurana)


    #################################################################################

    # RUN MSO MODEL [c]

    # create command line string
    os.system('make clean')
    os.system('make')
    executable = 'twoCptODE'

    # run c code
    runIt = [str(writeData)]
    runIt=np.append(runIt,str(tEnd))
    runIt=np.append(runIt,str(dt))
    runIt=np.append(runIt,f'{cap1:.2f}')
    runIt=np.append(runIt,f'{cap2:.2f}')
    runIt=np.append(runIt,f'{Elk1:.2f}')
    runIt=np.append(runIt,f'{Elk2:.2f}')
    runIt=np.append(runIt,f'{glk1:.2f}')
    runIt=np.append(runIt,f'{glk2:.2f}')
    runIt=np.append(runIt,f'{gKLT:.2f}')
    runIt=np.append(runIt,f'{gh:.2f}')
    runIt=np.append(runIt,f'{gKLT2:.2f}')
    runIt=np.append(runIt,f'{gh2:.2f}')
    runIt=np.append(runIt,f'{gM:.2f}')
    runIt=np.append(runIt,f'{gNa:.2f}')
    runIt=np.append(runIt,f'{gKHT:.2f}')
    runIt=np.append(runIt,f'{gAx:.2f}')
    runIt=np.append(runIt,f'{ipi:.2f}')
    runIt=np.append(runIt,f'{A:.2f}')
    runIt=np.append(runIt,f'{ITD:.2f}')
    runIt=np.append(runIt,f'{synStrength:.2f}')
    runIt=np.append(runIt,f'{nNeuron:.0f}')
    runIt=np.append(runIt,f'{audDep:.0f}')
    runIt=np.append(runIt,f'{iterNum:.0f}')
    
    u = uuid.uuid4()
    with open('runIt_'+str(u)+'.txt', 'w') as f:
        for r in runIt:
            f.write(r + ",\n")

    subprocess.run('./'+executable+' ' + str(u), shell=True)

    subprocess.run('rm runIt_'+str(u)+'.txt', shell=True)

    return 


# read voltage trace from data file
def getMSOvoltage(MSOfilename):
    file = open(MSOfilename,"r")
    t = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(0))
    v1 = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(1))
    v2 = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(2))
    file.close()
    return t,v1,v2





# main()
# plt.show()




# # read voltage trace from data file
# def getMSOvoltage(MSOfilename):

#     # MSOfilename = 'MSOlib/MSOVoltage'+'_I'+f'{istimStrength:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter'+ f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfilename,"r")
#     # lines = file.readlines()
#     t = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(0))
#     v1 = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(2))

#     file.close()

#     return t,v1,v2


























# # Intro figure. Figure 1. Add to illustrator
# def ANforIntroFigure():

#     # set up figure
#     fig = plt.figure(figsize=(2,1.2),layout='constrained')
#     gs = GridSpec(2, 1, figure=fig)

#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 15
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)

#     # run AN model
#     A = 52
#     ITD = 0.7
#     iterNum = 1 
#     fMod = 0
#     ipi = 4
#     audDep = 0

#     # MSO parameters
#     writeData = 1
#     synStrength = 21
    
#     # 250 pps
#     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,0,fMod,5) # run AN
#     ax=fig.add_subplot(gs[0,0])
#     ax.plot(t,gSum,'k',lw=.8)
#     ax.set_xlim([8,21])
#     ax.set_ylim([-2,8])
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='x',label1On=False)
#     ax.tick_params(axis='y',label1On=False)
#     ax.set_axis_off()

#     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,0,fMod,5) # run AN
#     ax = fig.add_subplot(gs[1,0])
#     ax.plot(t+ITD,gSum,'k',lw=.8)
#     ax.set_xlim([8,21])
#     ax.set_ylim([-2,8])
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='x',label1On=False)
#     ax.tick_params(axis='y',label1On=False)
#     ax.set_axis_off()

#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/figure/ANforIntro.eps', format='eps')

# # AN figure. Figure 2.
# def ANfigure():

#     nRep = 1000

#     fMod = 0

#     IPIvec = [4,2,1]
#     A0vec = [55,60,75]
#     # set up figure
#     fig = plt.figure(figsize=(7.1, 5.5),layout='constrained')
#     gs = GridSpec(5, 6, figure=fig)
#     # fig = plt.figure(figsize=(7.1, 4),constrained_layout=False)
#     # gs = fig.add_gridspec(nrows=1, ncols=2, left=0.03, right=0.9, bottom=0.15, wspace=0.2 )

#     ax1 = fig.add_subplot(gs[3:5,0:3])
#     ax2 = fig.add_subplot(gs[3:5,3:6])

#     T = 50 # 100 # ms, duration
#     for i in range(len(IPIvec)):


#         ipi = IPIvec[i] # ms 
#         print(ipi)
#         nPulse = int(np.floor(T/ipi))

#         # run AN and plot rasters
#         A0 = A0vec[i] # input strength
#         ax = fig.add_subplot(gs[0:2,(2*i):(2*(i+1))])
#         ht = np.arange(0,T+.1,.1)
#         y = np.zeros(len(ht)-1)
#         for iRep in range(nRep):
#             x = ANmodel(nPulse,ipi,A0,fMod)
#             h = np.histogram(x,ht)
#             y += h[0]
#             # print(len(h[0]),len(h[1]),len(ht),len(y))
#             ax.plot(x,1+iRep*np.ones(len(x)),'.k',markersize=1) # raster
#             ax.set_xlim([-1, 50])
#             ax.set_ylim([.5, 50.5])
#             ax.set_xlabel('time (ms)')

#         ax.spines[['right', 'top']].set_visible(False)
#         if i==0:
            
#             ax.set_title('(A1) %i pps' %int(1000/ipi))
#         elif i==1:
#             ax.set_title('(B1) %i pps' %int(1000/ipi))
#         else:
#             ax.set_title('(C1) %i pps' %int(1000/ipi))
#         if i==0:
#             ax.set_ylabel('trials')

#         # plot PSTH
#         ax = fig.add_subplot(gs[2,(2*i):(2*(i+1))])
#         y = y/nRep
#         dh = ht[1]-ht[0]
#         # ax.plot((ht[1:]+ht[:-1])/2,y,'-k',lw=2,label=str(1000/IPIvec[i]))
#         ax.bar((ht[1:]+ht[:-1])/2,y,width=dh,color='k')#,'-k',lw=2,label=str(1000/IPIvec[i]))
#         ax.set_xlim([-1, 50])
#         ax.set_ylim([0, 0.4])
#         ax.set_xlabel('time (ms)')
#         ax.spines[['right', 'top']].set_visible(False)
#         if i==0:
#             ax.set_ylabel('spike time histogram')
#         if i==0:
#             ax.set_title('(A2)')
#         elif i==1:
#             ax.set_title('(B2)')
#         else:
#             ax.set_title('(C2)')


#     T = 300 # ms, duration
#     nA = 30
#     IPIvec = [10, 4, 2, 1]
#     for i in range(len(IPIvec)):
#         print(IPIvec[i])
#         nPulse = int(np.floor(T/IPIvec[i]))

#         # firing rates with varying input strength
#         Avec = np.linspace(40,100,nA)
#         s = np.zeros((nRep,nA))
#         for iA in range(nA):
#             for iRep in range(nRep):
#                 x = ANmodel(nPulse,IPIvec[i],Avec[iA],fMod)
#                 s[iRep,iA] = len(x)

#         p = s/nPulse
#         m = np.mean(1000/T*s,0)
#         ax1.plot(Avec,m,'-',lw=2,label='%i pps' %int(1000/IPIvec[i]) )    
#         ax1.text(102,m[-1]-5,'%i pps' %int(1000/IPIvec[i]))
#         w = np.mean(p,0)
#         ax2.plot(Avec,w,'-',lw=2)#,label=str(IPIvec[i]))
#         if i==0:
#             ax2.text(102,w[-1]+.02,'%i pps' %int(1000/IPIvec[i]))
#         elif i==1:
#             ax2.text(102,w[-1]-.08,'%i pps' %int(1000/IPIvec[i]))
#         # elif i==2:
#         #     ax2.text(82,w[-1]-.15,'%i pps' %int(1000/IPIvec[i]))
#         else:
#             ax2.text(102,w[-1],'%i pps' %int(1000/IPIvec[i]))
#     ax1.set_xlabel('current (nA)')
#     ax1.set_ylabel('spike rate (Hz)')
#     ax1.set_xlim([40 ,115])
#     ax1.set_ylim([0 ,400])
#     # ax1.legend(frameon=False)
#     ax1.spines[['right', 'top']].set_visible(False)
#     ax1.set_title('(D)')

#     ax2.set_xlabel('current (nA)')
#     ax2.set_ylabel('# spikes per pulse')
#     ax2.set_xlim([40 ,115])
#     ax2.set_ylim([0, 1.05]) 
#     # ax2.legend()
#     ax2.spines[['right', 'top']].set_visible(False)
#     ax2.set_title('(E)')

#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/ANmodelFigure.eps', format='eps')


# # MSO voltage figure. Figure 3
# def voltageFigure():
#     fig = plt.figure(figsize=(7.1, 5.5),constrained_layout=False)
#     gs = fig.add_gridspec(nrows=2, ncols=3, left=0.08, right=0.99, bottom=0.1,wspace=0.1,hspace=0.8)
    
#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 50
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
#     # nt = np.size(t)

#     # run AN model
#     # A = 55
#     ITD = 0
#     nNeuron = 10
#     iterNum = 11 
#     fMod = 0

#     # MSO parameters
#     writeData = 1
#     synStrength = 21
    
#     # 250 pps
#     A = 54
#     ipi = 4
#     audDep = 0
#     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfileName,"r")
#     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     file.close()
#     ax = fig.add_subplot(gs[0,0])
#     ax.plot(t,v1,'-',lw=1.5)
#     ax.plot(t,v2,'-',lw=.75)
#     ax.set_xlim(-2,stimDuration+2)
#     ax.set_ylim(-80,50)
#     ax.set_xlabel('time (ms)')
#     ax.set_ylabel('voltage (mV)')
#     ax.set_title('(A1) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)  
#     ax.text(-2,80,'control',fontsize=10)

#     audDep = 1
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfileName,"r")
#     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     file.close()
#     ax = fig.add_subplot(gs[1,0])
#     ax.plot(t,v1,'-',lw=1.5)
#     ax.plot(t,v2,'-',lw=.75)
#     ax.set_xlim(-2,stimDuration+2)
#     ax.set_ylim(-80,50)
#     ax.set_xlabel('time (ms)')
#     ax.set_ylabel('voltage (mV)')
#     ax.set_title('(A2) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.text(-2,80,'deprived',fontsize=10)


#     # 500 pps
#     A = 59
#     ipi = 2
#     audDep = 0
#     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfileName,"r")
#     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     file.close()
#     ax = fig.add_subplot(gs[0,1])
#     ax.plot(t,v1,'-',lw=1.5)
#     ax.plot(t,v2,'-',lw=.75)
#     ax.set_xlim(-2,stimDuration+2)
#     ax.set_ylim(-80,50)
#     ax.set_xlabel('time (ms)')
#     ax.set_title('(B1) %i pps' %int(1000/ipi))
#     # ax.set_ylabel('voltage (mV)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)

#     audDep = 1
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfileName,"r")
#     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     file.close()
#     ax = fig.add_subplot(gs[1,1])
#     ax.plot(t,v1,'-',lw=1.5)
#     ax.plot(t,v2,'-',lw=.75)
#     ax.set_xlim(-2,stimDuration+2)
#     ax.set_ylim(-80,50)
#     ax.set_xlabel('time (ms)')
#     ax.set_title('(B2) %i pps' %int(1000/ipi))
#     # ax.set_ylabel('voltage (mV)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)

#     # 1000 pps
#     A = 81
#     ipi = 1
#     audDep = 0
#     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfileName,"r")
#     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     file.close()
#     ax = fig.add_subplot(gs[0,2])
#     ax.plot(t,v1,'-',lw=1.5,label='$V_1$')
#     ax.plot(t,v2,'-',lw=.75,label='$V_2$')
#     ax.set_xlim(-2,stimDuration+2)
#     ax.set_ylim(-80,50)
#     ax.set_xlabel('time (ms)')
#     ax.set_title('(C1) %i pps' %int(1000/ipi))
#     # ax.set_ylabel('voltage (mV)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.legend(frameon=False,loc='upper right')
#     ax.tick_params(axis='y',label1On=False)

#     audDep = 1
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfileName,"r")
#     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     file.close()
#     ax = fig.add_subplot(gs[1,2])
#     ax.plot(t,v1,'-',lw=1.5)
#     ax.plot(t,v2,'-',lw=.75)
#     ax.set_xlim(-2,stimDuration+2)
#     ax.set_ylim(-80,50)
#     ax.set_xlabel('time (ms)')
#     ax.set_title('(C2) %i pps' %int(1000/ipi))
#     # ax.set_ylabel('voltage (mV)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)

#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/voltageFigure.eps', format='eps')



# # FIGURE 4: MSO FIRING RATE
# def firingRateFigure():

#     fig = plt.figure(figsize=(7.1, 2.1),layout='constrained')
#     gs = GridSpec(1, 5, figure=fig)

#     # parameters
#     nNeuron = 10
#     synStrength = 21

#     ipi = 10
#     itd = 0
#     fMod = 0
#     nRep = 100

#     ax = fig.add_subplot(gs[0,0])    
#     audDep = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control',color='k')
    
#     audDep = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='deprived',color='tab:gray')
#     ax.set_ylim([0,400])
#     # ax.set_xlabel('current (nA)')
#     ax.set_ylabel('spike rate (Hz)')
#     ax.set_title('(A) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.legend(frameon=False,loc='upper left',fontsize='8')



#     ipi = 4
#     ax = fig.add_subplot(gs[0,1])    
#     audDep = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='k')
    
#     audDep = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='tab:gray')
#     ax.set_ylim([0,400])
#     # ax.set_xlabel('current (nA)')
#     ax.set_title('(B) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)

#     ipi = 2
#     ax = fig.add_subplot(gs[0,2])    
#     audDep = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='k')
    
#     audDep = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='tab:gray')
#     ax.set_ylim([0,400])
#     ax.set_xlabel('current (nA)')
#     ax.set_title('(C) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)


#     ipi = 1.33
#     ax = fig.add_subplot(gs[0,3])    
#     audDep = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='k')
    
#     audDep = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='tab:gray')
#     ax.set_ylim([0,400])
#     # ax.set_xlabel('current (nA)')
#     ax.set_title('(D) %i pps' %int(10*np.round(.1*1000/ipi)))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)




#     ipi = 1
#     ax = fig.add_subplot(gs[0,4])    
#     audDep = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='k')
    
#     audDep = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),color='tab:gray')
#     ax.set_ylim([0,400])
#     # ax.set_xlabel('current (nA)')
#     ax.set_title('(E) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)

#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/MSOfiringRate.eps', format='eps')



# # FIGURE 5: ITD TUNING CURVE
# def tuningCurveFigure():

#     fig = plt.figure(figsize=(7.1, 2.1),layout='constrained')
#     gs = GridSpec(1, 5, figure=fig)

#     # parameters
#     nNeuron = 10
#     synStrength = 21

#     ipi = 10
#     A = 51.
#     fMod = 0
#     nRep = 100

#     ax = fig.add_subplot(gs[0,0])    
#     audDep = 0
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control',color='k')
    
#     audDep = 1
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='deprived',color='tab:gray')
#     ax.set_ylim([0,250])
#     # ax.set_xlabel('time difference (ms)')
#     ax.set_ylabel('spike rate (Hz)')
#     ax.set_title('(A) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.legend(frameon=False,loc='upper left',fontsize="7.5")




#     ipi = 4
#     A = 53.
#     fMod = 0
#     nRep = 100

#     ax = fig.add_subplot(gs[0,1])    
#     audDep = 0
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control',color='k')
    
#     audDep = 1
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='deprived',color='tab:gray')
#     ax.set_ylim([0,250])
#     # ax.set_xlabel('time difference (ms)')
#     ax.set_title('(B) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)



#     # audDep = 2
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='AIS only')

#     # audDep = 3
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='soma only')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")

#     # audDep = 4
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='gKLT 1')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")

#     # audDep = 5
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='gKLT 1.5')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")






#     ipi = 2
#     A = 60.
#     fMod = 0
#     nRep = 100

#     ax = fig.add_subplot(gs[0,2])    
#     audDep = 0
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control',color='k')
    
#     audDep = 1
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='deprived',color='tab:gray')
#     ax.set_ylim([0,250])
#     ax.set_xlabel('time difference (ms)')
#     ax.set_title('(C) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)


#     # audDep = 2
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='AIS only')

#     # audDep = 3
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='soma only')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")

#     # audDep = 4
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='gKLT 1')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")

#     # audDep = 5
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='gKLT 1.5')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")







#     ipi = 1.33
#     A = 70.
#     fMod = 0
#     nRep = 100

#     ax = fig.add_subplot(gs[0,3])    
#     audDep = 0
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control',color='k')
    
#     audDep = 1
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='deprived',color='tab:gray')
#     ax.set_ylim([0,250])
#     # ax.set_xlabel('time difference (ms)')
#     ax.set_title('(D) %i pps' %int(10*np.round(.1*1000/ipi)))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)




#     # audDep = 2
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='AIS only')

#     # audDep = 3
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='soma only')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")

#     # audDep = 4
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='gKLT 1')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")

#     # audDep = 5
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='gKLT 1.5')
#     # ax.legend(frameon=False,loc='upper left',fontsize="7.5")








#     ipi = 1
#     A = 80.
#     fMod = 0
#     nRep = 100

#     ax = fig.add_subplot(gs[0,4])    
#     audDep = 0
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control',color='k')
    
#     audDep = 1
#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     ax.errorbar(x,np.mean(y,0),np.std(y,0),label='deprived',color='tab:gray')
#     ax.set_ylim([0,250])
#     # ax.set_xlabel('time difference (ms)')
#     ax.set_title('(E) %i pps' %int(1000/ipi))
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)




#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/tuningCurveFig.eps', format='eps')





# # summary ITD figure. FIGURE 6.
# def summaryITDfigure():

#     fig = plt.figure(figsize=(7.1, 2.1),layout='constrained')
#     gs = GridSpec(1, 5, figure=fig)

#     # parameters
#     nNeuron = 10
#     synStrength = 21

#     fMod = 0
#     nRep = 100

#     col = ['k','tab:gray'] # plt.rcParams['axes.prop_cycle'].by_key()['color']
#     shade1 = [.2,.2,.2]
#     shade2 = [.7,.7,.7]
#     # print(col)
#     # shade1 = [.73,.82,.9]
#     # shade2 = [.9,.82,.73]

#     ipi = 10
#     ax = fig.add_subplot(gs[0,0])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[0])
#     ax.plot(x,y0,color=col[0],label='control')
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[1])
#     ax.plot(x,y0,color=col[1],label='deprived')
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(A) %i pps' %int(1000/ipi))
#     ax.set_ylim([0,420])
#     ax.set_ylabel('firing rate (Hz)')
#     ax.legend(frameon=False,loc='upper left',fontsize='8')
#     ax.spines[['right', 'top']].set_visible(False)
  





#     ipi = 4
#     ax = fig.add_subplot(gs[0,1])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[0])
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[1])
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(B) %i pps' %int(1000/ipi))
#     ax.set_ylim([0,420])
#     # ax.set_xlabel('current (nA)')
#     ax.tick_params(axis='y',label1On=False)
#     ax.spines[['right', 'top']].set_visible(False)










#     ipi = 2
#     ax = fig.add_subplot(gs[0,2])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[0])
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[1])
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(C) %i pps' %int(1000/ipi))
#     ax.set_ylim([0,420])
#     ax.set_xlabel('current (nA)')
#     ax.tick_params(axis='y',label1On=False)
#     ax.spines[['right', 'top']].set_visible(False)





#     # audDep = 3
#     # itd = 0
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y0 = np.mean(y,0)

#     # itd = 1
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y1 = np.mean(y,0)
#     # ax.plot(x,y1,x,y0)
#     # ax.fill_between(x,y1,y0)
#     # ax.set_title('(C) %i pps' %int(1000/ipi))
#     # ax.set_ylim([0,420])
#     # ax.set_xlabel('current (nA)')
#     # ax.tick_params(axis='y',label1On=False)
#     # ax.spines[['right', 'top']].set_visible(False)









#     ipi = 1.33
#     ax = fig.add_subplot(gs[0,3])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 0.66
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[0])
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 0.66
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[1])
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(D) %i pps' %int(10*np.round(.1*1000/ipi)))
#     ax.set_ylim([0,420])
#     # ax.set_xlabel('current (nA)')
#     ax.tick_params(axis='y',label1On=False)
#     ax.spines[['right', 'top']].set_visible(False)





#     # audDep = 5
#     # itd = 0
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y0 = np.mean(y,0)

#     # itd = 0.66
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y1 = np.mean(y,0)
#     # ax.plot(x,y1,x,y0)
#     # ax.fill_between(x,y1,y0)
#     # ax.set_title('(D) %i pps' %int(10*np.round(.1*1000/ipi)))
#     # ax.set_ylim([0,420])
#     # # ax.set_xlabel('current (nA)')
#     # ax.tick_params(axis='y',label1On=False)
#     # ax.spines[['right', 'top']].set_visible(False)









#     ipi = 1
#     ax = fig.add_subplot(gs[0,4])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 0.5
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[0])
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 0.5
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,x,y0,color=col[1])
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(E) %i pps' %int(1000/ipi))
#     ax.set_ylim([0,420])
#     # ax.set_xlabel('current (nA)')
#     ax.tick_params(axis='y',label1On=False)
#     ax.spines[['right', 'top']].set_visible(False)


#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/summaryTuningFig.eps', format='eps')




# # FIGURE 7: ENVELOPE ITD TUNING CURVE
# def envelopeFigure():

#     fig = plt.figure(figsize=(7.1, 2.4),layout='constrained')
#     gs = GridSpec(1, 4, figure=fig)

#     # parameters
#     nNeuron = 10
#     synStrength = 21

#     fMod = 0
#     nRep = 100

#     col = ['k','tab:gray'] # plt.rcParams['axes.prop_cycle'].by_key()['color']
#     shade1 = [.2,.2,.2]
#     shade2 = [.7,.7,.7]

#     # col = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     # shade1 = [.73,.82,.9]
#     # shade2 = [.9,.82,.73]

#     fMod = 0
#     ipi = 0.33
#     ax = fig.add_subplot(gs[0,0])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 0.16
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[0])
#     ax.plot(x,y0,color=col[0],label='control')
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 0.16
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[1])
#     ax.plot(x,y0,color=col[1],label='deprived')
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(A) %i pps' %int(100*np.round(.01*1000/ipi)))
#     ax.text(79,299,'$f_m = $ %i Hz' %fMod, fontsize=10)
#     ax.set_ylim([0,320])
#     ax.set_xlabel('max. current (nA)')
#     ax.set_ylabel('firing rate (Hz)')
#     ax.legend(frameon=False,loc='upper left',fontsize='8',bbox_to_anchor=(0.1,0.8))
#     ax.spines[['right', 'top']].set_visible(False)
  




#     fMod = 100
#     ax = fig.add_subplot(gs[0,1])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[0])
#     ax.plot(x,y0,color=col[0],label='control')
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[1])
#     ax.plot(x,y0,color=col[1],label='deprived')
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(B) $f_m = $ %i Hz' %fMod)
#     ax.set_ylim([0,320])
#     ax.set_xlabel('max. current (nA)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)






#     fMod = 200
#     ax = fig.add_subplot(gs[0,2])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[0])
#     ax.plot(x,y0,color=col[0],label='control')
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[1])
#     ax.plot(x,y0,color=col[1],label='deprived')
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(C) $f_m = $ %i Hz' %fMod)
#     # ax.text(46,292,'$f_m = $ %i Hz' %fMod)
#     ax.set_ylim([0,320])
#     ax.set_xlabel('max. current (nA)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)
  





#     fMod = 300
#     ax = fig.add_subplot(gs[0,3])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[0])
#     ax.plot(x,y0,color=col[0],label='control')
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[1])
#     ax.plot(x,y0,color=col[1],label='deprived')
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(D) $f_m = $ %i Hz' %fMod)
#     # ax.text(46,292,'$f_m = $ %i Hz' %fMod)
#     ax.set_ylim([0,320])
#     ax.set_xlabel('max. current (nA)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)






#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/envelopeFig.eps', format='eps')








# # FIGURE OLD: ENVELOPE ITD TUNING CURVE
# def envelopeFigureOLD():

#     fig = plt.figure(figsize=(3.5, 2.4),layout='constrained')
#     gs = GridSpec(1, 2, figure=fig)

#     # parameters
#     nNeuron = 10
#     synStrength = 21

#     fMod = 0
#     nRep = 100

#     col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
#     shade1 = [.73,.82,.9]
#     shade2 = [.9,.82,.73]

#     fMod = 100
#     ipi = 1
#     ax = fig.add_subplot(gs[0,0])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[0])
#     ax.plot(x,y0,color=col[0],label='control')
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[1])
#     ax.plot(x,y0,color=col[1],label='deprived')
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(A) %i pps' %int(1000/ipi))
#     ax.text(46,292,'$f_m = $ %i Hz' %fMod)
#     ax.set_ylim([0,320])
#     ax.set_xlabel('current (nA)')
#     ax.set_ylabel('firing rate (Hz)')
#     ax.legend(frameon=False,loc='upper left',fontsize='8',bbox_to_anchor=(0.1,0.9))
#     ax.spines[['right', 'top']].set_visible(False)
  



#     fMod = 250
#     ipi = 0.4
#     ax = fig.add_subplot(gs[0,1])    
#     audDep = 0
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[0])
#     ax.plot(x,y0,color=col[0],label='control')
#     ax.fill_between(x,y1,y0,color=shade1)

#     audDep = 1
#     itd = 0
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y0 = np.mean(y,0)

#     itd = 1
#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     file = open(fileName,'r')
#     D = np.loadtxt(file)
#     file.close()
#     x = D[0,:]
#     y = D[1:,:]
#     y1 = np.mean(y,0)
#     ax.plot(x,y1,color=col[1])
#     ax.plot(x,y0,color=col[1],label='deprived')
#     ax.fill_between(x,y1,y0,color=shade2)
#     ax.set_title('(B) %i pps' %int(1000/ipi))
#     ax.text(47,292,'$f_m = $ %i Hz' %fMod)
#     # ax.set_xlim([40,60])
#     ax.set_ylim([0,320])
#     ax.set_xlabel('current (nA)')
#     # ax.set_ylabel('firing rate (Hz)')
#     ax.spines[['right', 'top']].set_visible(False)
#     ax.tick_params(axis='y',label1On=False)







#     # ax = fig.add_subplot(gs[1,0])    
#     # ipi = 10
#     # A = 51.
#     # fMod = 0
#     # nRep = 100
#     # audDep = 0
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')
    
#     # audDep = 1
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')
    
#     # audDep = 0
#     # fMod = 100
#     # ipi = 1
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')
    
#     # audDep = 1
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')
#     # ax.set_ylim([0,200])
#     # ax.set_title('(C)')
#     # ax.spines[['right', 'top']].set_visible(False)
#     # ax.set_xlabel('time difference (ms)')






#     # ax = fig.add_subplot(gs[1,1])    
#     # ipi = 4
#     # A = 52.
#     # fMod = 0
#     # nRep = 100
#     # audDep = 0
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')
    
#     # audDep = 1
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')


#     # audDep = 0
#     # A = 49
#     # fMod = 250
#     # ipi = 0.4
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')
    
#     # audDep = 1
#     # fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # ax.errorbar(x,np.mean(y,0),np.std(y,0),label='control')
#     # ax.set_ylim([0,200])
#     # ax.set_title('(D)')
#     # ax.set_xlabel('time difference (ms)')
#     # ax.spines[['right', 'top']].set_visible(False)
#     # ax.tick_params(axis='y',label1On=False)





#     # fMod = 100
#     # ipi = 1
#     # audDep = 0
#     # itd = 0
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:18]
#     # y = D[1:,:18]
#     # y0 = np.mean(y,0)

#     # itd = 1
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:18]
#     # y = D[1:,:18]
#     # y1 = np.mean(y,0)
#     # ax.plot(x,y1,color=col[0])
#     # ax.plot(x,y0,color=col[0],label='control')
#     # ax.fill_between(x,y1,y0,color=shade1)

#     # audDep = 1
#     # itd = 0
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y0 = np.mean(y,0)

#     # itd = 1
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y1 = np.mean(y,0)
#     # ax.plot(x,y1,color=col[1])
#     # ax.plot(x,y0,color=col[1],label='deprived')
#     # ax.fill_between(x,y1,y0,color=shade2)
#     # ax.set_title('(A) %i pps' %int(1000/ipi))
#     # ax.text(46,212,'$f_m = $ %i Hz' %fMod)
#     # ax.set_ylim([0,230])
#     # ax.set_xlabel('current (nA)')
#     # ax.set_ylabel('firing rate (Hz)')
#     # ax.legend(frameon=False,loc='upper left',fontsize='8',bbox_to_anchor=(0.1,0.85))
#     # ax.spines[['right', 'top']].set_visible(False)
  



#     # fMod = 200
#     # ipi = 0.5
#     # ax = fig.add_subplot(gs[1,1])    
#     # audDep = 0
#     # itd = 0
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y0 = np.mean(y,0)

#     # itd = 1
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y1 = np.mean(y,0)
#     # ax.plot(x,y1,color=col[0])
#     # ax.plot(x,y0,color=col[0],label='control')
#     # ax.fill_between(x,y1,y0,color=shade1)

#     # audDep = 1
#     # itd = 0
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y0 = np.mean(y,0)

#     # itd = 1
#     # fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{itd:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'
#     # file = open(fileName,'r')
#     # D = np.loadtxt(file)
#     # file.close()
#     # x = D[0,:]
#     # y = D[1:,:]
#     # y1 = np.mean(y,0)
#     # ax.plot(x,y1,color=col[1])
#     # ax.plot(x,y0,color=col[1],label='deprived')
#     # ax.fill_between(x,y1,y0,color=shade2)
#     # ax.set_title('(B) %i pps' %int(1000/ipi))
#     # ax.text(47,212,'$f_m = $ %i Hz' %fMod)
#     # # ax.set_xlim([40,60])
#     # ax.set_ylim([0,230])
#     # ax.set_xlabel('current (nA)')
#     # # ax.set_ylabel('firing rate (Hz)')
#     # ax.spines[['right', 'top']].set_visible(False)
#     # ax.tick_params(axis='y',label1On=False)






#     if saveFig:
#         fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/envelopeFig.eps', format='eps')







# def runFiringRate(ipi,ITD,fMod,audDep,Amin,Amax,nA,nRep,runAN):


#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 300
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)

#     nNeuron = 10
#     synStrength = 21

#     nSpike = np.zeros([nRep,nA])
#     Avec = np.linspace(Amin,Amax,nA)

#     fileName = 'rateData/rate_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'

#     for iRep in range(nRep):
#         for i in range(nA):
#             # ipi = 4.
#             A = Avec[i]
#             # ITD = 0
#             iterNum = iRep

#             if runAN:
#                 sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)
#                 ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#                 np.savetxt(ANfileName,gSum,fmt="%f")

#             # run MSO model
#             writeData = 0
#             MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)

#             # count MSO spikes
#             MSOfileName = 'MSOSpike.txt'
#             nSpike[iRep,i] = getSpikeCount(MSOfileName)

#         #rate[i,iRep] = np.mean(nSpike*(1000/stimDuration),1)
#     rate = nSpike*1000/stimDuration
#     np.savetxt(fileName, np.concatenate(( [Avec], rate)) )  


# def runTuningCurve(ipi,A,fMod,audDep,ITDmin,ITDmax,nITD,nRep,runAN):


#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 300
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)

#     nNeuron = 10
#     synStrength = 21

#     nSpike = np.zeros([nRep,nITD])
#     ITDvec = np.linspace(ITDmin,ITDmax,nITD)

#     fileName = 'rateData/itd_ipi' + f'{ipi:.2f}' + '_A'+f'{A:.2f}' + '_fMod'+f'{fMod:.0f}'+'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}' + '_nRep'+f'{nRep:.0f}'+ '.txt'

#     for iRep in range(nRep):
#         for i in range(nITD):
#             # ipi = 4.
#             ITD = ITDvec[i]
#             # ITD = 0
#             iterNum = iRep

#             if runAN:
#                 sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)
#                 ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#                 np.savetxt(ANfileName,gSum,fmt="%f")

#             # run MSO model
#             writeData = 0
#             MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)

#             # count MSO spikes
#             MSOfileName = 'MSOlib/MSOSpike'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#             nSpike[iRep,i] = getSpikeCount(MSOfileName)

#         #rate[i,iRep] = np.mean(nSpike*(1000/stimDuration),1)
#     rate = nSpike*1000/stimDuration
#     np.savetxt(fileName, np.concatenate(( [ITDvec], rate)) )  





# # OLD MSO figure. Figure 3.
# def MSOfigure():
    
#     fig = plt.figure(figsize=(18, 10),layout='constrained')
#     gs = GridSpec(3, 6, figure=fig)
    
#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 100
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
#     # nt = np.size(t)

#     # run AN model
#     A = 55
#     ITD = 0
#     nNeuron = 10
#     iterNum = 1 
#     fMod = 0
#     # MSO parameters
#     audDep = 0
#     writeData = 1
#     synStrength = 21
    
#     # runVoltage =1
#     # if runVoltage:
#     #     # 250 pps
#     #     ipi = 4
#     #     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#     #     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     #     np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#     #     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     #     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     #     file = open(MSOfileName,"r")
#     #     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     #     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     #     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     #     file.close()
#     #     ax = fig.add_subplot(gs[0,0:2])
#     #     ax.plot(t,v1,'-')
#     #     ax.plot(t,v2,'-')
#     #     ax.set_xlim(0,stimDuration)
#     #     ax.set_ylim(-80,35)
#     #     ax.set_xlabel('time (ms)')
#     #     ax.set_ylabel('voltage (mV)')
#     #     ax.spines[['right', 'top']].set_visible(False)

#     #     # 500 pps
#     #     ipi = 2
#     #     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#     #     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     #     np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#     #     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     #     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     #     file = open(MSOfileName,"r")
#     #     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     #     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     #     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     #     file.close()
#     #     ax = fig.add_subplot(gs[0,2:4])
#     #     ax.plot(t,v1,'-')
#     #     ax.plot(t,v2,'-')
#     #     ax.set_xlim(0,stimDuration)
#     #     ax.set_ylim(-80,35)
#     #     ax.set_xlabel('time (ms)')
#     #     ax.set_ylabel('voltage (mV)')
#     #     ax.spines[['right', 'top']].set_visible(False)


#     #     # 1000 pps
#     #     ipi = 1
#     #     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#     #     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     #     np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#     #     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#     #     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#     #     file = open(MSOfileName,"r")
#     #     t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#     #     v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#     #     v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#     #     file.close()
#     #     ax = fig.add_subplot(gs[0,4:6])
#     #     ax.plot(t,v1,'-')
#     #     ax.plot(t,v2,'-')
#     #     ax.set_xlim(0,stimDuration)
#     #     ax.set_ylim(-80,35)
#     #     ax.set_xlabel('time (ms)')
#     #     ax.set_ylabel('voltage (mV)')
#     #     ax.spines[['right', 'top']].set_visible(False)

#     IPIvec = [10, 5, 4, 2, 1]
#     Amin = 40
#     Amax = [55,60,65,75,75]
#     # AforITD = [51,53,55,60,65]
#     AforITD = [51,51.8,51.5,55.2,62]
#     nA = 10
#     nITD = 11
#     nRep = 20#00
#     ax1 = fig.add_subplot(gs[1:3,0:3])
#     ax2 = fig.add_subplot(gs[1:3,3:6])
#     for i in range(len(IPIvec)):

#         x1,y1 = firingRate(Amin,Amax[i],nA,IPIvec[i],0,nRep,audDep)
#         ax1.plot(x1,y1,'-o',lw=2,label=str(IPIvec[i]))    
#         x2,y2 = ITDtuning(0,np.minimum(1.5,IPIvec[i]/2),nITD,IPIvec[i],AforITD[i],0,nRep,audDep)
#         ax2.plot(x2,y2,'-o',lw=2,label=str(IPIvec[i]))    
#     ax1.spines[['right', 'top']].set_visible(False)
#     ax1.set_xlabel('current (nA)')
#     ax1.set_ylabel('firing rate (Hz)')
#     ax1.set_xlim([40,75])
#     ax1.set_ylim([0,450])
#     ax1.legend()
#     ax2.spines[['right', 'top']].set_visible(False)
#     ax2.set_xlabel('time difference (ms)')
#     ax2.set_ylabel('firing rate (Hz)')
#     ax2.set_xlim([0,1.5])
#     ax2.set_ylim([0,200])
#     ax2.legend()

#     plt.show()




# # sound deprived. Figure 4.
# def audDepfigure():

#     fig = plt.figure(figsize=(18, 10),layout='constrained')
#     gs = GridSpec(5, 3, figure=fig)
    
#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 50
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
#     # nt = np.size(t)

#     # run AN model
#     A = 55
#     fMod = 0
#     ITD = 0
#     nNeuron = 10
#     iterNum = 1 

#     # MSO parameters
#     audDep = 1 ######### SOUND DEPRIVED MODEL ########
#     writeData = 1
#     synStrength = 21
    
#     runVoltage =1
#     if runVoltage:
#         # 250 pps
#         ipi = 4
#         sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#         MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#         file = open(MSOfileName,"r")
#         t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#         v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#         v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#         file.close()
#         ax = fig.add_subplot(gs[0,0])
#         ax.plot(t,v1,'-')
#         ax.plot(t,v2,'-')
#         ax.set_xlim(0,stimDuration)
#         ax.set_ylim(-80,45)
#         ax.set_xlabel('time (ms)')
#         ax.set_ylabel('voltage (mV)')
#         ax.spines[['right', 'top']].set_visible(False)

#         # 500 pps
#         ipi = 2
#         sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#         MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#         file = open(MSOfileName,"r")
#         t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#         v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#         v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#         file.close()
#         ax = fig.add_subplot(gs[0,1])
#         ax.plot(t,v1,'-')
#         ax.plot(t,v2,'-')
#         ax.set_xlim(0,stimDuration)
#         ax.set_ylim(-80,45)
#         ax.set_xlabel('time (ms)')
#         ax.set_ylabel('voltage (mV)')
#         ax.spines[['right', 'top']].set_visible(False)


#         # 1000 pps
#         ipi = 1
#         sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron) # run AN
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         np.savetxt(ANfileName,gSum,fmt="%f") # save to AN file    
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep) # run MSO model
#         MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#         file = open(MSOfileName,"r")
#         t = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(0)) - delay
#         v1 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(1))
#         v2 = np.genfromtxt(MSOfileName, delimiter=' ', usecols=(2))
#         file.close()
#         ax = fig.add_subplot(gs[0,2])
#         ax.plot(t,v1,'-')
#         ax.plot(t,v2,'-')
#         ax.set_xlim(0,stimDuration)
#         ax.set_ylim(-80,45)
#         ax.set_xlabel('time (ms)')
#         ax.set_ylabel('voltage (mV)')
#         ax.spines[['right', 'top']].set_visible(False)

#     IPIvec = [4, 2, 1]
#     Amin = 40
#     Amax = [65,75,75]
#     # AforITD = [51,53,55,60,65]
#     AforITD = [51.5,55.2,62]
#     nA = 10
#     nITD = 11
#     nRep = 20#0#00
#     for i in range(len(IPIvec)):

#         ax = fig.add_subplot(gs[1:3,i])
#         x0,y0 = firingRate(Amin,Amax[i],nA,IPIvec[i],0,nRep,0)
#         ax.plot(x0,y0,'-ok',lw=2,label=str(IPIvec[i]))    

#         x1,y1 = firingRate(Amin,Amax[i],nA,IPIvec[i],0,nRep,1)
#         ax.plot(x1,y1,'-o',lw=2,label=str(IPIvec[i])) 

#         ax = fig.add_subplot(gs[3:5,i])
#         x2,y2 = ITDtuning(0,np.minimum(1.5,IPIvec[i]/2),nITD,IPIvec[i],AforITD[i],0,nRep,0)
#         ax.plot(x2,y2,'-ok',lw=2,label=str(IPIvec[i]))    

#         x3,y3 = ITDtuning(0,np.minimum(1.5,IPIvec[i]/2),nITD,IPIvec[i],AforITD[i],0,nRep,1)
#         ax.plot(x3,y3,'-o',lw=2,label=str(IPIvec[i]))    

#     # ax1.spines[['right', 'top']].set_visible(False)
#     # ax1.set_xlabel('current (nA)')
#     # ax1.set_ylabel('firing rate (Hz)')
#     # ax1.set_xlim([40,75])
#     # ax1.set_ylim([0,450])
#     # ax1.legend()
#     # ax2.spines[['right', 'top']].set_visible(False)
#     # ax2.set_xlabel('time difference (ms)')
#     # ax2.set_ylabel('firing rate (Hz)')
#     # ax2.set_xlim([0,1.5])
#     # ax2.set_ylim([0,200])
#     # ax2.legend()

#     plt.show()



# # # Envelope ITD. Figure 6.
# # def envelopeITDfigure():

# #     # time parameters
# #     tStart = 0; 
# #     delay = 10;
# #     stimDuration = 300
# #     dt = 0.002
# #     tEnd = 2*delay + stimDuration; 
# #     t = np.arange(tStart,tEnd,dt)
   
# #     # run AN model
# #     N = 11
# #     ITDvec = np.linspace(0.,1.5,N)
# #     nSpike = np.zeros(N)
# #     nNeuron = 10    
# #     A = 51.8
# #     audDep = 0
# #     fMod = 0
# #     nRep = 2

# #     ipi = 5
# #     fMod = 0
# #     x1,y1 = ITDtuning(0,1.5,N,ipi,A,fMod,nRep,0)
# #     x2,y2 = ITDtuning(0,1.5,N,ipi,A,fMod,nRep,1)
# #     plt.plot(x1,y1)
# #     plt.plot(x2,y2)

# #     ipi = 0.5
# #     fMod = 200
# #     x1,y1 = ITDtuning(0,1.5,N,ipi,A,fMod,nRep,0)
# #     x2,y2 = ITDtuning(0,1.5,N,ipi,A,fMod,nRep,1)
# #     plt.plot(x1,y1)
# #     plt.plot(x2,y2)


# # responses to current inputs
# def currentFigure():

#     # time parameters
#     tStart = 0; 
#     delay = 5;
#     stimDuration = 50
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
#     nt = np.size(t)
#     ipi = 0
#     ITD =0 
#     nNeuron = 0
#     iterNum = 0
#     synStrength = 0

#     Avec = np.arange(400,2201,300) # for plot
#     # Avec = np.arange(1350,1401,5) #  for phasic threshold
#     # Avec = np.arange(495,516,5) #  for tonic threshold
#     # Avec = np.arange(1200,1251,10) #  for tonic multispike

# # fig8 = plt.figure(constrained_layout=False)
# # gs1 = fig8.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.05)

#     fig = plt.figure(figsize=(7.1, 4),constrained_layout=False)
#     gs = fig.add_gridspec(nrows=1, ncols=2, left=0.03, right=0.9, bottom=0.15, wspace=0.2 )
#     # gs = GridSpec(1, 2, figure=fig)
#     ax1 = fig.add_subplot(gs[0])
#     ax2 = fig.add_subplot(gs[1])
    
#     for iA in range(len(Avec)):
#         A = Avec[iA]
        
#         #### CONTROL #####
#         audDep = 0
#         # create "AN" file for current injection
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         gSum = np.zeros_like(t)
#         for i in range(nt):  
#             if (t[i]>=delay and t[i]<(delay+stimDuration)):
#                 gSum[i] = A         
#         np.savetxt(ANfileName,gSum,fmt="%f")

#         # run MSO model
#         writeData = 1
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)
#         MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
        
#         t,v1,v2 = getMSOvoltage(MSOfileName)
#         ax1.plot(t,v2+iA*100,'k',lw=1)
#         ax1.set_xlabel('time (ms)')
#         ax1.set_title('A) control')
#         ax1.spines[['left', 'right', 'top']].set_visible(False)
#         ax1.tick_params(axis='y',which='both',left=False,labelleft=False)

#         # if iA==0:
#         #     ax[0].plot(t,v1,'k')
#         #   ax[0].set_xlim([52,60])

#         #### SOUND DEPRIVED #####
#         audDep = 1
#         # create "AN" file for current injection
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         gSum = np.zeros_like(t)
#         for i in range(nt):  
#             if (t[i]>=delay and t[i]<(delay+stimDuration)):
#                 gSum[i] = A         
#         np.savetxt(ANfileName,gSum,fmt="%f")

#         # run MSO model
#         writeData = 1
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)
#         MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
        
#         t,v1,v2 = getMSOvoltage(MSOfileName)
#         ax2.plot(t,v2+iA*100,'k',lw=1)
#         ax2.set_title('B) deprived')
#         ax2.set_xlabel('time (ms)')
#         ax2.spines[['left', 'right', 'top']].set_visible(False)
#         ax2.tick_params(axis='y',which='both',left=False,labelleft=False)
#         ax2.text(62, -50+iA*100,'%i pA' %int(A) )
#         # if iA==0:
#         #     ax[0,0].plot(t,v1)
#         #     ax[0,0].set_xlim([52,60])

#         if saveFig:
#             fig.savefig('/Users/jgoldwy1/Research/twoCptCI/latex/currentFiringType.eps', format='eps')

# def firingRate(Amin,Amax,nA,ipi,ITD,nRep,audDep):

#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 300
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)

#     fMod = 0

#     nNeuron = 10
#     synStrength = 21
#     # run AN model
#     # nRep = 100
#     # nA   = 20
#     Avec = np.linspace(Amin,Amax,nA)
#     nSpike = np.zeros([nA,nRep])
#     for iRep in range(nRep):
#         for i in range(nA):
#             # ipi = 4.
#             A = Avec[i]
#             # ITD = 0
#             iterNum = i
#             sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)

#             # save to AN file
#             ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#             np.savetxt(ANfileName,gSum,fmt="%f")

#             # run MSO model
#             writeData = 0
#             MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)

#             # count MSO spikes
#             MSOfileName = 'MSOlib/MSOSpike'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#             nSpike[i,iRep] = getSpikeCount(MSOfileName)

#     rate = np.mean(nSpike*(1000/stimDuration),1) 
#     return Avec,rate
    

# def ITDtuning(ITDmin,ITDmax,nITD,ipi,A,fMod,nRep,audDep):

#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 300
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
   
#     # fMod = 0

#     nNeuron = 10
#     synStrength = 21

#     # run AN model
#     # nRep = 100
#     ITDvec = np.linspace(ITDmin,ITDmax,nITD)
#     nSpike = np.zeros([nITD,nRep])
#     for iRep in range(nRep):
#         for i in range(nITD):
#             # ipi = 4.
#             # A = Avec[i]
#             ITD = ITDvec[i]
#             iterNum = i
#             sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)

#             # save to AN file
#             ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#             np.savetxt(ANfileName,gSum,fmt="%f")

#             # run MSO model
#             writeData = 0
#             MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)

#             # count MSO spikes
#             MSOfileName = 'MSOlib/MSOSpike'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#             nSpike[i,iRep] = getSpikeCount(MSOfileName)

#     rate = np.mean(nSpike*(1000/stimDuration),1) 
#     return ITDvec,rate


# def oldfiringRate(Amin,Amax,nA,ipi,ITD):

#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 300
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
   
#     # run AN model
#     nRep = 100
#     nA   = 20
#     Avec = np.linspace(Amin,Amax,nA)
#     nSpike = np.zeros([nA,nRep])
#     for iRep in range(nRep):
#         for i in range(nA):
#             # ipi = 4.
#             A = Avec[i]
#             # ITD = 0
#             nNeuron = 10
#             iterNum = i
#             sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)

#             # save to AN file
#             ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#             np.savetxt(ANfileName,gSum,fmt="%f")

#             # run MSO model
#             writeData = 0
#             audDep = 1
#             synStrength = 21
#             MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)

#             # count MSO spikes
#             MSOfileName = 'MSOlib/MSOSpike'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#             nSpike[i,iRep] = getSpikeCount(MSOfileName)

#     rate = np.mean(nSpike*(1000/stimDuration),1) 
#     return Avec,rate
#     # plt.plot(Avec,nSpike,'-o',lw=3)
#     # plt.show()



# def main(A,ipi,fMod,ITD,audDep):

#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 200
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
#     # nt = np.size(t)

#     # run AN model
#     # ipi = .5
#     # A = 51
#     # ITD = 1.5
#     # fMod = 0
#     nNeuron = 10#0
#     iterNum = 0
#     sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)

#     # save to AN file
#     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     np.savetxt(ANfileName,gSum,fmt="%f")

#     # run MSO model
#     writeData = 1
#     # audDep = 1
#     synStrength = 21
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
    
#     # plt.plot(t,gSum)
#     fig, ax = plt.subplots(2,1)
#     t,v1,v2 = getMSOvoltage(MSOfileName)
#     ax[0].plot(t,gSum)
#     ax[1].plot(t,v1,t,v2)
#     # plt.show()



# def currentInjection(A,audDep):

#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 200
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
#     nt = np.size(t)
#     ipi = 0
#     ITD =0 
#     # run AN model
#     # ipi = .5
#     # A = 51
#     # ITD = 1.5
#     # fMod = 100
#     nNeuron = 0
#     iterNum = 0
#     # sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)

#     # save to AN file
#     ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#     gSum = np.zeros_like(t)
#     for i in range(nt):  
#         if (t[i]>=delay and t[i]<(delay+stimDuration)):
#             gSum[i] = A 
        
#     np.savetxt(ANfileName,gSum,fmt="%f")

#     # run MSO model
#     writeData = 1
#     # audDep = 0
#     synStrength = 0
#     MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)
#     MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
    
#     # plt.plot(t,gSum)
#     fig, ax = plt.subplots(2,1)
#     t,v1,v2 = getMSOvoltage(MSOfileName)
#     ax[0].plot(t,gSum)

#     ax[1].plot(t,v1,t,v2)
#     # plt.show()






# def testPlots():
#     q = np.zeros(300)
#     for i in range(50):
#         s = ANmodel(300,1,50*(1+.05*10,fMod))
#         # print(N)
#         h = plt.hist(s,np.arange(0,301),fMod)
#         q += h[0]
#     plt.plot(q)
#     print(h[1])


#     plotANfiringRate = 0
#     if plotANfiringRate:
#         fig, ax = plt.subplots(1,2)

#         nA = 15

#         Avec = np.linspace(-10,15,nA)*.05*50+50 #85,nA)
#         N = np.zeros(nA)
#         for i in range(nA):
#             s = ANmodel(100,10,Avec[i],fMod)
#             if len(s)>0:
#                 print(s)
#                 s = s[(s>200) & (s<300)]
#             N[i] = len(s)
#         ax[0].plot(Avec,N/1,'-o')
#         ax[1].plot(Avec,N/100,'-o')

#         # Avec = np.linspace(40,95,nA)
#         N = np.zeros(nA)
#         for i in range(nA):
#             s = ANmodel(250,4,Avec[i],fMod)
#             if len(s)>0:
#                 s = s[(s>200) & (s<300)]
#             N[i] = len(s)
#         ax[0].plot(Avec,N/1,'-o')
#         ax[1].plot(Avec,N/250,'-o')

#         # Avec = np.linspace(40,150,nA)
#         N = np.zeros(nA)
#         for i in range(nA):
#             s = ANmodel(500,2,Avec[i],fMod)
#             if len(s)>0:
#                 s = s[(s>200) & (s<300)]
#             N[i] = len(s)
#         ax[0].plot(Avec,N/1,'-o')
#         ax[1].plot(Avec,N/500,'-o')

#         # Avec = np.linspace(40,150,nA)
#         N = np.zeros(nA)
#         for i in range(nA):
#             s = ANmodel(1000,1,Avec[i],fMod)
#             if len(s)>0:
#                 s = s[(s>200) & (s<300)]
#             N[i] = len(s)
#         ax[0].plot(Avec,N/1,'-o')
#         ax[1].plot(Avec,N/1000,'-o')

#     plt.show()







#     # # adapted from sylvia and anna code

#     # ipi = ipiVal
#     # pps = 1000/ipi
#     # npulse = int(np.ceil(tEnd*pps/1000.))
    
#     # for iNeuron in range(nNeuron):

#     #     spikeTrainL, spikeTimeL = multiPulse(D,npulse,istimStrength,ipi)
#     #     spikeTrain, spikeTime = multiPulse(D,npulse,istimStrength,ipi)
#     #     spikeTimeR = spikeTime + ITD
#     #     # print(spikeTime)
#     #     # loop through time bins to create 0s and 1s in small time bins
#     #     if len(spikeTimeL) != 0: # ONLY START THE FOR LOOP IF THE SPIKE TIME ARRAY IS NOT EMPTY
#     #         j = 0 # a counter to move through the spikeTime array
#     #         for i in np.arange(nt)-1:
#     #             if (spikeTimeL[j]>=t[i]) and (spikeTimeL[j]<t[i+1]):
#     #                 spike_arrL[i] += 1.
#     #                 if j < (len(spikeTimeL)-1):
#     #                     j += 1
#     #                 else:
#     #                     break
#     #     if len(spikeTimeR) != 0:
#     #         j = 0
#     #         for i in np.arange(nt)-1:
#     #             if (spikeTimeR[j]>=t[i]) and (spikeTimeR[j]<t[i+1]):
#     #                 spike_arrR[i] += 1.
#     #                 if j < (len(spikeTimeR)-1):
#     #                     j += 1
#     #                 else:
#     #                     break
        
#     # ng = nt # change this later
#     # gFunc = (1./0.21317) * (np.exp(-t[:ng]/.18) - np.exp(-t[:ng]/.1) )
  
#     # cL = np.convolve(gFunc,spike_arrL)  # convolve the EPSG with the spike events
#     # cR = np.convolve(gFunc,spike_arrR)  # convolve the EPSG with the spike events

#     # gSynL = cL[:nt]
#     # gSynR = cR[:nt]
#     # gSum = gSynL + gSynR

#     # ANfileName = 'ANlib/ANfile'+'_I'+f'{istimStrength:.2f}' + '_ipi' + f'{ipiVal:.0f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+ iterNum + '.txt'
#     # np.savetxt(ANfileName,gSum,fmt="%f")

#     # return gSum #ANfileName



# # def MSOmodel(gh,Vrest,gKLT,gNa,gKHT,glk1,gTot2,gAx):


# def ITDcurve(A,ipi,audDep):

#     # time parameters
#     tStart = 0; 
#     delay = 10;
#     stimDuration = 300
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)

#     fMod = 0
   
#     # run AN model
#     N = 11
#     ITDvec = np.linspace(0.,np.minimum(1.7, ipi/2),N)
#     nSpike = np.zeros(N)
#     # input: ipi 
#     # input: A
#     # input: audDep
#     for i in range(N):
#         ITD = ITDvec[i]
#         nNeuron = 10
#         iterNum = i
#         sL,sR,gSum = makeEPSG(stimDuration,dt,ipi,A,ITD,fMod,nNeuron)

#         # save to AN file
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         np.savetxt(ANfileName,gSum,fmt="%f")

#         # run MSO model
#         writeData = 0
#         synStrength = 21
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)

#         # count MSO spikes
#         MSOfileName = 'MSOlib/MSOSpike'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
#         nSpike[i] = getSpikeCount(MSOfileName)

#     plt.plot(ITDvec,nSpike,'-o',lw=3)
#     # plt.show()



# # responses to current inputs
# def testMembraneSpeed():

#     # time parameters
#     tStart = 0; 
#     delay = 5;
#     stimDuration = 50
#     dt = 0.002
#     tEnd = 2*delay + stimDuration; 
#     t = np.arange(tStart,tEnd,dt)
#     nt = np.size(t)
#     ipi = 0
#     ITD =0 
#     nNeuron = 0
#     iterNum = 0
#     synStrength = 0

#     audDep = 1

#     Avec = np.arange(-100,1,40) # for plot

#     fig = plt.figure(figsize=(7.1, 4),constrained_layout=False)
#     gs = fig.add_gridspec(nrows=1, ncols=2, left=0.03, right=0.9, bottom=0.15, wspace=0.2 )
#     # gs = GridSpec(1, 2, figure=fig)
#     ax1 = fig.add_subplot(gs[0])
#     ax2 = fig.add_subplot(gs[1])
    
#     for iA in range(len(Avec)):
#         A = Avec[iA]
        
#         #### CONTROL #####
#         audDep = 0
#         # create "AN" file for current injection
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         gSum = np.zeros_like(t)
#         for i in range(nt):  
#             if (t[i]>=delay and t[i]<(delay+stimDuration)):
#                 gSum[i] = A         
#         np.savetxt(ANfileName,gSum,fmt="%f")

#         # run MSO model
#         writeData = 1
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)
#         MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
        
#         t,v1,v2 = getMSOvoltage(MSOfileName)
#         ax1.plot(t,v2)
#         # ax1.set_xlabel('time (ms)')
#         # ax1.set_title('A) control')
#         # ax1.spines[['left', 'right', 'top']].set_visible(False)
#         # ax1.tick_params(axis='y',which='both',left=False,labelleft=False)

#         # if iA==0:
#         #     ax[0].plot(t,v1,'k')
#         #   ax[0].set_xlim([52,60])

#         #### SOUND DEPRIVED #####
#         audDep = 1
#         # create "AN" file for current injection
#         ANfileName = 'ANlib/ANfile'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+'_Iter'+f'{iterNum:.0f}' + '.txt'
#         gSum = np.zeros_like(t)
#         for i in range(nt):  
#             if (t[i]>=delay and t[i]<(delay+stimDuration)):
#                 gSum[i] = A         
#         np.savetxt(ANfileName,gSum,fmt="%f")

#         # run MSO model
#         writeData = 1
#         MSOmodel(tEnd,dt,ipi,ITD,A,synStrength,nNeuron,iterNum,writeData,audDep)
#         MSOfileName = 'MSOlib/MSOVoltage'+'_A'+f'{A:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter' + f'{iterNum:.0f}' + '.txt'
        
#         t,v1,v2 = getMSOvoltage(MSOfileName)
#         ax2.plot(t,v2)
#         # ax2.set_title('B) sound-deprived')
#         # ax2.set_xlabel('time (ms)')
#         # ax2.spines[['left', 'right', 'top']].set_visible(False)
#         # ax2.tick_params(axis='y',which='both',left=False,labelleft=False)
#         # ax2.text(62, -50+iA*100,'%i pA' %int(A) )
        

# ########## COUNT MSO SPIKES #############
# def getSpikeCount(MSOfilename):
#     file = open(MSOfilename,"r")
#     spikeTimes = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(0))
#     nSpike = spikeTimes.size
#     return nSpike


# ########## PLOTTING #############
# def plotMSOvoltage(MSOfilename):

#     # MSOfilename = 'MSOlib/MSOVoltage'+'_I'+f'{istimStrength:.2f}' + '_ipi' + f'{ipi:.2f}' + '_itd'+f'{ITD:.2f}' +'_nNeuron'+f'{nNeuron:.0f}'+ '_synStrength'+f'{synStrength:.2f}' +'_audDep'+f'{audDep:.0f}'+'_Iter'+ f'{iterNum:.0f}' + '.txt'
#     file = open(MSOfilename,"r")
#     # lines = file.readlines()
#     t = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(0))
#     v1 = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(1))
#     v2 = np.genfromtxt(MSOfilename, delimiter=' ', usecols=(2))

#     file.close()

#     plt.plot(t,v1,'-')
#     plt.plot(t,v2,'-')
#     plt.xlabel('time (ms)')
#     plt.ylabel('voltage (mV)')

#     plt.show()



    
# # Intro figure (FIGURE 1)
# # ANforIntroFigure()

# # AN figure (FIGURE 2)
# # ANfigure()

# # MSO voltage figure (FIGURE 3)
# # voltageFigure()

# # MSO firing rate (FIGURE 4)
# # firingRateFigure()

# # ITD TUNING CURVE (FIGURE 5)
# # tuningCurveFigure()

# # summary ITD (FIGURE 6)
# # summaryITDfigure() #

# # Envelope ITD Figure (FIGURE 7) #
# # envelopeFigure()

# # Current response figure (FIGURE 8)
# # currentFigure()



# # OLD MSO figure (FIGURE 3)
# # MSOfigure()

# # OLD Sound Dep Figure (FIGURE 4)
# # audDepfigure()



# # main(60,1)

# # nA = 3
# # nRep =5
# # ipi = 1
# # ITD = 0
# # audDep = 0
# # x,y=firingRate(61.5,64,nA,ipi,ITD,nRep,audDep)
# # print(x)
# # print(y)
# # ITDcurve(51,8,0)
# # ITDcurve(51,8,1)
# # ITDcurve(49.2,8,1)
# # ITDcurve(55,4,0)
# # ITDcurve(55,4,1)
# # ITDcurve(60,2,0)
# # ITDcurve(60,2,1)
# # ITDcurve(60,1,0)
# # ITDcurve(60,1,1)


# # ITDcurve(51.5,4,0)
# # ITDcurve(51.5,4,1)

# # ITDcurve(55,2,0)
# # ITDcurve(55,2,1)

# # ITDcurve(51,10,0)
# # ITDcurve(51,10,1)

# # ITDcurve(52,5,0)

# # ITDcurve(55,4,0)
# # ITDcurve(55,4,1)

# # ITDcurve(60,2,0)
# # ITDcurve(65,1,0)

# # currentInjection(1400,0)
# # currentInjection(550,1)


# # testMembraneSpeed()

# # main(0,10,0)
# # main(50,1,1)

# #main(A,ipi,fMod,ITD,audDep):
# # main(51,10,0,1)

# # main(90,.33,200,0,1)
# # main(90,.33,200,1,1)

# plt.show()







# # create firing rate data
# # runFiringRate(ipi=10,ITD=0, fMod=0, audDep=0, Amin=40, Amax=55, nA=16, nRep=100, runAN=1)
# # runFiringRate(ipi=10,ITD=0, fMod=0, audDep=1, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=0, fMod=0, audDep=2, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=0, fMod=0, audDep=3, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=0, fMod=0, audDep=4, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=0, fMod=0, audDep=5, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)

# # runFiringRate(ipi=10,ITD=1.0, fMod=0, audDep=0, Amin=40, Amax=55, nA=16, nRep=100, runAN=1)
# # runFiringRate(ipi=10,ITD=1.0, fMod=0, audDep=1, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=1.0, fMod=0, audDep=2, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=1.0, fMod=0, audDep=3, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=1.0, fMod=0, audDep=4, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)
# # runFiringRate(ipi=10,ITD=1.0, fMod=0, audDep=5, Amin=40, Amax=55, nA=16, nRep=100, runAN=0)

# ## runFiringRate(ipi=5,ITD=0, fMod=0, audDep=0, Amin=40, Amax=60, nA=21, nRep=100, runAN=1)
# ## runFiringRate(ipi=5,ITD=0, fMod=0, audDep=1, Amin=40, Amax=60, nA=21, nRep=100, runAN=0)

# ## runFiringRate(ipi=5,ITD=1.0, fMod=0, audDep=0, Amin=40, Amax=60, nA=21, nRep=100, runAN=1)
# ## runFiringRate(ipi=5,ITD=1.0, fMod=0, audDep=1, Amin=40, Amax=60, nA=21, nRep=100, runAN=0)

# ## runFiringRate(ipi=4, ITD=0, fMod=0, audDep=0, Amin=46, Amax=62, nA=17, nRep=100, runAN=1)
# ## runFiringRate(ipi=4, ITD=0, fMod=0, audDep=1, Amin=46, Amax=62, nA=17, nRep=100, runAN=0)

# ## runFiringRate(ipi=4, ITD=1, fMod=0, audDep=0, Amin=46, Amax=62, nA=17, nRep=100, runAN=1)
# ## runFiringRate(ipi=4, ITD=1, fMod=0, audDep=1, Amin=46, Amax=62, nA=17, nRep=100, runAN=0)

# # runFiringRate(ipi=4, ITD=0, fMod=0, audDep=0, Amin=40, Amax=74, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=4, ITD=0, fMod=0, audDep=1, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=0, fMod=0, audDep=2, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=0, fMod=0, audDep=3, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=0, fMod=0, audDep=4, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=0, fMod=0, audDep=5, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=4, ITD=1.0, fMod=0, audDep=0, Amin=40, Amax=74, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=4, ITD=1.0, fMod=0, audDep=1, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=1.0, fMod=0, audDep=2, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=1.0, fMod=0, audDep=3, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=1.0, fMod=0, audDep=4, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=4, ITD=1.0, fMod=0, audDep=5, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=2, ITD=0, fMod=0, audDep=0, Amin=40, Amax=74, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=2, ITD=0, fMod=0, audDep=1, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=0, fMod=0, audDep=2, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=0, fMod=0, audDep=3, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=0, fMod=0, audDep=4, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=0, fMod=0, audDep=5, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=2, ITD=1.0, fMod=0, audDep=0, Amin=40, Amax=74, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=2, ITD=1.0, fMod=0, audDep=1, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=1.0, fMod=0, audDep=2, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=1.0, fMod=0, audDep=3, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=1.0, fMod=0, audDep=4, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=2, ITD=1.0, fMod=0, audDep=5, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=1.33, ITD=0, fMod=0, audDep=0, Amin=40, Amax=91, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=1.33, ITD=0, fMod=0, audDep=1, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0, fMod=0, audDep=2, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0, fMod=0, audDep=3, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0, fMod=0, audDep=4, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0, fMod=0, audDep=5, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)

# ## runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=0, Amin=40, Amax=74, nA=18, nRep=100, runAN=1)
# ## runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=1, Amin=40, Amax=74, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=0, Amin=40, Amax=91, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=1, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=2, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=3, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=4, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1.33, ITD=0.66, fMod=0, audDep=5, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=1, ITD=0, fMod=0, audDep=0, Amin=40, Amax=91, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=1, ITD=0, fMod=0, audDep=1, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0, fMod=0, audDep=2, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0, fMod=0, audDep=3, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0, fMod=0, audDep=4, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0, fMod=0, audDep=5, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=1, ITD=0.5, fMod=0, audDep=0, Amin=40, Amax=91, nA=18, nRep=100, runAN=1)
# # runFiringRate(ipi=1, ITD=0.5, fMod=0, audDep=1, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0.5, fMod=0, audDep=2, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0.5, fMod=0, audDep=3, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0.5, fMod=0, audDep=4, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)
# # runFiringRate(ipi=1, ITD=0.5, fMod=0, audDep=5, Amin=40, Amax=91, nA=18, nRep=100, runAN=0)


# # TUNING CURVES
# ## runTuningCurve(ipi=10,A=50,fMod=0,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# ## runTuningCurve(ipi=10,A=50,fMod=0,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=10,A=51,fMod=0,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=10,A=51,fMod=0,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=10,A=51,fMod=0,audDep=2,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=10,A=51,fMod=0,audDep=3,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=10,A=51,fMod=0,audDep=4,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=10,A=51,fMod=0,audDep=5,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# ## runTuningCurve(ipi=10,A=52,fMod=0,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# ## runTuningCurve(ipi=10,A=52,fMod=0,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# ## runTuningCurve(ipi=4,A=54,fMod=0,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# ## runTuningCurve(ipi=4,A=54,fMod=0,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=4,A=53,fMod=0,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=4,A=53,fMod=0,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=4,A=53,fMod=0,audDep=2,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=4,A=53,fMod=0,audDep=3,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=4,A=53,fMod=0,audDep=4,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=4,A=53,fMod=0,audDep=5,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# ## runTuningCurve(ipi=4,A=52,fMod=0,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# ## runTuningCurve(ipi=4,A=52,fMod=0,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=2,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=3,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=4,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=5,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=1.33,A=70,fMod=0,audDep=0,ITDmin=0,ITDmax=0.66,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=1.33,A=70,fMod=0,audDep=1,ITDmin=0,ITDmax=0.66,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1.33,A=70,fMod=0,audDep=2,ITDmin=0,ITDmax=0.66,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1.33,A=70,fMod=0,audDep=3,ITDmin=0,ITDmax=0.66,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1.33,A=70,fMod=0,audDep=4,ITDmin=0,ITDmax=0.66,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1.33,A=70,fMod=0,audDep=5,ITDmin=0,ITDmax=0.66,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=1,A=80,fMod=0,audDep=0,ITDmin=0,ITDmax=0.5,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=1,A=80,fMod=0,audDep=1,ITDmin=0,ITDmax=0.5,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1,A=80,fMod=0,audDep=2,ITDmin=0,ITDmax=0.5,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1,A=80,fMod=0,audDep=3,ITDmin=0,ITDmax=0.5,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1,A=80,fMod=0,audDep=4,ITDmin=0,ITDmax=0.5,nITD=11,nRep=100,runAN=0)
# # runTuningCurve(ipi=1,A=80,fMod=0,audDep=5,ITDmin=0,ITDmax=0.5,nITD=11,nRep=100,runAN=0)




# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=2,ITDmin=0,ITDmax=1.,nITD=11,nRep=3,runAN=1)
# # runTuningCurve(ipi=2,A=60,fMod=0,audDep=3,ITDmin=0,ITDmax=1.,nITD=11,nRep=3,runAN=0)




# # FOR ENVELOPE - remember to set Abottom differently for each pulse rate
# # runFiringRate(ipi=0.33, ITD=0, fMod=0, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=0, fMod=0, audDep=1, Amin/=70, Amax=94, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=0.33, ITD=0.16, fMod=0, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=0.16, fMod=0, audDep=1, Amin=70, Amax=94, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=0.33, ITD=0, fMod=100, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=0, fMod=100, audDep=1, Amin=70, Amax=94, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=0.33, ITD=1, fMod=100, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=1, fMod=100, audDep=1, Amin=70, Amax=94, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=0.33, ITD=0, fMod=200, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=0, fMod=200, audDep=1, Amin=70, Amax=94, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=0.33, ITD=1, fMod=200, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=1, fMod=200, audDep=1, Amin=70, Amax=94, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=0.33, ITD=0, fMod=300, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=0, fMod=300, audDep=1, Amin=70, Amax=94, nA=18, nRep=100, runAN=0)

# # runFiringRate(ipi=0.33, ITD=1, fMod=300, audDep=0, Amin=70, Amax=94, nA=18, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=1, fMod=300, audDep=1, Amin=70, Amax=94, nA=18, nRep=100, runAN=0)






# # runFiringRate(ipi=0.33, ITD=0, fMod=0, audDep=0, Amin=70, Amax=94, nA=18, nRep=3, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=0, fMod=0, audDep=1, Amin=70, Amax=94, nA=18, nRep=3, runAN=0)

# # runFiringRate(ipi=0.33, ITD=1, fMod=0, audDep=0, Amin=70, Amax=94, nA=18, nRep=3, runAN=1) # for summary fig
# # runFiringRate(ipi=0.33, ITD=1, fMod=0, audDep=1, Amin=70, Amax=94, nA=18, nRep=3, runAN=0)










# # runFiringRate(ipi=1, ITD=0, fMod=100, audDep=0, Amin=50, Amax=75, nA=16, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=1, ITD=0, fMod=100, audDep=1, Amin=50, Amax=75, nA=16, nRep=100, runAN=0)

# # runFiringRate(ipi=1, ITD=1, fMod=100, audDep=0, Amin=50, Amax=75, nA=16, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=1, ITD=1, fMod=100, audDep=1, Amin=50, Amax=75, nA=16, nRep=100, runAN=0)

# # runFiringRate(ipi=0.4, ITD=0, fMod=250, audDep=0, Amin=50, Amax=75, nA=16, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.4, ITD=0, fMod=250, audDep=1, Amin=50, Amax=75, nA=16, nRep=100, runAN=0)

# # runFiringRate(ipi=0.4, ITD=1, fMod=250, audDep=0, Amin=50, Amax=75, nA=16, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.4, ITD=1, fMod=250, audDep=1, Amin=50, Amax=75, nA=16, nRep=100, runAN=0)

# # runFiringRate(ipi=0.4, ITD=0, fMod=250, audDep=0, Amin=50, Amax=90, nA=21, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.4, ITD=0, fMod=250, audDep=1, Amin=50, Amax=90, nA=21, nRep=100, runAN=0)

# # runFiringRate(ipi=0.4, ITD=1, fMod=250, audDep=0, Amin=50, Amax=90, nA=21, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.4, ITD=1, fMod=250, audDep=1, Amin=50, Amax=90, nA=21, nRep=100, runAN=0)





# # runTuningCurve(ipi=1,A=52,fMod=100,audDep=0,ITDmin=0,ITDmax=1,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=1,A=52,fMod=100,audDep=1,ITDmin=0,ITDmax=1,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=0.5,A=52,fMod=200,audDep=0,ITDmin=0,ITDmax=1,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=0.5,A=52,fMod=200,audDep=1,ITDmin=0,ITDmax=1,nITD=11,nRep=100,runAN=0)


# # runFiringRate(ipi=0.5, ITD=0, fMod=200, audDep=0, Amin=40, Amax=60, nA=21, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.5, ITD=0, fMod=200, audDep=1, Amin=40, Amax=60, nA=21, nRep=100, runAN=0)

# # runFiringRate(ipi=0.5, ITD=1, fMod=200, audDep=0, Amin=40, Amax=60, nA=21, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.5, ITD=1, fMod=200, audDep=1, Amin=40, Amax=60, nA=21, nRep=100, runAN=0)


# # runFiringRate(ipi=0.4, ITD=0, fMod=250, audDep=0, Amin=40, Amax=60, nA=21, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.4, ITD=0, fMod=250, audDep=1, Amin=40, Amax=60, nA=21, nRep=100, runAN=0)

# # runFiringRate(ipi=0.4, ITD=1, fMod=250, audDep=0, Amin=40, Amax=60, nA=21, nRep=100, runAN=1) # for summary fig
# # runFiringRate(ipi=0.4, ITD=1, fMod=250, audDep=1, Amin=40, Amax=60, nA=21, nRep=100, runAN=0)

# # runTuningCurve(ipi=1,A=51,fMod=100,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=1,A=51,fMod=100,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=.4,A=52,fMod=250,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=.4,A=52,fMod=250,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=.4,A=50,fMod=250,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=.4,A=50,fMod=250,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=.4,A=49,fMod=250,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=.4,A=49,fMod=250,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)

# # runTuningCurve(ipi=.4,A=48,fMod=250,audDep=0,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=1)
# # runTuningCurve(ipi=.4,A=48,fMod=250,audDep=1,ITDmin=0,ITDmax=1.,nITD=11,nRep=100,runAN=0)
