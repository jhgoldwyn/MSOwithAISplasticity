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
    r = np.random.rand(nPulse) # draw random numbers for stochastic spiking
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