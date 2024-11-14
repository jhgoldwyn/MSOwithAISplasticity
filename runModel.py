# import needed functions
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from modelFunctions import *

# plotting options
matplotlib.rcParams.update({'font.size':14, 'font.family':'arial', 'figure.figsize':(12,8)})

# parameter values
A      = 55 # pulse amplitude [pA]
ipi    = 2  # interpulse interval [ms]
fMod   = 0  # modulation frequency [Hz] (0=unmodulated)
ITD    = .1  # time difference (ms) 
audDep = 0  # model type: 0=control, 1=auditory deprived

# run model
t,v1,v2,g = main(A,ipi,fMod,ITD,audDep)

# plot results
fig = plt.figure(figsize=(16,12))#,layout='constrained')
gs = GridSpec(5, 1, figure=fig)

ax = fig.add_subplot(gs[0:2])    
ax.plot(t,v2,lw=2)
ax.set_ylim([-80,40])
ax.set_xlabel('time (ms)')
ax.set_ylabel('$V_2$ voltage (mV)')

ax = fig.add_subplot(gs[2:4])    
ax.plot(t,v1,lw=2)
ax.set_ylim([-80,40])
ax.set_xlabel('time (ms)')
ax.set_ylabel('$V_1$ voltage (mV)')

ax = fig.add_subplot(gs[4])    
ax.plot(t,g,lw=2)
ax.set_ylim([-1,20])
ax.set_xlabel('time (ms)')
ax.set_ylabel('conductance (nS)')

plt.show()
