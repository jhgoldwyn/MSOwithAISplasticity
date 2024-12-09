# MSOwithAISplasticity

Python and C code for model of MSO response to CI stimulation with AIS plasticity in the two-compartment model MSO neuron

This code accompanies the manuscript:

Axon initial segment plasticity caused by auditory deprivation degrades time difference sensitivity in a model of neural responses to cochlear implants

by
Anna Jing (=), Sylvia Xi (=), Ivan Fransazov, Joshua H. Goldwyn

Swarthmore College

(=) These authors contributed equally

runModel.py can be executed from command line

python3 runModel.py

output is a figure showing synaptic and voltage responses to CI stimulus

parameter values can be changed within the runModel.py script

modelFunctions.py: functions and subroutines for setting up and running the AN model, creating synaptic inputs, and setting up the MSO model

twoCpt.c: C code for the MSO neuron model

makefile: make file to compile C code
