#include <stdio.h>
#include <math.h> 
#include <stdlib.h>
#include <string.h>

FILE *MSOfile, *ANfile, *paramFile;

// what data to save
int writeData; /* command line input */

// simulation run time
double tStop; /* command line input */

/* FIXED PARAMETERS */
/* reversal potentials*/
double ena=55.,ek=-106.,eh=-47.,esyn=0.;
double spikeThreshold = -20.; 

/* general variables */
double tEnd;
double dt;//=.0001;
double phi;
double cap[2];
double gax;
double iax;
double v[2],w[2],z[2],rf[2],rs[2];
double m,h,n,p,u;
double elk[2];
double glk[2],gklt[2],gh[2];
double gna,gkht,gM;
double ilk[2],iklt[2],ih[2];
double ina,ikht,iM;
double winf[2],zinf[2],rinf[2];
double minf,hinf,ninf,pinf,uinf;
double tauw[2],tauz[2],taurs[2],taurf[2];
double taum,tauh,taun,taup,tauu;
double tha,qa,Ra,Rb,tadj,aM,bM ; // M current
double isyn;
float gsyn;
double vOld;
double ITD,istimStrength,synStrength;
int iterNum,istimInt,itdInt,nNeuron,audDep;
char ANfilename[100],MSOfilename[100],paramFileName[100];
double ipi;

int main(int argc, char *argv[])
{

  double t=0;
  double dv[2];
  double dw[2],dz[2],drf[2],drs[2];
  double dm,dh,dn,dp,du;
  double kr,phi;
  char synstr[20];
  int counter;
  int i;
  char paramVal[60];

  /* Read in Parameters from text file */ 
  sprintf(paramFileName, "runIt_%s.txt",argv[1]); 
  paramFile = fopen(paramFileName,"r");
  counter = 0;
  while(fgets(paramVal, 10, paramFile)) {
    counter+=1;
    if (counter==1){
      writeData = atoi(paramVal);
    }
    if (counter==2){
      tEnd = atof(paramVal);
    }
    if (counter==3){
      dt = atof(paramVal);
    }
    if (counter==4){
      cap[0] = atof(paramVal);
    }
    if (counter==5){
      cap[1] = atof(paramVal);
    }
    if (counter==6){
      elk[0] = atof(paramVal);
    }
    if (counter==7){
      elk[1] = atof(paramVal);
    }
    if (counter==8){
      glk[0] = atof(paramVal);
    }
    if (counter==9){
      glk[1] = atof(paramVal);
    }
    if (counter==10){
      gklt[0] = atof(paramVal);
    }
    if (counter==11){
      gh[0] = atof(paramVal);
    }
    if (counter==12){
      gklt[1] = atof(paramVal);
    }
    if (counter==13){
      gh[1] = atof(paramVal);
    }
    if (counter==14){
      gM = atof(paramVal);
    }
    if (counter==15){
      gna = atof(paramVal);
    }
    if (counter==16){
      gkht = atof(paramVal);
    }
    if (counter==17){
      gax = atof(paramVal);
    }
    if (counter==18){
      ipi = atof(paramVal);
    }
    if (counter==19){
      istimStrength = atof(paramVal);
    }
    if (counter==20){
      ITD = atof(paramVal);
    }
    if (counter==21){
      synStrength = atof(paramVal);
    }
    if (counter==22){
      nNeuron = atoi(paramVal);
    }
    if (counter==23){
      audDep = atoi(paramVal);
    }
    if (counter==24){
      iterNum = atoi(paramVal);
    }
  } // end read in paramters
  fclose(paramFile);

  /* open file for reading synaptic data */
  ANfile = fopen("ANfile.txt","r"); 
  
  /* setup file for saving data */
  if (writeData==0)//spikecount
    {
      sprintf(MSOfilename, "MSOSpike.txt");
      MSOfile=fopen(MSOfilename, "w");
    }
  else if (writeData==1)//spikecount
    {
      sprintf(MSOfilename, "MSOVoltage.txt");
      MSOfile=fopen(MSOfilename, "w");
    }
  /* initial values */
  for (i = 0; i < 2; i++) {
    v[i] = -58.;
    w[i] = 0.4850;
    z[i] = 0.6457;
    rf[i] = 0.4219;
    rs[i] = 0.4219;
  }

  m = 0.0545;
  h = 0.2368;
  n = 0.0136;
  p = 0.0029;
  u = 0.0427;
  kr = 0.65;
  phi = 0.85;


  /* Euler Loop */
  while(t<(tEnd-dt/2.))
  {

    /* synaptic input */
    fgets(synstr,20, ANfile);

    gsyn = atof(synstr);
    if (synStrength>0){    
      isyn = synStrength*gsyn*(v[0]-esyn);}
    else{
      isyn = -gsyn;}
    

    for (i = 0; i < 2; i++) {
    
    // KLT
    winf[i] = 1. / (1. + exp(-(v[i]+57.3)/11.7) ); 
    zinf[i] = (1.-.22) / (1.+exp((v[i]+57.)/5.44)) + .22;
    tauw[i] = .46*(100. / (6.*exp((v[i]+75.)/12.15) + 24.*exp(-(v[i]+75.)/25.) + .55));
    tauz[i] = .24*(1000. / (exp((v[i]+60.)/20.) + exp(-(v[i]+60.)/8.)) + 50.);
    
    // H 
    rinf[i] = 1. / (1 + exp((v[i]+60.3)/7.3));
    taurf[i] = 10000. / ( (-7.4*(v[i]+60))/(exp(-(v[i]+60)/0.8)-1) + 65*exp(-(v[i]+56)/23) );
    taurs[i] = 1000000. / ( (-56*(v[i]+59))/(exp(-(v[i]+59)/0.8)-1) + 0.24*exp(-(v[i]-68)/16) );

    }
    
    // Na
    minf = 1.  / (1.+exp(-(v[1]+38.)/7.));
    hinf = 1. / (1.+exp((v[1]+65.)/6.));
    taum = .24*((10 / (5*exp((v[1]+60) / 18) + 36*exp(-(v[1]+60) / 25))) + 0.04);
    tauh = .24* (100. / (7.*exp((v[1]+60)/11.) + 10.*exp(-(v[1]+60.)/25.)) + 0.6);

    // KHT 
    ninf = 1. / sqrt(1 + exp(-(v[1] + 15) / 5));
    pinf = 1. / (1 + exp(-(v[1] + 23) / 6));
    taun =  0.24* ((100 / (11*exp((v[1]+60) / 24) + 21*exp(-(v[1]+60) / 23))) + 0.7);
    taup = 0.24* ((100 / (4*exp((v[1]+60) / 32) + 5*exp(-(v[1]+60) / 22))) + 5);

    // M 
    tha = -30;
    Ra = 0.001;
    Rb = 0.001;
    qa = 9.;
    tadj = pow(( (35-23) / 10), 2.3); // temperature adjustm to 35 with q10=2.3
    aM = Ra * (v[1] - tha) / (1 - exp(-(v[1] - tha)/qa));
    bM = -Rb * (v[1] - tha) / (1 - exp((v[1] - tha)/qa));
    tauu = (1/tadj) * (1 /( aM+bM ));
    uinf = aM /( aM+bM );


    // currents
    for (i = 0; i < 2; i++) {
      ilk[i] = glk[i]*(v[i]-elk[i]);
      iklt[i]  = gklt[i]*(w[i]*w[i]*w[i]*w[i])*z[i]*(v[i]-ek);
      ih[i] = gh[i]*(kr*rf[i] + (1-kr)*rs[i]) * (v[i]-eh);
    }

    ikht = gkht*(phi*(n*n) + (1-phi)*p ) * (v[1]-ek);
    iM   = gM*u*(v[1]-ek);
    ina  = gna*(m*m*m)*h*(v[1]-ena);
    iax = gax*(v[0]-v[1]);

    /* ODEs */
    dv[0] = -(ilk[0] + iklt[0] + ih[0] + iax + isyn)/cap[0];
    dv[1] = -(ilk[1] + iklt[1] + ih[1] + ina + ikht + iM - iax)/cap[1];

    for (i = 0; i < 2; i++) {
      dw[i] = (winf[i] - w[i])  / tauw[i];
      dz[i] = (zinf[i] - z[i])  / tauz[i];
      drf[i] = (rinf[i] - rf[i])  / taurf[i];
      drs[i] = (rinf[i] - rs[i])  / taurs[i];
    }

    dm = (minf - m)  / taum;
    dh = (hinf - h)  / tauh;
    dn = (ninf - n)  / taun;
    dp = (pinf - p)  / taup;
    du = (uinf - u)  / tauu;
    
    // /* update variables */
    vOld = v[1];
    t += dt;
    for (i = 0; i < 2; i++) {
      v[i] = v[i] + dt*dv[i];
      w[i] = w[i] + dt*dw[i];
      z[i] = z[i] + dt*dz[i];
      rf[i] = rf[i] + dt*drf[i];
      rs[i] = rs[i] + dt*drs[i];
    }
    m = m + dt*dm;
    h = h + dt*dh;
    n = n + dt*dn;
    p = p + dt*dp;
    u = u + dt*du;

  
    /* write data */
    if (writeData==0)//spikecount
    {
      if ( (vOld<=spikeThreshold) && (v[1]>spikeThreshold))
      {
        fprintf(MSOfile,"%.5f\n",t);
      }
    }
    else if (writeData==1) // voltage
    {
      fprintf(MSOfile,"%.5f %.5f %.5f \n",t,v[0],v[1]);
    }
    else if (writeData==2)
    {
      fprintf(MSOfile,"%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",t,v[0],v[1],w[0],z[0],rf[0],rs[0],w[1],z[1],rf[1],rs[1],m,h,n,p,u);
    }

  }  /* end Euler loop */

  fclose(MSOfile);
  fclose(ANfile);

} /* end main */
