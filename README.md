# rprac
```
a=1;
b=2;
n=10;
f[x_]:=x^2-2;
Plot[f[x],{x,1,2}]
For[i=1,i<=n,i++,{c=(a+b)/2,If [f[a]*f[c]<0,b=c,a=c],Print[N[c]]}]




a=1;
b=2;
n=6;
f[x_]:=x^3+2*x^2-3*x-1;
Plot[f[x],{x,1,2}]
For[i=1,i<=n,i++,{p=(a*f[b]-b*f[a])/(f[b]-f[a]),If[f[a]*f[p]>0,a=p,b=p],Print[N[p]]}]


'''
'''
#  single of cold pill
k1=1.386;
k2=0.1386;
tend=15;
eqnsDeg={x'[t]==-k1*x[t],y'[t]==k1*x[t]-k2*y[t],x[0]==1,y[0]==0};
solDeg=DSolve[eqnsDeg,{x[t],y[t]},t];
Plot[Evaluate[{x[t],y[t]}/.solDeg],{t,0,tend},
PlotStyle->{Blue,Red},PlotLegends->{"Decongestant","Antihistamine"}
AxesLabel->{"t","Concentration"},PlotRange->All]

2. course of cold pill
k1=1.386;
k2=0.1386;
tend=15;
l=100
eqnsDeg={x'[t]==l-k1*x[t],y'[t]==k1*x[t]-k2*y[t],x[0]==1,y[0]==0};
solDeg=DSolve[eqnsDeg,{x[t],y[t]},t];
Plot[Evaluate[{x[t],y[t]}/.solDeg],{t,0,tend},
PlotStyle->{Blue,Red},PlotLegends->{"Decongestant","Antihistamine"}
AxesLabel->{"t","Concentration"},PlotRange->All]

3.lake pollution
f=48000000;
v=28000000;
cin=30000000;
s1=DSolve[{c'[t]==(f/v)(cin-c[t]),c[0]==10^7},c[t],t]
s2=DSolve[{c'[t]==(f/v)(cin-c[t]),c[0]==2*10^7},c[t],t]
s3=DSolve[{c'[t]==(f/v)(cin-c[t]),c[0]==3*10^7},c[t],t]
s4=DSolve[{c'[t]==(f/v)(cin-c[t]),c[0]==4*10^7},c[t],t]
s5=DSolve[{c'[t]==(f/v)(cin-c[t]),c[0]==5*10^7},c[t],t]
Plot[Evaluate[c[t] /.{s1,s2,s3,s4,s5}],{t,0,10},PlotRange->Full,PlotLegends->Automatic]

4.seasonal flow model
f=1000000(1+6*Sin[2*Pi*t])
v=28000000
cin=1000000(10+10*Cos[2*Pi*t])
s6=NDSolve[{c'[t]==(f/v)(cin-c[t]),c[0]==10^7},c[t],{t,0,10}]
Plot[Evaluate[c[t]/.s6],{t,0,10},PlotRange->Full,PlotLegends->Automatic]

5.harvesting
r=Input["enter the growth constant"]
h=Input["enter the harvesting constant"]
s1=DSolve[{x'[t]==r*x[t]-h,x[0]==200},x[t],t]
s2=DSolve[{x'[t]==r*x[t]-h,x[0]==400},x[t],t]
s3=DSolve[{x'[t]==r*x[t]-h,x[0]==600},x[t],t]
s4=DSolve[{x'[t]==r*x[t]-h,x[0]==800},x[t],t]
s5=DSolve[{x'[t]==r*x[t]-h,x[0]==1000},x[t],t]
Plot[Evaluate[x[t]/.{s1,s2,s3,s4,s5}],
{t,0,1},PlotRange->Full,PlotLegends->Automatic]

6.prey predator model
\[Beta]=1
\[Alpha]=0.5
c1=0.01
c2=0.005
c=NDSolve[{x'[t]==\[Beta]*x[t]-c1*x[t]*y[t],y'[t]==c2*x[t]*y[t]-\[Alpha]*y[t],x[0]==200,y[0]==80},{x[t],y[t]},{t,0,20}]
Plot[Evaluate[{x[t],y[t]}/.c],{t,0,20},PlotRange->Full,
PlotLegends->{"x[t]-Prey","y[t]-Preedator"},PlotStyle->Thick]
7.epidemic model
"s[t]=[" enter the susceptible population "]
i[t]=["enter the infective population"]
r[t]=["enter the recovered population"]"
\[Beta]=Input[]
\[Gamma]=Input[]
sol=NDSolve[{s'[t]==-\[Beta]*s[t]*i[t],i'[t]==\[Beta]*s[t]*i[t]-\[Gamma]*i[t],r'[t]==\[Gamma]*i[t],s[0]==762,i[0]==1,r[0]==0,{s[t],i[t],r[t]},{t,0,20}]
Plot[Evaluate[{s[t],i[t],r[t]}/.sol],{t,0,30},
PlotRange->Full,PlotLegends->Automatic]
8. logistic growth
r=Input["enter the growth constant"]
k=Input["enter the capacity constant"]
s1=DSolve[{x'[t]==r*x[t]*(1-x[t]/k),x[0]==200},x[t],t]
s2=DSolve[{x'[t]==r*x[t]*(1-x[t]/k),x[0]==400},x[t],t]
s3=DSolve[{x'[t]==r*x[t]*(1-x[t]/k),x[0]==600},x[t],t]
s4=DSolve[{x'[t]==r*x[t]*(1-x[t]/k),x[0]==800},x[t],t]
s5=DSolve[{x'[t]==r*x[t]*(1-x[t]/k),x[0]==1000},x[t],t]
Plot[Evaluate[x[t]/.{s1,s2,s3,s4,s5}],
{t,0,1},PlotRange->Full,PlotLegends->Automatic]





























































































