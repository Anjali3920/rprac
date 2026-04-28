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



























































































