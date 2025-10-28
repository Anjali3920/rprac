# rprac
```
a=1;
b=2;
n=10;
f[x_]:=x^2-2;
Plot[f[x],{x,1,2}]
For[i=1,i<=n,i++,{c=(a+b)/2,If [f[a]*f[c]<0,b=c,a=c],Print[N[c]]}]


p0=1;
p=p0;
n=6;
f[x_]:=x^3+2*x^2-3*x-1;
Plot[f[x],{x,1,2}]
For[i=1,i<=n,i++,{p=p-f[p]/f'[p],Print[N[p]]}]


p0=0;
p1=1;
n=4;
f[x_]:=x^3-5*x+1;
Plot[f[x],{x,0,1}]
For[i=1,i<=n,i++,{p=p1-(p1-p0)/(f[p1]-f[p0])*f[p1],p0=p1,p1=p,Print[N[p]]}]



```
