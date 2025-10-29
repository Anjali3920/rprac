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

n=3;
A=Table[{{5,1,2},{-3,9,4},{1,2,-7}}];
RHS=Table[{10,-14,-33}];
x=Table[{0,0,,0}];
max=10;   
#earlier=x jac0bi
For[k=1,k<=max,k++,
For[i=1,i<=n,i++,sum=0;                                                       # *earlier[[j]]];
For[j=1,j<=n,j++,If[i!=j,sum=sum+A[[i,j]]*x[[j]]]];
x[[i]]=N [(RHS[[i]]-sum)/(A[[i,i]])]];
    Print[x]];
  #Print[x];earlier=x]

#euler
f[x_, y_] := 2 x + y
x0 = 0;
y0 = 1;
h = 0.1;
xf = 1;
n = (xf - x0)/(h)//N
Do[xi = x0 + i h; yi+1 = yi + h(2xi + yi),{i, 0, n}]
TableForm[Table[{xi, yi, - 2 - 2 xi + 3 * Exp[xi]}, {i,0,n}],
 TableHeadings-> {None, {x, Approx y, Exact y}}]
Plot1 = ListPlot[Table[{xi, yi}, {i, 0, n}], Joined->True]
Plot2 = Plot[- 2 - 2 x + 3 * Exp[x],{x, 0, 1}, PlotStyle-> {Red}]
a = Show[Plot1, Plot2]

#trapezoidal
f[x_]:=1/(1+x^2);
a=0;
b=1;
c=N[Integrate[f[x],{x,0,1}]];
Print["the exact value of the integral is:",c]
Trapezoidal=N[(b-a)/(2)*(f[a]+f[b])];
Print[Trapezoidal]
Print[N[Abs[Trapezoidal-c]]]


f[x_]:=1/(1+x^2);
a=0;
b=1;
c=N[Integrate[f[x],{x,0,1}]];
Print["the exact value of the integral is:",c]
Simpson=N[(b-a)/(6)*(f[a]+4*f[(a+b)]/(2)+f[b])];
Print[Simpson]
Print[N[Abs[Simpson-c]]]


 ![Image](https://github.com/user-attachments/assets/43dd420d-d46a-4202-859c-942d63ac638b)





```
![Image](https://github.com/user-attachments/assets/262ac98c-724a-4e3f-a635-06fc175e6683)

![Image](https://github.com/user-attachments/assets/923209a5-6904-4eb6-9299-f22486629ad5)


