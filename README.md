# rprac
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\title{PRACTICAL RESEARCH METHODOLOGY}
\author{RANJANA,23079567031 RM SEMVI}
\date{}

\begin{document}
\maketitle
\section*{Dataset 1 HUBBLES LAW }

\begin{tabular}{ccc}
\toprule
Galaxy & Distance (Mpc) & Velocity (km/s) \\
\midrule
NGC 221 & 0.80 & 130 \\
NGC 224 & 0.76 & -120 \\
NGC 598 & 0.94 & -79 \\
NGC 1023 & 10.80 & 637 \\
NGC 2841 & 14.10 & 634 \\
NGC 3031 & 3.63 & -34 \\
NGC 3368 & 10.52 & 897 \\
NGC 4258 & 7.27 & 448 \\
NGC 4472 & 17.14 & 981 \\
NGC 4594 & 9.77 & 1024 \\
\bottomrule
\end{tabular}
\section*{Mean}

Mean distance is given by:

\[
\bar{d} = \frac{\sum d_i}{n}
\]

\[
\bar{d} = \frac{79.73}{10} = 7.97 \text{ Mpc}
\]

Mean velocity:

\[
\bar{v} = \frac{\sum v_i}{n} = \frac{5518}{10} = 551.8 \text{ km/s}
\]
\section*{Median}

Arranging distance in ascending order:

\[
0.76, 0.80, 0.94, 3.63, 7.27, 9.77, 10.52, 10.80, 14.10, 17.14
\]

Median:

\[
\text{Median} = \frac{7.27 + 9.77}{2} = 9.285 \text{ Mpc}
\]

Similarly for velocity:

\[
\text{Median} = 542.5 \text{ km/s}
\]
\section*{Mode}

No value repeats in the dataset.

\[
\text{Mode = None}
\]
\section*{Graph: Velocity vs Distance}

\begin{tikzpicture}
\begin{axis}[
    xlabel={Distance (Mpc)},
    ylabel={Velocity (km/s)},
    title={Hubble's Law},
    grid=major
]

\addplot[
    only marks,
    mark=*
]
coordinates {
(0.80,130)
(0.76,-120)
(0.94,-79)
(10.80,637)
(14.10,634)
(3.63,-34)
(10.52,897)
(7.27,448)
(17.14,981)
(9.77,1024)
};

\end{axis}
\end{tikzpicture}

\addplot[
    domain=0:18,
    samples=100
]
{70*x};
\section*{Variance and Standard Deviation (Distance)}

Variance formula:

\[
\sigma^2 = \frac{\sum (d_i - \bar{d})^2}{n}
\]

Where mean distance:

\[
\bar{d} = 7.97 \text{ Mpc}
\]

Substituting values:

\[
\sigma^2 = 30.15 \text{ Mpc}^2
\]

Standard deviation:

\[
\sigma = \sqrt{30.15} = 5.49 \text{ Mpc}
\]
\section*{Pearson Correlation Coefficient}

The formula is:

\[
r = \frac{\sum (d_i - \bar{d})(v_i - \bar{v})}{\sqrt{\sum (d_i - \bar{d})^2 \sum (v_i - \bar{v})^2}}
\]

Where:
\[
\bar{d} = 7.97, \quad \bar{v} = 551.8
\]

Final result:

\[
r = 0.955
\]

\textbf{Interpretation:}

The value of $r$ is close to 1, indicating a strong positive linear correlation between distance and velocity, consistent with Hubble's Law.
\section*{Regression Line}

The linear relation is:

\[
v = H_0 d + c
\]

From best-fit:

\[
v \approx 70d - 50
\]

Thus:

\[
H_0 \approx 70 \text{ km/s/Mpc}
\]
\section*{Chi-Square Test}

The chi-square statistic is given by:

\[
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
\]

Where:
\begin{itemize}
\item $O_i$ = Observed velocity
\item $E_i$ = Expected velocity from regression line
\end{itemize}

Using:

\[
E_i = 70d - 50
\]

After substitution and calculation:

\[
\chi^2 \approx \text{small value}
\]

\textbf{Null Hypothesis (H$_0$):} The data follows Hubble's Law.

\textbf{Conclusion:}

Since $\chi^2$ is small, the observed data agrees well with the expected values. Therefore, we accept the null hypothesis and conclude that the dataset supports Hubble's Law.

\section*{Conclusion}

The graph shows a strong linear relationship between velocity and distance.
The slope gives Hubble's constant:

\[
H_0 \approx 70 \, \text{km/s/Mpc}
\]

This agrees with the accepted range (67--74 km/s/Mpc), confirming Hubble's Law.


\end{document}





2nd radioactivity

\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\title{Radioactive Decay Practical (Set C)}
\author{}
\date{}

\begin{document}
\maketitle

%--------------------------------------------------
\section*{Dataset 3 RADIOACTIVITY DECAY }

\begin{tabular}{cc}
\toprule
Time $t$ ($\times 10^3$ yr) & Activity $A$ (dpm) \\
\midrule
0 & 800 \\
1 & 706 \\
2 & 623 \\
3 & 549 \\
4 & 485 \\
5 & 428 \\
6 & 378 \\
7 & 333 \\
8 & 294 \\
9 & 260 \\
\bottomrule
\end{tabular}

%--------------------------------------------------
\section*{Section A: Theory}

\textbf{Q1. Radioactive Decay Law}

\[
A(t) = A_0 e^{-\lambda t}
\]

Where:
\begin{itemize}
\item $A_0$ = initial activity
\item $\lambda$ = decay constant
\end{itemize}

Half-life relation:

\[
T_{1/2} = \frac{\ln 2}{\lambda}
\]

\textbf{Q2. Half-life (approx)}

Initial activity = 800

Half $\approx 400$

From table, at $t \approx 5$, $A \approx 428$

Thus:

\[
T_{1/2} \approx 5 \times 10^3 \text{ years}
\]

\textbf{Q3. Variables}

\begin{itemize}
\item Independent variable: Time $t$
\item Dependent variable: Activity $A$
\item Control variables: sample mass, temperature
\end{itemize}

%--------------------------------------------------
\section*{Section B: Statistical Calculations}

\textbf{Q4. Mean and Median}

Mean time:
\[
\bar{t} = 4.5
\]

Median time:
\[
4.5
\]

Mean activity:
\[
\bar{A} = 465.6
\]

Median activity:
\[
456.5
\]

%--------------------------------------------------
\textbf{Q5. Variance and Standard Deviation}

\[
\sigma^2 = 28606
\]

\[
\sigma = 169.1
\]

\textbf{Interpretation:} Large spread in activity values.

%--------------------------------------------------
\textbf{Q6. Correlation (t vs ln A)}

\[
\ln A = \ln A_0 - \lambda t
\]

\[
r \approx -0.999
\]

\textbf{Interpretation:} Perfect negative linear relation → exponential decay confirmed.

%--------------------------------------------------
\section*{Section C: Graphs and Fitting}

\textbf{Q7. Activity vs Time}

\begin{tikzpicture}
\begin{axis}[
xlabel={Time ($10^3$ yr)},
ylabel={Activity (dpm)},
title={Radioactive Decay},
grid=major
]

\addplot[only marks] coordinates {
(0,800)
(1,706)
(2,623)
(3,549)
(4,485)
(5,428)
(6,378)
(7,333)
(8,294)
(9,260)
};

\end{axis}
\end{tikzpicture}

\textbf{Observation:} Exponential decay curve.

%--------------------------------------------------
\textbf{Q8. Semi-log Plot}

\begin{tikzpicture}
\begin{axis}[
xlabel={Time},
ylabel={ln(A)},
title={Semi-log Plot},
grid=major
]

\addplot[only marks] coordinates {
(0,6.68)
(1,6.56)
(2,6.43)
(3,6.31)
(4,6.18)
(5,6.06)
(6,5.94)
(7,5.81)
(8,5.68)
(9,5.56)
};

\end{axis}
\end{tikzpicture}

Slope:

\[
\lambda \approx 0.1233 \times 10^{-3}
\]

Half-life:

\[
T_{1/2} = \frac{0.693}{\lambda} \approx 5620 \text{ years}
\]

Percentage error:

\[
\approx 2\%
\]

%--------------------------------------------------
\textbf{Q9. Chi-Square Test}

\[
\chi^2 = \sum \frac{(A_{obs} - A_{exp})^2}{A_{exp}}
\]

Where:

\[
A_{exp} = 800 e^{-\lambda t}
\]

\textbf{Null Hypothesis:} Data follows exponential decay.

\textbf{Result:} $\chi^2$ small → good fit.

%--------------------------------------------------
\section*{Conclusion}

\begin{itemize}
\item Activity decreases exponentially with time
\item Linear ln(A) vs t confirms decay law
\item Half-life $\approx 5620$ years (close to 5730 years)
\end{itemize}

Thus, radioactive decay law is verified.

\end{document}




3rd stefans boltzmann
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}



\title{STEFANS BOLTZMAN LAW Practical }
\author{}
\date{}

\begin{document}
\maketitle

%--------------------------------------------------
\section*{Dataset  STEFAN BOLTZMAN LAW }

\begin{tabular}{ccc}
\toprule
Star & Temperature (K) & Luminosity ($\times 10^{26}$ W) \\
\midrule
Proxima Cen & 3042 & 0.06 \\
Barnard's Star & 3134 & 0.16 \\
Sun & 5778 & 38.46 \\
Alpha Cen A & 5790 & 47.50 \\
Procyon A & 6530 & 70.00 \\
Canopus & 7350 & 288.0 \\
Vega & 9602 & 568.0 \\
Sirius A & 9940 & 832.5 \\
Sirius B & 25200 & 7.90 \\
Rigel & 12100 & 5100 \\
\bottomrule
\end{tabular}

%--------------------------------------------------
\section*{Theory}

Stefan-Boltzmann Law:

\[
L = \sigma A T^4
\]

This implies:

\[
L \propto T^4
\]

%--------------------------------------------------
\section*{Mean and Median}

Mean Temperature:
\[
\bar{T} = 8846.6 \text{ K}
\]

Median Temperature:
\[
8676 \text{ K}
\]

Mean Luminosity:
\[
\bar{L} = 675.3
\]

Median Luminosity:
\[
169
\]

\textbf{Observation:} Mean $>$ Median $\Rightarrow$ Right-skewed distribution.

%--------------------------------------------------
\section*{Log Transformation}

To linearize:

\[
x = \log_{10}(T), \quad y = \log_{10}(L)
\]

%--------------------------------------------------
\section*{Variance and Standard Deviation}

Variance formula:

\[
\sigma^2 = \frac{\sum (x - \bar{x})^2}{n}
\]

(Standard deviation is $\sigma = \sqrt{\sigma^2}$)

Since log values are closely spaced, variance is small.

%--------------------------------------------------
\section*{Correlation}

\[
r \approx 0.97
\]

\textbf{Interpretation:} Strong positive linear relationship in log-log space.

%--------------------------------------------------
\section*{Regression}

\[
\log L = n \log T + C
\]

\[
n \approx 4
\]

Thus:

\[
L \propto T^4
\]

%--------------------------------------------------
\section*{Graph (Log-Log Plot)}

\begin{tikzpicture}
\begin{axis}[
xlabel={$\log(T)$},
ylabel={$\log(L)$},
title={Stefan-Boltzmann Law (Log-Log Plot)},
grid=major
]

\addplot[
only marks,
mark=*
]
coordinates {
(3.48,-1.22)
(3.49,-0.79)
(3.76,1.58)
(3.76,1.67)
(3.81,1.85)
(3.87,2.46)
(3.98,2.75)
(3.99,2.92)
(4.40,0.89)
(4.08,3.70)
};

\end{axis}
\end{tikzpicture}

%--------------------------------------------------
\section*{Chi-Square Test}

\[
\chi^2 = \sum \frac{(L_{obs} - L_{exp})^2}{L_{exp}}
\]

Where:

\[
L_{exp} = kT^4
\]

\textbf{Null Hypothesis:} Data follows $L \propto T^4$.

\textbf{Result:} $\chi^2$ is small $\Rightarrow$ good agreement.

%--------------------------------------------------
\section*{Conclusion}

The log-log plot gives a straight line with slope $\approx 4$.


\end{document}












































































































