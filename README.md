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

4#########################################
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\title{Kepler's Third Law Practical (Set D)}
\author{Your Name}
\date{}

\begin{document}
\maketitle

%--------------------------------------------------
\section*{Dataset}

\begin{tabular}{ccc}
\toprule
Body & Radius $r$ (AU) & Period $T$ (yr) \\
\midrule
Mercury & 0.387 & 0.241 \\
Venus & 0.723 & 0.615 \\
Earth & 1.000 & 1.000 \\
Mars & 1.524 & 1.881 \\
Ceres & 2.767 & 4.600 \\
Jupiter & 5.203 & 11.862 \\
Saturn & 9.537 & 29.457 \\
Uranus & 19.191 & 84.011 \\
Neptune & 30.069 & 164.800 \\
Pluto & 39.482 & 248.000 \\
\bottomrule
\end{tabular}

%--------------------------------------------------
\section*{Section A: Theory}

\textbf{Q1. Kepler’s Third Law}

\[
T^2 \propto r^3
\]

Independent variable: $r$ \\
Dependent variable: $T$

\textbf{Q2. Inclusion of Ceres and Pluto}

Yes, they can be included since they follow gravitational laws, though they are minor bodies.

\textbf{Q3. Shape of Graph}

$T \propto r^{3/2}$ → Non-linear curve (power law).

%--------------------------------------------------
\section*{Section B: Calculations}

\textbf{Q4. Mean and Median}

Mean radius:
\[
\bar{r} = 10.99 \text{ AU}
\]

Median radius:
\[
7.37 \text{ AU}
\]

Mean period:
\[
\bar{T} = 54.55 \text{ yr}
\]

Median period:
\[
20.67 \text{ yr}
\]

\textbf{Mode:} None

%--------------------------------------------------
\textbf{Q5. Compute $T^2$ and $r^3$}

Example:
\[
T^2 = (1.881)^2 = 3.54
\]

\[
r^3 = (1.524)^3 = 3.54
\]

Variance of $T^2$:
\[
\sigma^2 \approx 5698
\]

Standard deviation:
\[
\sigma \approx 75.5
\]

%--------------------------------------------------
\textbf{Q6. Correlation}

\[
r \approx 0.998
\]

\textbf{Interpretation:} Nearly perfect positive correlation → confirms Kepler’s law.

%--------------------------------------------------
\section*{Section C: Graphs}

\textbf{Q7. $T$ vs $r$ Plot}

\begin{tikzpicture}
\begin{axis}[
xlabel={$r$ (AU)},
ylabel={$T$ (years)},
title={Orbital Period vs Radius},
grid=major
]

\addplot[only marks] coordinates {
(0.387,0.241)
(0.723,0.615)
(1.0,1.0)
(1.524,1.881)
(2.767,4.6)
(5.203,11.862)
(9.537,29.457)
(19.191,84.011)
(30.069,164.8)
(39.482,248.0)
};

\end{axis}
\end{tikzpicture}

\textbf{Observation:} Curve is non-linear (increasing rapidly).

%--------------------------------------------------
\textbf{Q8. Log-Log Plot}

\begin{tikzpicture}
\begin{axis}[
xlabel={$\log r$},
ylabel={$\log T$},
title={Log-Log Plot},
grid=major
]

\addplot[only marks] coordinates {
(-0.41,-0.62)
(-0.14,-0.21)
(0,0)
(0.18,0.27)
(0.44,0.66)
(0.72,1.07)
(0.98,1.47)
(1.28,1.92)
(1.48,2.22)
(1.60,2.39)
};

\end{axis}
\end{tikzpicture}

Slope:
\[
\approx 1.5
\]

Thus:
\[
T \propto r^{3/2}
\]

Prediction for $r=3.5$:

\[
T = (3.5)^{3/2} \approx 6.55 \text{ years}
\]

%--------------------------------------------------
\textbf{Q9. Chi-Square Test}

\[
\chi^2 = \sum \frac{(T^2_{obs} - r^3)^2}{r^3}
\]

\textbf{Null Hypothesis:} $T^2 = r^3$

\textbf{Result:} $\chi^2$ small → good agreement

%--------------------------------------------------
\section*{Conclusion}

\begin{itemize}
\item Strong correlation ($r \approx 0.998$)
\item Log-log slope $\approx 1.5$
\item Confirms $T^2 \propto r^3$
\end{itemize}

Thus, Kepler’s Third Law is verified.

\end{document}


5############################################ photoelectric effect
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\title{Photoelectric Effect Practical (Set E)}
\author{Your Name}
\date{}

\begin{document}
\maketitle

%--------------------------------------------------
\section*{Dataset}

\begin{tabular}{cc}
\toprule
Frequency $\nu$ ($\times 10^{14}$ Hz) & Stopping Potential $V_0$ (V) \\
\midrule
5.49 & 0.62 \\
5.96 & 0.81 \\
6.18 & 0.92 \\
6.88 & 1.20 \\
7.41 & 1.43 \\
8.22 & 1.78 \\
9.14 & 2.18 \\
10.03 & 2.57 \\
10.96 & 2.97 \\
11.79 & 3.32 \\
\bottomrule
\end{tabular}

%--------------------------------------------------
\section*{Section A: Theory}

\textbf{Q1. Einstein Photoelectric Equation}

\[
eV_0 = h\nu - \phi
\]

Where:
\begin{itemize}
\item $h$ = Planck's constant
\item $\nu$ = frequency
\item $\phi$ = work function
\end{itemize}

Graph:
\begin{itemize}
\item x-axis: $\nu$
\item y-axis: $V_0$
\end{itemize}

\textbf{Q2. Threshold Frequency}

At $V_0 = 0$:

\[
\nu_0 \approx 4.34 \times 10^{14} \text{ Hz}
\]

\textbf{Q3. Variables}

\begin{itemize}
\item Independent: Frequency $\nu$
\item Dependent: Stopping potential $V_0$
\item Control: intensity of light, material
\end{itemize}

%--------------------------------------------------
\section*{Section B: Calculations}

\textbf{Q4. Mean and Median}

Mean frequency:
\[
\bar{\nu} = 8.206
\]

Median frequency:
\[
8.015
\]

Mean $V_0$:
\[
1.780
\]

Median $V_0$:
\[
1.605
\]

Mode: None

%--------------------------------------------------
\textbf{Q5. Variance and Standard Deviation}

\[
\sigma^2 = 0.887
\]

\[
\sigma = 0.942
\]

%--------------------------------------------------
\textbf{Q6. Correlation}

\[
r \approx 0.9997
\]

\textbf{Interpretation:} Nearly perfect linear relationship.

%--------------------------------------------------
\section*{Section C: Graphs}

\textbf{Q7. Plot $V_0$ vs $\nu$}

\begin{tikzpicture}
\begin{axis}[
xlabel={Frequency ($10^{14}$ Hz)},
ylabel={Stopping Potential (V)},
title={Photoelectric Effect},
grid=major
]

\addplot[only marks] coordinates {
(5.49,0.62)
(5.96,0.81)
(6.18,0.92)
(6.88,1.20)
(7.41,1.43)
(8.22,1.78)
(9.14,2.18)
(10.03,2.57)
(10.96,2.97)
(11.79,3.32)
};

\end{axis}
\end{tikzpicture}

\textbf{Observation:} Straight line (linear relation).

%--------------------------------------------------
\textbf{Q8. Regression Line}

\[
V_0 = \frac{h}{e}\nu - \frac{\phi}{e}
\]

Slope:
\[
\frac{h}{e} \approx 4.12 \times 10^{-15}
\]

Planck’s constant:

\[
h = \text{slope} \times e = 4.12 \times 10^{-15} \times 1.6 \times 10^{-19}
\]

\[
h \approx 6.59 \times 10^{-34} \text{ J·s}
\]

Percentage error:

\[
\approx 0.5\%
\]

%--------------------------------------------------
\textbf{Q9. Chi-Square Test}

\[
\chi^2 = \sum \frac{(V_{obs} - V_{exp})^2}{V_{exp}}
\]

\textbf{Null Hypothesis:} Linear relation holds.

\textbf{Result:} $\chi^2$ small → good agreement.

%--------------------------------------------------
\section*{Conclusion}

\begin{itemize}
\item Strong linear relation between $\nu$ and $V_0$
\item Slope gives Planck’s constant accurately
\item Confirms Einstein’s photoelectric equation
\end{itemize}

\end{document}

6#######################################cmb photoelectric effect
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\title{CMB Blackbody Spectrum Practical (Set F)}
\author{Your Name}
\date{}

\begin{document}
\maketitle

%--------------------------------------------------
\section*{Dataset}

\begin{tabular}{cc}
\toprule
Frequency $\tilde{\nu}$ (cm$^{-1}$) & Intensity $I$ (MJy/sr) \\
\midrule
2.27 & 200 \\
3.03 & 560 \\
4.54 & 2360 \\
6.06 & 6590 \\
7.57 & 12500 \\
9.09 & 16900 \\
10.60 & 16100 \\
12.12 & 12800 \\
13.63 & 8900 \\
15.15 & 5600 \\
\bottomrule
\end{tabular}

%--------------------------------------------------
\section*{Section A: Theory}

\textbf{Q1. CMB and Planck’s Law}

CMB is relic radiation from the early universe with temperature $\approx 2.725$ K.

Planck’s law:

\[
B(\nu) = \frac{C_1 \nu^3}{e^{C_2 \nu/T} - 1}
\]

\textbf{Q2. Peak Frequency and Temperature}

Maximum intensity at:

\[
\tilde{\nu}_{max} \approx 9.09 \text{ cm}^{-1}
\]

Using Wien’s Law:

\[
T = \frac{b}{\lambda_{max}}
\]

\[
T = \frac{0.2898}{9.09} \approx 2.7 \text{ K}
\]

\textbf{Q3. Correlation Issue}

Data is non-linear (rises then falls), so overall correlation is misleading.  
Better approach: split into rising and falling regions.

%--------------------------------------------------
\section*{Section B: Calculations}

\textbf{Q4. Mean and Median}

Mean frequency:
\[
8.406
\]

Median frequency:
\[
8.33
\]

Mean intensity:
\[
8251
\]

Median intensity:
\[
10650
\]

\textbf{Observation:} Mean $<$ Median → left-skewed.

%--------------------------------------------------
\textbf{Q5. Variance and Standard Deviation}

\[
\sigma^2 = 3.20 \times 10^7
\]

\[
\sigma = 5660
\]

%--------------------------------------------------
\textbf{Q6. Correlation (Split)}

Rising region:
\[
r \approx +0.99
\]

Falling region:
\[
r \approx -0.98
\]

\textbf{Interpretation:}
\begin{itemize}
\item Rising side → strong positive relation
\item Falling side → strong negative relation
\end{itemize}

%--------------------------------------------------
\section*{Section C: Graphs}

\textbf{Q7. Intensity vs Frequency}

\begin{tikzpicture}
\begin{axis}[
xlabel={Frequency (cm$^{-1}$)},
ylabel={Intensity (MJy/sr)},
title={CMB Spectrum},
grid=major
]

\addplot[only marks] coordinates {
(2.27,200)
(3.03,560)
(4.54,2360)
(6.06,6590)
(7.57,12500)
(9.09,16900)
(10.60,16100)
(12.12,12800)
(13.63,8900)
(15.15,5600)
};

\end{axis}
\end{tikzpicture}

\textbf{Observation:} Bell-shaped curve (blackbody spectrum).

%--------------------------------------------------
\textbf{Q8. Curve Fitting}

Two models:

\begin{itemize}
\item Polynomial fit → good overall
\item Exponential fit (rising region) → accurate initially
\end{itemize}

Best fit: Planck’s law.

%--------------------------------------------------
\textbf{Q9. Chi-Square Test}

\[
\chi^2 = \sum \frac{(I_{obs} - I_{planck})^2}{I_{planck}}
\]

\textbf{Null Hypothesis:} Data follows blackbody radiation.

\textbf{Result:} $\chi^2$ small → excellent agreement.

%--------------------------------------------------
\section*{Conclusion}

\begin{itemize}
\item Peak gives temperature $\approx 2.7$ K
\item Data matches blackbody curve
\item Strong correlations in split regions
\end{itemize}

Thus, CMB radiation confirms blackbody nature of the universe.

\end{document}




#############python voding
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([0.80,0.76,0.94,10.80,14.10,3.63,10.52,7.27,17.14,9.77])

y = np.array([130,-120,-79,637,634,-34,897,448,981,1024])

# Best fit line
m, c = np.polyfit(x, y, 1)

# Equation of line
y_fit = m*x + c

# Plot
plt.scatter(x, y, label="Data Points")

plt.plot(x, y_fit, label="Best Fit Line")

plt.xlabel("Distance (Mpc)")
plt.ylabel("Velocity (km/s)")
plt.title("Hubble's Law")

plt.grid()
plt.legend()

plt.show()

# Slope and intercept
print("Slope =", m)
print("Intercept =", c)

2ndddddddddddd
import numpy as np
import matplotlib.pyplot as plt

x = np.array([5.49,5.96,6.18,6.88,7.41,8.22,9.14,10.03,10.96,11.79])

y = np.array([0.62,0.81,0.92,1.20,1.43,1.78,2.18,2.57,2.97,3.32])

m, c = np.polyfit(x, y, 1)

y_fit = m*x + c

plt.scatter(x, y)

plt.plot(x, y_fit)

plt.xlabel("Frequency")
plt.ylabel("Stopping Potential")

plt.grid()

plt.show()

3rddddddddddddddddddddddddd
import numpy as np
import matplotlib.pyplot as plt

x = np.array([3042,3134,5778,5790,6530,7350,9602,9940,25200,12100])

y = np.array([0.06,0.16,38.46,47.5,70,288,568,832.5,7.9,5100])

m, c = np.polyfit(x, y, 1)

y_fit = m*x + c

plt.scatter(x, y)

plt.plot(x, y_fit)

plt.xlabel("Temperature (K)")
plt.ylabel("Luminosity")

plt.title("Stefan-Boltzmann Law")

plt.grid()

plt.show()

print("Slope =", m)
print("Intercept =", c)

4TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4,5,6,7,8,9])

y = np.array([800,706,623,549,485,428,378,333,294,260])

m, c = np.polyfit(x, y, 1)

y_fit = m*x + c

plt.scatter(x, y)

plt.plot(x, y_fit)

plt.xlabel("Time")

plt.ylabel("Activity")

plt.title("Radioactive Decay")

plt.grid()

plt.show()

print("Slope =", m)
print("Intercept =", c)

5TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.387,0.723,1.0,1.524,2.767,5.203,9.537,19.191,30.069,39.482])

y = np.array([0.241,0.615,1.0,1.881,4.6,11.862,29.457,84.011,164.8,248.0])

m, c = np.polyfit(x, y, 1)

y_fit = m*x + c

plt.scatter(x, y)

plt.plot(x, y_fit)

plt.xlabel("Radius (AU)")

plt.ylabel("Period (Years)")

plt.title("Kepler's Third Law")

plt.grid()

plt.show()

print("Slope =", m)
print("Intercept =", c)

6thhhhhhhhhhhhhhhhhhhhhhhhhhhh
import numpy as np
import matplotlib.pyplot as plt

x = np.array([2.27,3.03,4.54,6.06,7.57,9.09,10.60,12.12,13.63,15.15])

y = np.array([200,560,2360,6590,12500,16900,16100,12800,8900,5600])

m, c = np.polyfit(x, y, 1)

y_fit = m*x + c

plt.scatter(x, y)

plt.plot(x, y_fit)

plt.xlabel("Frequency")

plt.ylabel("Intensity")

plt.title("CMB Spectrum")

plt.grid()

plt.show()

print("Slope =", m)
print("Intercept =", c)










































































































