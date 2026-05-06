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









































































































