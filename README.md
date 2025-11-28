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


p0=0;
p1=1;
n=6;
f[x_]:=x^3-5*x+1;
Plot[f[x],{x,0,1}]
For[i=1,i<=n,i++,{p=p1-(p1-p0)/(f[p1]-f[p0])*f[p1],p0=p1,p1=p ,Print[N[p]]}]








































##########
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants (set hbar^2/2m = 1 for simplicity)
L = 1.0  # length of the box
x_vals = np.linspace(0, L, 500)

def schrodinger(x, y, E):
    psi, phi = y
    dpsi_dx = phi
    dphi_dx = -E * psi   # since hbar^2/2m = 1
    return [dpsi_dx, dphi_dx]

def shoot(E):
    # y = [psi, dpsi/dx]
    sol = solve_ivp(schrodinger, [0, L], [0, 1], t_eval=x_vals, args=(E,))
    psi_L = sol.y[0, -1]
    return psi_L, sol


# Find energy eigenvalues by scanning
energies = np.linspace(0, 200, 2000)
psi_L_vals = []

for E in energies:
    psi_L, _ = shoot(E)
    psi_L_vals.append(psi_L)

# Find approximate zeros → eigenvalues
sign_change_indices = np.where(np.diff(np.sign(psi_L_vals)))[0]
eigen_values = energies[sign_change_indices]
print("Estimated Eigenvalues:", eigen_values)

# Plot first 3 wavefunctions
plt.figure(figsize=(10, 7))
for i, E in enumerate(eigen_values[:3]):
    psi_L, sol = shoot(E)
    psi = sol.y[0]
    psi /= np.max(np.abs(psi))  # Normalize for plotting
    plt.plot(x_vals, psi, label=f"Eigenstate {i+1}, E ≈ {E:.2f}")

plt.title("Particle in a Box - Wavefunctions via Shooting Method")
plt.xlabel("Position x")
plt.ylabel("Wavefunction ψ(x)")
plt.legend()
plt.grid(True)
plt.show()

#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Well parameters
a = 1.0        # half width
V0 = 50.0      # well depth
x_min, x_max = -2, 2   # boundaries
N = 1000
x_vals = np.linspace(x_min, x_max, N)

# Finite Potential Well Function
def V(x):
    return 0 if abs(x) <= a else V0

# Schrödinger Equation (converted to 1st order form)
def schrodinger(x, y, E):
    psi, phi = y
    return [phi, - (E - V(x)) * psi]

# Shooting Method: returns psi at boundary for given E
def shooting(E):
    y0 = [0, 1]  # psi(x_min)=0, psi'(x_min)=1
    sol = solve_ivp(schrodinger, [x_min, x_max], y0, t_eval=x_vals, args=(E,))
    psi = sol.y[0]
    return psi[-1], psi

# Scan energies to detect sign change which indicates eigenvalues
energies = np.linspace(0.1, V0 - 0.1, 500)
psi_boundary = []

for E in energies:
    psi_L, _ = shooting(E)
    psi_boundary.append(psi_L)

# Find eigen energy where sign flips → ψ(x_max)=0
indexes = np.where(np.diff(np.sign(psi_boundary)))[0]
eigenE = energies[indexes]
print("Eigen Energies Found:", eigenE)

# Plot wavefunctions of first 3 bound states
plt.figure(figsize=(10,7))
for i, E in enumerate(eigenE[:3]):
    psi_L, psi = shooting(E)
    psi /= np.max(np.abs(psi))  # normalize
    plt.plot(x_vals, psi, label=f"n={i+1}, E={E:.2f}")

plt.axvline(-a, color='k', linestyle='--')
plt.axvline(a, color='k', linestyle='--')
plt.title("Finite Potential Well — Shooting Method")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.grid(True)
plt.legend()
plt.show()

##
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Range and resolution
x_min, x_max = -6, 6
N = 2000
x_vals = np.linspace(x_min, x_max, N)

# Schrodinger Equation (1st order format)
def schrodinger(x, y, E):
    psi, phi = y
    dpsi_dx = phi
    dphi_dx = (x**2 - 2*E) * psi
    return [dpsi_dx, dphi_dx]

# Shooting method (even states -> psi(0)=1, phi(0)=0)
def shoot(E, parity="even"):
    if parity == "even":
        init = [1.0, 0.0]  # ψ(0)=1, ψ'(0)=0
    else:
        init = [0.0, 1.0]  # ψ(0)=0, ψ'(0)=1 (odd states)
    
    sol = solve_ivp(schrodinger, [0, x_max], init,
                    t_eval=np.linspace(0, x_max, N),
                    args=(E,))
    
    psi = sol.y[0]
    return psi[-1], psi, sol.t

# Scan energies to find eigenvalues
energies = np.linspace(0, 10, 500)
psi_end_even, psi_end_odd = [], []

for E in energies:
    pe,_,_ = shoot(E, "even")
    po,_,_ = shoot(E, "odd")
    psi_end_even.append(pe)
    psi_end_odd.append(po)

# find sign changes → eigenvalues
even_idx = np.where(np.diff(np.sign(psi_end_even)))[0]
odd_idx = np.where(np.diff(np.sign(psi_end_odd)))[0]

EVEN_E = energies[even_idx]
ODD_E = energies[odd_idx]

print("Even State Energies:", EVEN_E)
print("Odd State Energies:", ODD_E)

# Plot first 3 states
plt.figure(figsize=(10,7))

for i, E in enumerate(sorted(np.concatenate((EVEN_E, ODD_E)))[:3]):
    last_val, psi_half, t_half = shoot(E, "even" if i%2==0 else "odd")
    
    # Mirror wavefunction for negative x
    psi = np.concatenate(((-1)**i * psi_half[::-1], psi_half))
    x_full = np.concatenate((-t_half[::-1], t_half))
    
    psi /= np.max(np.abs(psi))  # normalize for plot
    
    plt.plot(x_full, psi, label=f"n={i}, E={E:.2f}")

plt.title("Quantum Harmonic Oscillator - Shooting Method")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.grid(True)
plt.legend()
plt.show()

######
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Dimensionless Hydrogen radial Schrödinger equation
def schrodinger(rho, y, l, epsilon):
    u, uprime = y
    V = (l*(l+1)/rho**2) - (2/rho)  # effective potential
    return [uprime, (V + 2*epsilon) * u]

# Shooting method for a given energy
def shoot(epsilon, l):
    rho = np.linspace(1e-6, 20, 2000)
    sol = solve_ivp(schrodinger, [rho[0], rho[-1]],
                    [0, 1e-4], args=(l, epsilon),
                    t_eval=rho, method="RK45")
    return sol.t, sol.y[0], sol.y[0][-1]

# Find eigenvalues by scanning energy range
def find_energy(l):
    eps_range = np.linspace(-1.5, -0.1, 500)
    boundary_vals = []

    for eps in eps_range:
        _, _, val = shoot(eps, l)
        boundary_vals.append(val)

    idx = np.where(np.diff(np.sign(boundary_vals)))[0]
    return eps_range[idx]

# Find l=0 eigen-energies (1s, 2s)
energies = find_energy(l=0)
print("Dimensionless Eigen Energies Found:", energies[:2])

# Plot wavefunctions
plt.figure(figsize=(10, 7))

for i, eps in enumerate(energies[:2]):
    rho, u, _ = shoot(eps, 0)
    u_norm = u / np.max(np.abs(u))
    plt.plot(rho, u_norm, label=f"State n={i+1}, ε={eps:.3f}")

plt.xlabel("ρ = r/a₀")
plt.ylabel("Radial Wavefunction u(ρ) (normalized)")
plt.title("Hydrogen Atom Wavefunctions (Shooting Method)")
plt.grid(True)
plt.legend()
plt.show()

# Convert to physical energy (eV)
E0 = -13.6  # Hydrogen ground state energy
physical_energies = energies[:2] * E0

print("\nPhysical Energy Eigenvalues (approx):")
print("Ground State (n=1):", physical_energies[0], "eV")
print("1st Excited State (n=2):", physical_energies[1], "eV")

#######
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Constants
hbar = 1.0545718e-34   # J*s
mH = 1.6735575e-27     # mass Hydrogen atom (kg)
mu = mH / 2            # Reduced mass for H2
eV = 1.60218e-19        # J → eV conversion

# Morse Potential parameters for H2
De = 4.744 * eV        # Convert to Joules
a = 1.942e10           # Convert Å⁻¹ → m⁻¹ (1 Å = 1e-10 m)
xe = 0.7414e-10        # Convert Å → meters

# Spatial grid
xmin, xmax = 0.1e-10, 3e-10  # Bond length range in meters
N = 2000
x = np.linspace(xmin, xmax, N)
dx = x[1] - x[0]

# Morse potential
V = De * (1 - np.exp(-a*(x - xe)))**2

# Finite Difference Hamiltonian matrix
diag = (hbar**2 / (mu * dx**2)) + V
off_diag = -hbar**2 / (2 * mu * dx**2) * np.ones(N-1)

# Hamiltonian matrix construction
H = np.diag(diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)

# Solve eigenvalue problem
E, psi = eigh(H)

# Extract ground state
E0 = E[0] / eV    # Convert J to eV
psi0 = psi[:, 0]
psi0_norm = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, x))

print("Lowest Vibrational Energy of H2:")
print(f"E0 ≈ {E0:.4f} eV")

# Plot wavefunction
plt.figure(figsize=(9,6))
plt.plot(x*1e10, psi0_norm, label="Ground State ψ₀")
plt.xlabel("Bond length x (Å)")
plt.ylabel("Normalized Wavefunction")
plt.title("H₂ Vibrational Ground State - Morse Potential")
plt.grid(True)
plt.legend()
plt.show()

######## finite differ
import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
m = 9.10938356e-31      # Electron mass (kg)
L = 1e-9                # Length of 1D box = 1 nm

# Discretization
N = 1000
dx = L / (N + 1)
x = np.linspace(0, L, N)

# Hamiltonian using finite difference method
diag = np.full(N, -2.0)
off_diag = np.ones(N - 1)
H = (-hbar**2 / (2 * m * dx**2)) * (np.diag(diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1))

# Solve eigenvalue problem
E, psi = np.linalg.eigh(H)

# Convert energy from Joules to eV
eV = 1.602176634e-19
energies_eV = E / eV

# Normalize wavefunctions
for i in range(3):
    psi[:, i] /= np.sqrt(np.trapz(psi[:, i]**2, x))

# Plot first three wavefunctions
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(x, psi[:, i], label=f"n={i+1}, E={energies_eV[i]:.3f} eV")

plt.title("Wavefunctions for Particle in a 1D Box (Finite Difference)")
plt.xlabel("Position x (m)")
plt.ylabel("Wavefunction ψ(x)")
plt.legend()
plt.grid(True)
plt.show()

# Print first few energy levels
for i in range(3):
    print(f"Energy Level {i+1}: {energies_eV[i]:.4f} eV")

##############
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# Constants
hbar = 1.054571817e-34  # J*s
m_red = 8.37e-28  # Reduced mass for H2 in kg

# Morse potential parameters for H2
De = 4.744 * 1.60218e-19  # Joules
a = 1.942e10  # 1/m (converted from Å^-1)
xe = 0.741e-10  # meters

# Spatial grid
xmin, xmax = 0.2e-10, 3e-10
N = 2000
x = np.linspace(xmin, xmax, N)
dx = x[1] - x[0]

def morse_potential(x):
    return De * (1 - np.exp(-a * (x - xe)))**2

def schrodinger(E):
    psi = np.zeros(N)
    phi = np.zeros(N)
    
    # Initial boundary conditions
    psi[0] = 0.0
    phi[0] = 1e-5
    
    for i in range(1, N):
        k1_psi = dx * phi[i-1]
        k1_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1]) - E)*psi[i-1]

        k2_psi = dx * (phi[i-1] + 0.5*k1_phi)
        k2_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1] + dx/2) - E)*(psi[i-1] + 0.5*k1_psi)

        k3_psi = dx * (phi[i-1] + 0.5*k2_phi)
        k3_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1] + dx/2) - E)*(psi[i-1] + 0.5*k2_psi)

        k4_psi = dx * (phi[i-1] + k3_phi)
        k4_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1]) - E)*(psi[i-1] + k3_psi)

        psi[i] = psi[i-1] + (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi) / 6
        phi[i] = phi[i-1] + (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi) / 6

    return psi[-1]

# Root finding for ground state energy
E_lower = 0.0
E_upper = De

E_ground = bisect(schrodinger, E_lower, E_upper)
E_eV = E_ground / 1.60218e-19

print("Ground State Vibrational Energy of H2:", round(E_eV, 4), "eV")

# Solve again to get psi for plotting
def solve_wavefunction(E):
    psi = np.zeros(N)
    phi = np.zeros(N)
    psi[0] = 0.0
    phi[0] = 1e-5
    
    for i in range(1, N):
        k1_psi = dx * phi[i-1]
        k1_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1]) - E)*psi[i-1]

        k2_psi = dx * (phi[i-1] + 0.5*k1_phi)
        k2_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1] + dx/2) - E)*(psi[i-1] + 0.5*k1_psi)

        k3_psi = dx * (phi[i-1] + 0.5*k2_phi)
        k3_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1] + dx/2) - E)*(psi[i-1] + 0.5*k2_psi)

        k4_psi = dx * (phi[i-1] + k3_phi)
        k4_phi = dx * (2*m_red/hbar**2)*(morse_potential(x[i-1]) - E)*(psi[i-1] + k3_psi)

        psi[i] = psi[i-1] + (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi) / 6
        phi[i] = phi[i-1] + (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi) / 6

    # Normalize
    psi /= np.sqrt(np.trapz(psi**2, x))
    return psi

psi_ground = solve_wavefunction(E_ground)

# Plot ground state wavefunction
plt.plot(x*1e10, psi_ground)
plt.title("Ground State Wavefunction - Morse Potential (H2)")
plt.xlabel("Bond Length x (Å)")
plt.ylabel("ψ(x)")
plt.grid()
plt.show()



































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

#euler
x[0]=0;
y[0]=1;
a=0;b=4;h=1;
f[x_y_]=x+y;
x[j_]:=y[j-1]+h;
y[j_]:=y[j-1]+h*f[x[j-1],y[j-1]];
Grid[Prepend[Table[{j,N[x[j],2],N[y[j],7]},{j,1,4}],{"j","x[j]","y[j]"}],
Dividers->{ALLTrue,ALLTrue}]



 ![Image](https://github.com/user-attachments/assets/43dd420d-d46a-4202-859c-942d63ac638b)





```
![Image](https://github.com/user-attachments/assets/262ac98c-724a-4e3f-a635-06fc175e6683)

![Image](https://github.com/user-attachments/assets/923209a5-6904-4eb6-9299-f22486629ad5)
<img width="1280" height="947" alt="image" src="https://github.com/user-attachments/assets/1670dc4a-2e19-4b3f-bc08-0a67a8ba3260" />



