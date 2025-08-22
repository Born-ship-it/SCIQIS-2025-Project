# SCIQIS-2025-Project
This Github project is a part of the DTU course:
10387 Scientific computing in quantum information science (link: https://kurser.dtu.dk/course/10387)

This project is part of the exam. Cavity QED was a topic covered doing the course, which I drove deeper into. I choose this as my interest alings wiht its direct relation and application to quantum information science. Done by adapting and writing code in python/ipython notebooks.

Install packages:
<pre><code>
pip install git+https://github.com/Born-ship-it/SCIQIS-2025-Project.git
</code></pre>

Clone repository:
<pre><code>
git clone https://github.com/Born-ship-it/SCIQIS-2025-Project.git
cd SCIQIS-2025-Project
pip install .
</code></pre>

Install dependencies inside your environment with:
<pre><code>
run uv sync
</code></pre>
utliizing the .toml file from the repo.


## Computational Quantum Optomechanics

Developed a comprehensive numerical framework using QuTiP to simulate and analyze quantum optomechanical systems. Implemented master equation solutions for investigating:

**Quantum Dynamical Properties:**
- Time evolution of expectation values for system observables
- Linear entropy calculations quantifying quantum coherence
- Hilbert space truncation validation through commutator analysis
- Fock state population distribution dynamics
- Wigner quasi-probability distributions in phase space
- Rabi oscillation characterization and frequency extraction

**Technical Achievements:**
- Built interactive visualization suite using matplotlib.animation
- Created comparative analysis between atomic-cavity and optomechanical systems
- Developed automated data processing pipelines for quantum property extraction

**Outputs:**
- Atom-Cavity.ipynb: a single two-level atom interacting with a electromagnetic cavity
- Cavity-Mirror.ipynb: a moving mirror interacting with a electromagnetic cavity

## Course relevant/reflecting stuff

Made good use of the QuTiP library for simulating time evolution of Hamiltonians using master equation solver, doing the course I have finished doing the Jaynes-Cumming model of atom-cavity and cavity-mirror (Bose et al. PRA 56, 4175 (1997)) system. Used core scientific computing packages for Python: Numpy, Scipy, Matplotlib.

I cover computing topics from the course, such as profiling and speeding up my code (timing, line profiling, arrays vs. loops, caching, numexpr, Numba, arrays), visually the physics (Interactive visualisation with ipywidgets).

Got introduced to Github, using basic version control with git and managing a small the project.

Adapted python modules with to import modules from ipython notebooks and python files.

The course encouraged using AI to speed up programming, etc., on the condition I could understand what my code does.