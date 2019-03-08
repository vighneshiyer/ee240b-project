- Use PySpice to programmatically construct a SPICE netlist describing the filter.
- Use Akhab to perform AC symbolic analysis on a SPICE netlist.
- Explore using NGSPICE as a local spice engine for iterating on the design away from the server farm.
- Potential flow:
    1. Use scipy to calculate filter ZPK for a given filter style and input parameter set (look at the extra credit stuff for what to support)
    1. Use PySpice to construct an abstract netlist with R/C/L (as parameters) and ideal op-amps (just a differential gain VCVS) given a particular filter topology (multiple feedback, sallen-key, etc.) and of a particular order
    1. Take the generated netlist and feed it into Ahkab and run a symbolic AC analysis which should give us the SS transfer function
    1. Using sympy, take the symbolic transfer function, find the symbolic expression and poles, zeros and DC gain. Then equate the desired ZPK of the filter to the symbolic expressions, run a solver, and find optimal values of R/C/L
    1. Plug in these physical values into the netlist and run a SPICE simulation to verify correctness
    1. Add op-amp non-idealities and potentially re-run symbolic analysis or just begin stochastic tweaking of the R/C/L to compensate out the non-idealities
    1. Perform noise analysis to analyze the dynamic range of the filter and continue tweaking using the same stochastic search
    1. Estimate power consumption by again performing symbolic analysis of the bias currents at DC and then doing P = IV.
