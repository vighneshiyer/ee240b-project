from filter.tf_design import FilterType, design_lpf, plot_filters_gain, plot_filters_group_delay, group_delay_variation, freq_range
from filter.lut_construction import construct_ideal_lut
from filter.topology_analysis import *
from scipy.signal import freqs
import argparse
import sympy as sp
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
from functools import reduce
from scipy.optimize import minimize
plt.style.use('ggplot')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Design a low-pass filter')
    parser.add_argument('--save-plots', dest='save_plots', action='store_true', help='save generated plots in figs/')
    parser.add_argument('--show-plots', dest='show_plots', action='store_true', help='show generated plots')
    parser.add_argument('--ota-analysis', dest='ota_analysis', action='store_true', help='perform symbolic analysis of an OTA filter')
    args = parser.parse_args()

    spec = LPFSpec(
        passband_corner=OrdFreq(20e6).w(),
        stopband_corner=OrdFreq(200e6).w(),
        stopband_atten=25,
        passband_ripple=1,
        group_delay_variation=3e-9,
        dynamic_range=50
    )
    ftype_specs = {}  # Dict[FilterType, BA]
    for ftype in FilterType:
        ba = design_lpf(spec, ftype=ftype)
        print("Filter type {} requires order {} with group delay {}ns".format(
            ftype.value, ba.order(),
            round(group_delay_variation(ba, spec)*1e9, 3)))
        print("\tPoles: {}".format(ba.to_zpk().P))
        print("\tZeros: {}".format(ba.to_zpk().Z))
        print("\tK: {}".format(ba.to_zpk().K))
        print("\tBA: {}".format(ba))
        ftype_specs[ftype] = ba

    if args.save_plots or args.show_plots:
        print("Plotting filter gain")
        fig1, ax = plt.subplots(1, 1, figsize=figaspect(1/3))
        plot_filters_gain(ftype_specs, spec, ax)
        if args.save_plots:
            fig1.savefig('figs/tf_gain.pdf', bbox_inches='tight')
        fig2, ax = plt.subplots(1, 1, figsize=figaspect(1/3))
        print("Plotting filter group delay")
        plot_filters_group_delay(ftype_specs, spec, ax)
        if args.save_plots:
            fig2.savefig('figs/tf_group_delay.pdf', bbox_inches='tight')

    nonideal_dict = {
        'Av': 100,
        'RO': 200,
        'Cload': 40e-15,
        'gamma': 2, # short-channel?
        'gm': 0.01 #
    }

    ac_params = {
        'start': 1e3,
        'stop': 1e9,
        'pts': 100
    }
    Rbase = 900
    m = 1.5
    Cbase = 9e-12
    n = 1.5
    sk0spec = SallenKeySpec(
        r1=Rbase*m,
        r2=Rbase/m,
        c1=Cbase*n,
        c2=Cbase/n,
        e1=nonideal_dict['Av'],
        ro=nonideal_dict['RO']
    )
    mfbspec = MFBSpec(
        r1=Rbase/m,
        r2=Rbase/m,
        r3=Rbase*m,
        c1=Cbase*n,
        c2=Cbase/n,
        e1=nonideal_dict['Av'],
        ro=nonideal_dict['RO']
    )
    ota3spec = OTA3Spec(
        r1=Rbase,
        c1=Cbase,
        c2=Cbase,
        gm=nonideal_dict['gm'],
        ro=nonideal_dict['Av']/nonideal_dict['gm'],
        bw=1e8
    )

    if args.ota_analysis:
        # Step 1: take the ideal transfer function, and for reasonable ranges of gm, R, solve for the necessary C
        # to build a LUT of potential design points
        filter_poles = ftype_specs[FilterType.BUTTERWORTH].to_zpk().P
        chosen_filter = ftype_specs[FilterType.BUTTERWORTH]
        ideal_lut = construct_ideal_lut(desired_filter=ftype_specs[FilterType.BUTTERWORTH])
        #print(ideal_lut)

        # Step 2: nonideality analysis
        lpf, subs, nsrcs = build_lpf([FT.OTA3], [ota3spec], ro=True, cl=False)
        tf = run_sym(lpf, 'V1', True)

        C1_0, C2_0, G1_0, R1_0, RO_0 = sp.symbols('C1_0 C2_0 G1_0 R1_0 RO_0')
        C, R, gm0, ro, wbw = sp.symbols('C R gm0 r_o wbw', real=True)
        s = sp.symbols('s')
        #sym_poles = [p.subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm0 / (1 - (s/wbw)), RO_0: ro}) for p in tf['poles']]
        sym_gain = tf['gain'].subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm0 / (1 - (s/wbw)), RO_0: ro})
        #sym_poles = sp.solve(sp.fraction(sym_gain)[1], s)
        #sp.pprint(sym_poles)
        sp.pprint(sym_gain)

        sym_gain_subs = sym_gain.subs({ro: 20 / gm0, wbw: 1 / ((20 / gm0) * 400e-18)})
        sp.pprint(sym_gain_subs)
        #sym_poles_lambda = list(map(lambda p: sp.lambdify([R, C, gm0], p), sym_poles_subs))
        sym_gain_lambda = sp.lambdify([R, C, gm0, s], sym_gain_subs)

        w = freq_range(spec, 50)
        w, h = freqs(chosen_filter.B, chosen_filter.A, worN=w)

        def cost(y):
            #print(y)
            actual_mag = list(map(lambda s: sym_gain_lambda(y[0], y[1], y[2], s), w*1j))
            return np.linalg.norm(actual_mag - h, 2)

        res = minimize(cost, x0=ideal_lut[0], method='Nelder-Mead',
                       options={'maxfev': 10000, 'xatol': 1e-3, 'fatol': 1e-12, 'adaptive': False})

        print(res)
        print(ideal_lut[0])

        res = minimize(cost, x0=ideal_lut[10], method='Nelder-Mead',
                       options={'maxfev': 10000, 'xatol': 1e-3, 'fatol': 1e-12, 'adaptive': False})

        print(res)
        print(ideal_lut[10])

        if args.show_plots:
            plot_final_filter(rac, hs, spec)
