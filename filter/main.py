import argparse
import sympy as sp
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from filter.specs import LPFSpec, OrdFreq
from filter.tf_design import FilterType, design_lpf, plot_filters_gain, plot_filters_group_delay, group_delay_variation
from filter.topology_analysis import *
from filter.optimize import fit_filter_circuit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Design a low-pass filter')
    parser.add_argument('--save-plots', dest='save_plots', action='store_true', help='save generated plots in figs/')
    parser.add_argument('--show-plots', dest='show_plots', action='store_true', help='show generated plots')
    args = parser.parse_args()

    spec = LPFSpec(
        passband_corner=OrdFreq(20e6).w(),
        stopband_corner=OrdFreq(200e6).w(),
        stopband_atten=55,
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
        print("\t{}".format(ba.to_zpk()))
        print("\t{}".format(ba))
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
        'Cload': 40e-15
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

    lpf, subs, nsrcs = build_lpf([FT.SK, FT.SK], [sk0spec, sk0spec])
    rac = run_ac(lpf)
    tf = run_sym(lpf, 'V1', True)
    fit_filter_circuit(ftype_specs[FilterType.BUTTERWORTH], tf, [sk0spec, sk0spec])
    #hs = check_specs(lpf, rac, tf, spec, subs, nsrcs)
    if args.show_plots:
        plot_final_filter(rac, hs, spec)

    if args.show_plots:
        plt.show()
