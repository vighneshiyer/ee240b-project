import argparse
import sympy as sp
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from filter.specs import LPFSpec, OrdFreq
from filter.tf_design import FilterType, design_lpf, plot_filters_gain, plot_filters_group_delay, group_delay_variation
from filter.topology_analysis import build_sk2, run_ac, run_sym, check_specs
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
        group_delay_variation=3e-9
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

    if args.show_plots:
        plt.show()

    nonideal_dict = {
        'Av': 100,
        'RO': 200,
        'Cload': 40e-15
    }

    specs_dict = {
        'gain': 0,             #dB
        'passband': 20e6,      #Hz
        'pass_att': 3,         #dB
        'stopband': 200e6,     #Hz
        'stop_att': 55,        #dB
        'grp_del': 3,          #ns
        'gain_ripple': 1       #dB
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
    design_dict = { #keys must match the instance names of each component in design
        'R1': Rbase*m,
        'R2': Rbase/m,
        'C1': Cbase*n,
        'C2': Cbase/n,
        'CL': nonideal_dict['Cload'],
        'E1': nonideal_dict['Av'],
        'E2': nonideal_dict['Av'],
        'RO': nonideal_dict['RO']
    }

    lpf = build_sk2(design_dict)
    rac = run_ac(lpf)
    tf_v1 = run_sym(lpf, 'V1', False)
    fit_filter_circuit(spec, ftype_specs[FilterType.BUTTERWORTH], tf_v1, design_dict)
    """
    symb_poles = tf_v1['poles'][0]
    free_vars = list(symb_poles.free_symbols)
    print(free_vars)
    free_vars = list(filter(lambda x: 'R2' in str(x) or 'C2' in str(x), free_vars))
    print(free_vars)
    sol = sp.nonlinsolve([
        tf_v1['poles'][0] - zpk.P[0],
        tf_v1['poles'][1] - zpk.P[1]
    ], free_vars)
    sp.pprint(sol)
    print(sol)
    """
    #check_specs(rac, tf_v1, design_dict, specs_dict)
