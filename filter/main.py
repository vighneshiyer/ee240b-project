import argparse
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from filter.tf_design import FilterType, design_lpf, plot_filters_gain, plot_filters_group_delay, group_delay_variation
from filter.lut_construction import construct_ideal_lut
from filter.topology_analysis import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Design a low-pass filter')
    parser.add_argument('--save-plots', dest='save_plots', action='store_true', help='save generated plots in figs/')
    parser.add_argument('--show-plots', dest='show_plots', action='store_true', help='show generated plots')
    parser.add_argument('--ota-analysis', dest='ota_analysis', action='store_true', help='perform symbolic analysis of an OTA filter')
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
        print("\tPoles: {}".format(ba.to_zpk().P))
        print("\tZeros: {}".format(ba.to_zpk().Z))
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
        ideal_lut = construct_ideal_lut(desired_filter=ftype_specs[FilterType.BUTTERWORTH])
        #print(ideal_lut)
        # denom_simple = sp.fraction(tf_gain)[1].expand().collect(s)
        # tf_gain = sp.fraction(tf_gain)[0] / denom_simple
        # sp.pprint(tf_gain.subs({R: 1/g1}).simplify())
        # fit_filter_circuit(ftype_specs[FilterType.BUTTERWORTH], tf, [sk0spec, sk0spec])
        # amp_bw = {
        #   'G1_0': ota3spec.bw,
        #   'G1_1': ota3spec.bw
        # }
        # poles = tf['poles']
        # poles = list(map(lambda p: p.subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm}), poles))
        # sp.pprint(poles)
        # sp.pprint(tf['zeros'])
        # hs = check_specs(lpf, rac, tf, spec, subs, nsrcs, amp_bw)
        if args.show_plots:
            plot_final_filter(rac, hs, spec)
