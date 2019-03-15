import argparse
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
from filter.tf_design import FilterType, design_lpf, plot_filters_gain, plot_filters_group_delay, group_delay_variation
from filter.specs import LPFSpec, OrdFreq


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
        ftype_specs[ftype] = ba

    if args.save_plots or args.show_plots:
        plt.style.use('ggplot')
        width, height = figaspect(1/2)
        print("Plotting filter gain")
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        plot_filters_gain(ftype_specs, spec, ax)
        if args.show_plots:
            plt.show()
        if args.save_plots:
            plt.savefig('figs/tf_gain.pdf', bbox_inches='tight')
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        print("Plotting filter group delay")
        plot_filters_group_delay(ftype_specs, spec, ax)
        if args.show_plots:
            plt.show()
        if args.save_plots:
            plt.savefig('figs/tf_group_delay.pdf', bbox_inches='tight')
