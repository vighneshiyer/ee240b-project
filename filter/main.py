from filter.tf_design import LPFSpec, FilterType, design_lpf, OrdFreq, plot_filters_gain, plot_filters_group_delay, group_delay_variation
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt


if __name__ == "__main__":
    spec = LPFSpec(
        passband_corner=OrdFreq(20e6).to_w(),
        stopband_corner=OrdFreq(200e6).to_w(),
        stopband_atten=55,
        passband_ripple=1,
        group_delay_variation=3e-9
    )
    ftype_specs = {}  # Dict[FilterType, BA]
    for ftype in FilterType:
        ba = design_lpf(spec, ftype=ftype)
        print("Filter type {} requires order {}".format(ftype.value, ba.order()))
        ftype_specs[ftype] = ba
        print(group_delay_variation(ba, spec))
    print(ftype_specs)

    plt.style.use('ggplot')
    width, height = figaspect(1/2)
    fig, ax = plt.subplots(2, 1, figsize=(width, height))
    plot_filters_gain(ftype_specs, spec, ax[0])
    plot_filters_group_delay(ftype_specs, spec, ax[1])
    plt.tight_layout()
    plt.show()
