import math
from filter.tf_design import LPFSpec, FilterType, design_lpf, plot_filters, AngularFreq, OrdFreq


class main:
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
        print(ftype_specs)
        plot_filters(ftype_specs, spec)
