from scipy.integrate import quad
import argparse
import sympy as sp
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
import sys
from itertools import zip_longest
import scipy

from filter.specs import ZPK
from filter.topology_analysis import TopologyAnalyzer
from filter.topologies import *
from filter.topology_construction import *
from filter.tf_design import *

plt.style.use('ggplot')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Design a low-pass filter')
    parser.add_argument('--save-plots', dest='save_plots', action='store_true', help='save generated plots in figs/')
    parser.add_argument('--show-plots', dest='show_plots', action='store_true', help='show generated plots')
    args = parser.parse_args()

    spec = LPFSpec(
        passband_corner=OrdFreq(25e6).w(),
        stopband_corner=OrdFreq(200e6).w(),
        stopband_atten=55,
        passband_ripple=1,
        group_delay_variation=2.5e-9,
        dynamic_range=50
    )
    ftype_specs = {}  # Dict[FilterType, BA]
    for ftype in FilterType:
        ba = design_lpf(spec, ftype=ftype)
        if ba is not None:
            print("Filter type {} requires order {} with group delay {}ns".format(
                ftype.value, ba.order(),
                round(group_delay_variation(ba, spec)*1e9, 3)))
            print("\tPoles: {}".format(ba.to_zpk().P))
            print("\tZeros: {}".format(ba.to_zpk().Z))
            print("\tK: {}".format(ba.to_zpk().K))
            #print("\tBA: {}".format(ba))
            ftype_specs[ftype] = ba

    """
    Trial of splitting the bessel filter in 2 2-pole sections
    bessel_zpk = ftype_specs[FilterType.BESSEL].to_zpk()
    bessel_1 = ZPK(Z=[], P=bessel_zpk.P[0:2], K=math.sqrt(bessel_zpk.K))
    bessel_2 = ZPK(Z=[], P=bessel_zpk.P[2:4], K=math.sqrt(bessel_zpk.K))
    ftype_specs = {
        FilterType.BESSEL: bessel_1.to_ba(),
        FilterType.BUTTERWORTH: bessel_2.to_ba()
    }
    """

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

    # Pick a filter type and split it into 2-pole stages, each stage to be implemented by one circuit topology
    desired_filter = ftype_specs[FilterType.BESSEL]

    def split_filter(ba: BA) -> List[BA]:
        #tf2sos(ba.B, ba.A)
        zpk = ba.to_zpk()
        assert len(zpk.Z) == 0, "I haven't handled zeros"
        stage_poles = list(zip_longest(*(iter(zpk.P),) * 2))
        stage_zpks = [ZPK([], p, np.power(zpk.K, 1/len(stage_poles))) for p in stage_poles]
        return stage_zpks
        #return [z.to_ba() for z in stage_zpks]

    #desired_filter_split = split_filter(desired_filter)
    desired_filter_zpk = desired_filter.to_zpk()
    desired_filter_split = [
            ZPK(desired_filter_zpk.Z, desired_filter_zpk.P[0:2], np.power(desired_filter_zpk.K, 1/2)),
            ZPK(desired_filter_zpk.Z, desired_filter_zpk.P[3:5], np.power(desired_filter_zpk.K, 1/2))
    ]
    print("Chosen filter split: {}".format(desired_filter_split))
    def ota4_char(g1, g3, c2, c4, gm):
        wo = np.sqrt(((g1 + gm) * g3) / (c2 * c4))
        Q = np.sqrt((g1 + gm) * g3 * c2 * c4) / (g3 * c2 + (g1 + g3)*c4)
        K = (g1*g3 - g1*gm) / (g1*g3 + g3*gm)
        return (wo, Q, K)

    for zpk in desired_filter_split:
        Q = - abs(zpk.P[0]) / (2 * np.real(zpk.P[0]))
        wo = abs(zpk.P[0])
        gm = 327e-6
        K = -1.5
        def loss(y):
            g1 = y[0]
            g3 = - (g1*gm) / ((K-1)*g1 + K*gm)
            wo_got, Q_got, K_got = ota4_char(g1, g3, y[1], y[1], gm)
            return np.sum(((wo_got - wo)/wo)**2 + ((Q_got - Q)/Q)**2 + ((K_got - K)/K)**2)
        res = scipy.optimize.minimize(loss, x0=[1/10e3, 200e-15], method='Nelder-Mead',
                   options={'maxfev': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        print("Aiming for Q = {}, wo = {} Hz, K = {}".format(Q, wo / (2 * math.pi), K))
        #print(res)
        g1 = res.x[0]
        g3 = - (g1*gm) / ((K-1)*g1 + K*gm)
        final_values = ota4_char(g1, g3, res.x[1], res.x[1], gm)
        print("Got Q = {}, wo = {} Hz, K = {}".format(final_values[1], final_values[0] / (2 * math.pi), final_values[2]))
        print("Got R1 = {}, R3 = {}, C2 = {}, C4 = {}".format(1/g1, 1/g3, res.x[1], res.x[1]))

    plt.show()
    sys.exit(1)
    """
    # Construct a topology to best match each stage
    for filter_stage in desired_filter_split:
        cascade = [OTA3(OTA3Values(ro=True))]
        circuit, subs_dict, noise_srcs = build_lpf(cascade)
        sym_tf = run_sym(circuit, 'V1')
        sym_gain = sym_tf['gain']

        sym_gain_lambda, lut = construct_lut(desired_filter, sym_gain)

    w, h = freqs(desired_filter.B, desired_filter.A)
    plt.figure()
    plt.semilogx(w / (2*np.pi), 20*np.log10(np.abs(h)), color='red', linewidth=5)
    for entry in lut:
        real_h = list(map(lambda x: sym_gain_lambda(
            C1_0=entry[0], C2_0=entry[1], R1_0=entry[2], G1_0=entry[3], RO_0=900e3, wbw=200e6*2*np.pi, s=1j*x), w))
        plt.semilogx(w / (2*np.pi), 20*np.log10(np.abs(real_h)), linewidth=2, label=str(entry))
    plt.semilogx(w / (2*np.pi), 20*np.log10(np.abs(real_h)))
    plt.legend()
    plt.show()
    """
    # Step 2: nonideality analysis
    lpf, subs, nsrcs = build_lpf([FT.OTA3], [ota3spec], ro=True, cl=False)
    tf = run_sym(lpf, 'V1', True)

    C1_0, C2_0, G1_0, R1_0, RO_0 = sp.symbols('C1_0 C2_0 G1_0 R1_0 RO_0')
    C, R, gm0, ro, wbw = sp.symbols('C R gm0 r_o wbw', real=True)
    s = sp.symbols('s')
    sym_gain = tf['gain'].subs({C1_0: C, C2_0: C, R1_0: R, G1_0: gm0 / (1 - (s/wbw)), RO_0: ro})

    nonideal_lut_header, nonideal_lut = construct_ota_lut()
    ro_idx = nonideal_lut_header.index("ro")
    wbw_idx = nonideal_lut_header.index("wbw")
    gm_idx = nonideal_lut_header.index("gm")
    Id_idx = nonideal_lut_header.index("idc")

    full_lut = []
    full_lut_header = nonideal_lut_header + ["R"] + ["C"]
    R_idx = full_lut_header.index("R")
    C_idx = full_lut_header.index("C")
    for nonideal_lut_line in nonideal_lut:
        sym_gain_subs = sym_gain.subs({ro: nonideal_lut_line[ro_idx], wbw: nonideal_lut_line[wbw_idx], gm0: nonideal_lut_line[gm_idx]})
        sym_gain_lambda = sp.lambdify([R, C, s], sym_gain_subs)

        w = freq_range(spec, 50)
        w, h = freqs(chosen_filter.B, chosen_filter.A, worN=w)

        def cost(y):
            passband_atten = 20*np.log10(abs(sym_gain_lambda(y[0], y[1], spec.passband_corner*1j)))
            stopband_atten = 20*np.log10(abs(sym_gain_lambda(y[0], y[1], spec.stopband_corner*1j)))

            actual_mag = list(map(lambda s: sym_gain_lambda(y[0], y[1], s), w*1j))
            gdelay = (-np.diff(np.unwrap(np.angle(actual_mag))) / np.diff(w))
            passband_idx = find_nearest_idx(w, spec.passband_corner)
            max_gdelay = np.max(gdelay[0:passband_idx])
            min_gdelay = np.min(gdelay[0:passband_idx])
            gdelay_variation = np.abs(max_gdelay - min_gdelay)

            return np.linalg.norm(actual_mag - h, 2)
            # TODO: figure out a better cost function which doesn't impose the original TF's strictness around
            # the exact passband and stopband corner + attenuation
            #return min(0, passband_atten + 3) + max(0, stopband_atten + spec.stopband_atten) + \
                   #min(0, gdelay_variation - spec.group_delay_variation) + np.linalg.norm(np.log10(actual_mag - h), 2)

        res = minimize(cost, x0=ideal_lut[70][0:2], method='Nelder-Mead',
                       options={'maxfev': 10000, 'xatol': 1e-3, 'fatol': 1e-12, 'adaptive': False})

        full_lut.append(np.append(nonideal_lut_line, res.x))

    def analyze_noise_power():
        for lut_line in full_lut:
            print("LUT Line: {}".format(lut_line))
            vi2 = 0
            kT4 = 4*k*ahkab.constants.Tref
            subs['R1_0'] = lut_line[R_idx]
            subs['C1_0'] = lut_line[C_idx]
            subs['C2_0'] = lut_line[C_idx]
            subs['G1_0'] = lut_line[gm_idx]
            subs['RO_0'] = lut_line[ro_idx]

            # Integrate input-referred noise power using symbolic analysis
            for s in nsrcs:
                if s.startswith('INR'):
                    in2 = kT4/subs[s[2:]]
                elif s.startswith('INE') or s.startswith('ING'):
                    in2 = kT4*nonideal_dict['gamma']*lut_line[gm_idx]
                tfn = run_sym(lpf, s)

                # Just get the input-referred noise density at DC and multiply by the passband to speed up calculation
                vni2 = sp.lambdify(f, sp.Abs(((tfn['gain']/tf['gain']).subs(subs_syms(tfn, subs))))**2 * in2)
                vi2 += quad(vni2, 1, spec.passband_corner.f())[0]

                # Input referred noise of 2nd stage (assuming same R and C)
                vni2_2 = sp.lambdify(f, sp.Abs(((tfn['gain']/tf['gain']**2).subs(subs_syms(tfn, subs))))**2 * in2)
                vi2 += quad(vni2, 1, spec.passband_corner.f())[0]

            print("\tTotal input referred noise power: {} V^2".format(vi2))

            # Assume allowable swing (zero-peak) is VDD/2 - V*
            dr = 10**(spec.dynamic_range/10)
            vi_min = sp.sqrt(vi2*dr*2)
            print('\tMin reqd voltage swing for DR: {} V'.format(vi_min))
            if vi_min > 0.2:
                print('\tFails dynamic range: voltage swing of {} V not attainable!'.format(vi_min))
            else:
                print('\tPasses dynamic range!')

            def power(Id):
                diff_factor = 2
                vdd = 1.2
                stages = 2
                branches = 2
                return diff_factor * Id * vdd * stages * branches

            p = power(lut_line[Id_idx])
            print("\tPower: {} W".format(p))
            yield(vi2, p)

    noise_power_data = list(analyze_noise_power())

    fig, ax1 = plt.subplots(figsize=figaspect(1/3))

    ax2 = ax1.twinx()
    ax1.plot([x[Id_idx] for x in full_lut], [x[0] for x in noise_power_data])
    ax2.plot([x[Id_idx] for x in full_lut], [x[1] for x in noise_power_data])

    ax1.set_xlabel('$I_{ds}$')
    ax1.set_ylabel('Noise Power [$V^2$]')
    ax2.set_ylabel('Estimated Static Power [W]')

    plt.savefig('figs/noise_power.pdf')
    if args.show_plots:
        plot_final_filter(rac, hs, spec)
