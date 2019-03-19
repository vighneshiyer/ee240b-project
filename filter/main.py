from filter.tf_design import FilterType, design_lpf, plot_filters_gain, plot_filters_group_delay, group_delay_variation, \
    freq_range, find_nearest_idx
from filter.lut_construction import construct_ideal_lut, construct_ota_lut
from filter.topology_analysis import *
from scipy.signal import freqs
from scipy.integrate import quad
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
