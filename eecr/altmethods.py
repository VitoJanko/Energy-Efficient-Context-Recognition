import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def creature_of_habit(contexts, transitions, sequence, setting_to_sequence, setting_to_energy,
                      quality_metric, alphas, max_setting):
    settings = list(setting_to_sequence.keys())
    order = _get_max(transitions, contexts)
    Q = _make_q(settings, contexts, max_setting, sequence, setting_to_sequence, quality_metric)
    cfgs = []
    for i in range(1, len(contexts)):
        #print(i)
        cfgs += [_assignment(i, a, contexts, Q, order, setting_to_energy) for a in alphas]
    return cfgs


def _get_max(transitions, activities):
    order = {}
    for (i, act) in enumerate(activities):
        temp = sorted(range(len(transitions)), key=lambda x: transitions[i][x])[::-1]
        order[act] = [activities[j] for j in temp]
    return order


def _transform_to_binary(sequence, activity):
    return sequence.apply(lambda x: "Same" if x == activity else "Other")


def _compare_binaries(setting, activity, max_setting, sequence, setting_to_sequence, quality_metric):
    seq1 = _transform_to_binary(setting_to_sequence[setting], activity)
    seq = _transform_to_binary(sequence, activity)
    seq2 = _transform_to_binary(setting_to_sequence[max_setting], activity)
    cf1 = confusion_matrix(seq, seq1, labels=["Same", "Other"])
    cf2 = confusion_matrix(seq, seq2, labels=["Same", "Other"])
    acc1 = quality_metric(cf1)
    acc2 = quality_metric(cf2)
    return acc2-acc1


def _make_q(settings, activities, max_setting, sequence, setting_to_sequence, quality_metric):
    df = pd.DataFrame()
    for act in activities:
        lst = pd.Series([_compare_binaries(s, act, max_setting, sequence, setting_to_sequence, quality_metric)
                         for s in settings], settings)
        df[act] = lst
    return df


def _assignment(p, alpha, contexts, Q, order, setting_to_energy):
    cfg = []
    for act in contexts:
        nxt_all = order[act][:p]
        cond = pd.Series(index=Q.index, data=[True]*len(Q))
        for nxt in nxt_all:
            cond = cond & (Q[nxt] < alpha)
        df = pd.DataFrame()
        df["values"] = Q[nxt_all[0]][cond]
        df["energy"] = pd.Series(df.index.values, df.index.values).apply(lambda x: setting_to_energy[x])
        df = df.sort_values(['energy', 'values'], ascending=[True, True])
        if len(df) == 0:
            res = Q[nxt_all[0]].idxmin()
        else:
            res = df.index.values[0]
        cfg.append(res)
    return cfg


def simple_solutions(contexts, sequence, setting_to_sequence, setting_to_energy, quality):
    config1 = _simple_a(contexts, sequence, setting_to_sequence, setting_to_energy)
    config2 = _simple_b(contexts, sequence, setting_to_sequence, setting_to_energy, quality)
    return [config1, config2]


def _simple_a(contexts, sequence, setting_to_sequence, setting_to_energy):
    config = []
    settings = list(setting_to_sequence.keys())
    for (i, act) in enumerate(contexts):
        max_acc = 0
        max_eng = max(setting_to_energy.values()) + 1000
        max_setting = settings[0]
        for setting in settings:
            cf = confusion_matrix(sequence, setting_to_sequence[setting], labels=contexts)
            if np.sum(np.array(cf), axis=0)[i] < len(sequence):
                if cf[i][i] > max_acc or (cf[i][i] == max_acc and setting_to_energy[setting] < max_eng):
                    max_acc = cf[i][i]
                    max_setting = setting
                    max_eng = setting_to_energy[setting]
        config.append(max_setting)
    return config


def _simple_b(contexts, sequence, setting_to_sequence, setting_to_energy, quality):
    config = []
    settings = list(setting_to_sequence.keys())
    for (i, act) in enumerate(contexts):
        max_acc = 0
        max_eng = max(setting_to_energy.values()) + 1000
        max_setting = settings[0]
        seq_true = _transform_to_binary(sequence, act)
        for setting in settings:
            seq_predicted = _transform_to_binary(setting_to_sequence[setting], act)
            cf = confusion_matrix(seq_true, seq_predicted, labels=["Same", "Other"])
            acc = quality(cf)
            if acc > max_acc or (acc == max_acc and setting_to_energy[setting] < max_eng):
                max_acc = acc
                max_setting = setting
                max_eng = setting_to_energy[setting]
        config.append(max_setting)
    return config


def episodic_sampling(sequence_true, sequence_predicted, energy_on, energy_off, performance, dec, inc, active=1):
    cfgs = []
    vals = []
    for d in dec:
        for i in inc:
            cfgs.append((i, d))
            vals.append(_episodic_tester(sequence_true, sequence_predicted,
                                         energy_on, energy_off, performance, d, i,
                                         active=1))
    return cfgs, vals


def _episodic_tester(sequence_true, sequence_predicted, energy_on, energy_off, performance, dec, inc, active=1):
    active_timer = active
    sleep_timer = -1
    sequence = []
    current_activity = sequence_true[0]
    energy = 0
    sleep_length = 0
    last_activity = sequence_true[0]
    for i in range(len(sequence_true)):
        if active_timer > 0:
            current_activity = sequence_predicted[i]
            sequence.append(current_activity)
            energy += energy_on
            active_timer -= 1
            if active_timer == 0:
                if last_activity == current_activity:
                    sleep_length += inc
                else:
                    sleep_length *= dec
                last_activity = current_activity
                sleep_timer = int(sleep_length)
                if sleep_timer == 0:
                    active_timer = active
        elif sleep_timer > 0:
            sequence.append(current_activity)
            energy += energy_off
            sleep_timer -= 1
            if sleep_timer == 0:
                active_timer = active
    cf = confusion_matrix(sequence_true, sequence)
    prf = performance(cf)
    eng = energy / float(len(sequence_true))
    return prf, eng
