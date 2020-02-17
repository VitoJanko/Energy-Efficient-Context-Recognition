from sklearn.metrics import confusion_matrix


def test_sca_dca(sequence_true, sequences, lengths, energies, energy_off,
                 performance, active=1, energy_sequences=None):
    active_timer = active
    sleep_timer = -1
    sequence = []
    current_activity = sequence_true[0]
    energy = 0
    for i in range(len(sequence_true)):
        if active_timer > 0:
            current_activity = sequences[current_activity][i]
            sequence.append(current_activity)
            if energy_sequences is None:
                energy += energies[current_activity]
            else:
                energy += energy_sequences[current_activity][i]
            active_timer -= 1
            if active_timer == 0:
                sleep_timer = lengths[current_activity] - 1
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
