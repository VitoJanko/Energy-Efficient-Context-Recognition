def test_sca_dca(sequence_true, sequences, lengths, energies, energy_off, performance, active=1, energy_sequences=None):
    # print lengths
    # print energies
    # print active
    # print energy_off
    # print [len(x) for x in SHL_configuration]
    # active = max(0,active-1)
    active_timer = active
    sleep_timer = -1
    sequence = []
    current_activity = sequence_true[0]
    energy = 0
    for i in range(len(sequence_true)):
        # print "Step 1"
        # print active_timer, sleep_timer
        if active_timer > 0:
            current_activity = sequences[current_activity][i]
            sequence.append(current_activity)
            if energy_sequences is None:
                energy += energies[current_activity]
            else:
                energy += energy_sequences[current_activity][i]
            # print "append 1"
            active_timer -= 1
            if active_timer == 0:
                sleep_timer = lengths[current_activity] - 1
                if sleep_timer == 0:
                    active_timer = active


        # print "Step 2"
        # print active_timer, sleep_timer
        elif sleep_timer > 0:
            sequence.append(current_activity)
            energy += energy_off
            sleep_timer -= 1
            if sleep_timer == 0:
                active_timer = active
        # print "Step 3"
        # print active_timer, sleep_timer

    # print sequence
    cf = confusion_matrix(sequence_true, sequence)
    prf = performance(cf)
    eng = energy / float(len(sequence_true))

    return prf, eng  # cf