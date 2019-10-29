import random


def initialisations(nb_doctors, case, n_args):
    '''
    We use four standard cases here, namely following cases:

    Case 1: One doctor has strong influence, stubbornness and higher belief in certain conclusion.
    Other two doctors uncertain about both conclusions and have weak influence and stubbornness.
    Expectation: committee reaches conclusion early, less number of argumentation steps. As decision
    is dominated by this doctor.

    Case 2: Two doctor strong influence and stubbornness and third uncertain doctor. Both doctors
    believe in opposite conclusion. Bipolar committee. Expectation: both try and influence the third doctor,
    decision from committee depends on how well they can influence the third doctor.

    Case 3: All three doctors with very weak influence and stubbornness and uncertain belief arrays towards
    both the conclusions. Expectation: more rounds of arguments needed to finally reach a conclusion.
    The increase should be significant from all the previous cases.

    Case 4: All three doctors with very strong influence and stubbornness and uncertain belief arrays
    towards both conclusion. Expectation: more rounds of arguments needed to finally reach a conclusion.
    Similar to case 3.

    Case 5: All three doctors with 0.5 stubbornness and influence. Uncertain belief arrays.
    Will be an interesting observation.

    :param nb_doctors: number of doctors in the committee
    :param case: which case would you like to simulate
    :param  n_args: number of arguments
    :return: a list of belief arrays and list of stubborn influence value for each doctor
    '''

    strong_influence_set = (0.8, 0.9, 0.95)
    strong_stubbornn_set = (0.8, 0.9, 0.95)
    weak_influence_set = (0.1, 0.2, 0.3)
    weak_stubbornn_set = (0.1, 0.2, 0.3)
    uncertain_belief_set = (0.4, 0.5, 0.6)
    strong_belief_set = (0.8, 0.9, 0.95)
    weak_belief_set = (0, 0.1, 0.2, 0.3)
    strong_belief_conclusion = [[0.90, 0.3, 0.90, 0.2, 0.4], [0.3, 0.90, 0.2, 0.80, 0.90]]

    belief_array_list = []
    influence_stubbornn_list = []

    try:
        converted_case = int(case)
    except ValueError:
        converted_case = case

    if converted_case == 1:
        '''
        Case 1: One doctor has strong influence, stubbornness and higher belief in certain conclusion.
        Other two doctors uncertain about both conclusions and have weak influence and stubbornness.
        Expectation: committee reaches conclusion early, less number of argumentation steps. As decision
        is dominated by this doctor.
        '''
        for doctor in range(nb_doctors):
            if doctor == 0:
                influence = random.choice(strong_influence_set)
                stubbornn = random.choice(strong_stubbornn_set)
                belief_array = random.choice(strong_belief_conclusion)
                influence_stubbornn_list.append((influence, stubbornn))
                belief_array_list.append(belief_array)
            else:
                influence = random.choice(weak_influence_set)
                stubbornn = random.choice(weak_stubbornn_set)
                belief_array = [random.choice(uncertain_belief_set) for x in range(n_args)]
                influence_stubbornn_list.append((influence, stubbornn))
                belief_array_list.append((belief_array))

        return (belief_array_list, influence_stubbornn_list)

    elif converted_case == 2:
        '''
        Case 2: Two doctor strong influence and stubbornness and third uncertain doctor. Both doctors
        believe in opposite conclusion. Bipolar committee. Expectation: both try and influence the third doctor,
        decision from committee depends on how well they can influence the third doctor.
        '''
        for doctor in range(nb_doctors):
            if doctor == 0:
                influence = random.choice(strong_influence_set)
                stubbornn = random.choice(strong_stubbornn_set)
                belief_array = strong_belief_conclusion[0]
                influence_stubbornn_list.append((influence, stubbornn))
                belief_array_list.append(belief_array)
            if doctor == 1:
                influence = random.choice(strong_influence_set)
                stubbornn = random.choice(strong_stubbornn_set)
                belief_array = strong_belief_conclusion[1]
                influence_stubbornn_list.append((influence, stubbornn))
                belief_array_list.append(belief_array)
            elif doctor == 2:
                influence = random.choice(weak_influence_set)
                stubbornn = random.choice(weak_stubbornn_set)
                belief_array = [random.choice(uncertain_belief_set) for x in range(n_args)]
                influence_stubbornn_list.append((influence, stubbornn))
                belief_array_list.append((belief_array))

        return (belief_array_list, influence_stubbornn_list)

    elif converted_case == 3:
        '''
        Case 3: All three doctors with very weak influence and stubbornness and uncertain belief arrays towards
        both the conclusions. Expectation: more rounds of arguments needed to finally reach a conclusion.
        The increase should be significant from all the previous cases.
        '''
        for doctor in range(nb_doctors):
            influence = random.choice(weak_influence_set)
            stubbornn = random.choice(weak_stubbornn_set)
            belief_array = [random.choice(uncertain_belief_set) for x in range(n_args)]
            influence_stubbornn_list.append((influence, stubbornn))
            belief_array_list.append((belief_array))

        return (belief_array_list, influence_stubbornn_list)

    elif converted_case == 4:
        '''
        Case 4: All three doctors with very strong influence and stubbornness and uncertain belief arrays
        towards both conclusion. Expectation: more rounds of arguments needed to finally reach a conclusion.
        Similar to case 3.
        '''
        for doctor in range(nb_doctors):
            influence = random.choice(strong_influence_set)
            stubbornn = random.choice(strong_stubbornn_set)
            belief_array = [random.choice(uncertain_belief_set) for x in range(n_args)]
            influence_stubbornn_list.append((influence, stubbornn))
            belief_array_list.append((belief_array))

        return (belief_array_list, influence_stubbornn_list)

    elif converted_case == 5:
        '''
        Case 5: All three doctors with 0.5 stubbornness and influence. Uncertain belief arrays.
        Will be an interesting observation.
        '''
        for doctor in range(nb_doctors):
            influence = 0.5
            stubbornn = 0.5
            belief_array = [random.choice(uncertain_belief_set) for x in range(n_args)]
            influence_stubbornn_list.append((influence, stubbornn))
            belief_array_list.append((belief_array))

        return (belief_array_list, influence_stubbornn_list)

    else:  # default case
        belief_array = [[0.75, 0.30, 0.80, 0.50, 0.50],
                        [0.80, 0.50, 0.70, 0.40, 0.50],
                        [0.40, 0.70, 0.55, 0.75, 0.98]]
        influence_stubbornn_list = [(0.5, 0.5), (0.5, 0.5), (0.75, 0.75)]
        return belief_array, influence_stubbornn_list
