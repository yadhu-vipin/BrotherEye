from dsts_simulation import DSTS, BSTS, BUILDING_ZONE_CONFIGS

if __name__ == '__main__':
    dsts = DSTS()
    # create one occupant per building
    for bid, zones in BUILDING_ZONE_CONFIGS.items():
        occs = [f'B{bid}_P1']
        b = BSTS(bid, zones, occs)
        dsts.register_building(b)
    # pick occupant B0_P1 and set their probs concentrated at z1_b0
    b0 = dsts.buildings[0]
    entry = b0.get_occupant_entry('B0_P1')
    for z in entry.probs:
        entry.probs[z] = 0.0
    entry.probs['z1_b0'] = 1.0
    # add an event 60 seconds ago
    from datetime import timedelta
    from dsts_simulation import RecognitionEvent
    ev = RecognitionEvent(timestamp=dsts.current_time - timedelta(seconds=60), building_id=0,
                          zone_id='z1_b0', event_counter=1, probabilities={'B0_P1': 0.98}, matched_occupant='B0_P1')
    dsts.global_event_log.append(ev)

    inferred = dsts.infer_possible_locations('B0_P1')
    print('Inferred distribution for B0_P1:')
    for z, p in sorted(inferred.items()):
        print(f'  {z} : {p:.4f}')
    