def MSB(model, pt, ct, key, state, bnum):
    label_length = 2
    label = [0]*label_length
    key_arr = [0]*16
    key_arr[bnum] = key
    leak = model.modelobj.leakage(pt, ct, key_arr, bnum)
    if leak < 128:
        label[0] = 1
        return label
    else:
        label[1] = 1
        return label    

def LSB(model, pt, ct, key, state, bnum):
    label_length = 2
    label = [0]*label_length
    key_arr = [0]*16
    key_arr[bnum] = key
    leak = model.modelobj.leakage(pt, ct, key_arr, bnum)
    if leak % 2 == 0:
        label[0] = 1
        return label
    else:
        label[1] = 1
        return label

def HW(model, pt, ct, key, state, bnum):
    label_length = 2
    label = [0]*label_length
    hyp = model.leakage(pt, ct, key, bnum, state)
    if hyp < 4:
        label[0] = 1
        return label
    else:
        label[1] = 1
        return label

def raw_HW(model, pt, ct, key, state, bnum):
    label_length = 9
    label = [0]*label_length
    hyp = model.leakage(pt, ct, key, bnum, state)
    label[hyp] = 1
    return label