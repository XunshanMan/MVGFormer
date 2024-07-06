
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def str_to_number(x):

    # int or float
    out = float(x) if '.' in x else int(x)

    return out

def judge_float_or_int(x):
    if x.isnumeric():
        return True
    if x.count('.') == 1 and x.replace('.','').isdigit():
        return True
    return False

def parse_value(value):
    if value == 'False' or value == 'false':
        value = False
    elif value == 'True' or value == 'true':
        value = True
    # deal with int type
    elif judge_float_or_int(value):
        value = str_to_number(value)
    elif '[' in value:
        # deal with list
        value = value.replace('[','')
        value = value.replace(']','')
        values = value.split(',')
        value = [ str_to_number(v) for v in values ]
    return value

def parse_string_to_keyvalue(args):
    key_value = []
    for s in args:
        if '=' in s:
            key_and_value = s.split('=')
            key = key_and_value[0]
            value = key_and_value[1]
            # parse value, e.g., list
            value = parse_value(value)

            # split key into dict 
            key_sep = key.split('.')
            if len(key_sep) == 2:
                value = {key_sep[1]: value}
                kv_pair = {key_sep[0] : value}
            elif len(key_sep) == 1:
                kv_pair = {key_sep[0] : value}
            else:
                raise NotImplementedError('Only support one layer parameter.')
            # key_value[ss[0]] = ss[1]
            key_value.append(kv_pair)
    return key_value

if __name__ == '__main__':
    test_str = ['THIS.THAT=[1,3,9]', 'BOOL=True']
    output = parse_string_to_keyvalue(test_str)
    output_check = [
        {'THIS' : {
            'THAT' : [1,3,9]
        }},
        {'BOOL' : True}
    ]
    assert(output_check == output)
    print(output)

    test_str = ['THIS.THAT=[1,3.1,9]']
    output = parse_string_to_keyvalue(test_str)
    output_check = [
        {'THIS' : {
            'THAT' : [1,3.1,9]
        }}
    ]
    assert(output_check == output)
    print(output)

    test_str = ['THIS.THAT=2.1']
    output_check = [
        {'THIS' : {
            'THAT' : 2.1
        }}
    ]
    output = parse_string_to_keyvalue(test_str)
    assert(output_check == output)
    print(output)

    test_str = ['THIS.THAT=ASTRING']
    output = parse_string_to_keyvalue(test_str)
    output_check = [
        {'THIS' : {
            'THAT' : 'ASTRING'
        }}
    ]
    assert(output_check == output)
    print(output)

    print('All unit tests pass.')