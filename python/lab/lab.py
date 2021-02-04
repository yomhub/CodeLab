import sys
import struct
from codecs import decode

def rec(v):
    if(v<=1):
        return 0
    elif(v%2==0):
        return rec(v+1)+v
    else:
        return rec(v-3)-v

def bin_to_float32(b:str):
    """ Convert binary string to a float based on 32bit ieee 754. """
    f = int('01000001101011000111101011100001', 2)
    return struct.unpack('f', struct.pack('I', f))[0]

def bin_to_float(b:str):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)

if __name__ == "__main__":
    print(rec(10))
    # print(float(-0b10111))
    # print(bin_to_float("10010"))
    print('end')
    pass