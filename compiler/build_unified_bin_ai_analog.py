
import os                               # make directory
import argparse                         # arguments parser
import struct                           # convert to binary data

"""
structure

32 byte : NpuDataHeaderInfrom_t

96 + 128 align byte : model 1(128 align)

96 + 128 aling byte : model 2(128 align)

...
"""

def ParseData(input, output_path):
    ##################################################
    # Figure out It is binary file
    ##################################################
    for i in input:
        if i[-3:] == 'bin':
            pass
        else:
            raise Exception("It is not a bin file")
        
    model_num = len(input)
    
    ##################################################
    # Open output bin and set
    ##################################################
    bin_out = open(output_path, 'wb')
    WriteBytes = 0                      # for count written bytes

    ##################################################
    # Write model num (4B)
    ##################################################

    bindata = struct.pack('I', model_num)
    bin_out.write(bindata)
    WriteBytes += 4

    ##################################################
    # Write model Size (4B)
    ##################################################
    bindata = struct.pack('I', 0)
    bin_out.write(bindata)
    WriteBytes += 4
    for i in input:
        size = os.path.getsize(i)
        print(size)
        bindata = struct.pack('I', size)
        bin_out.write(bindata)
        WriteBytes += 4
    
    while(WriteBytes < 32): # for 128 align
        bindata = struct.pack('I', 0)
        bin_out.write(bindata)
        WriteBytes += 4     
    

    ##################################################
    # Write model binary (4B)
    ##################################################
    for i in input:
        bin_in = open(i, 'rb')
        bin_data = bin_in.read()
        bin_in.close()
        bin_out.write(bin_data)


    ##################################################
    # Write test image (512x512x3B)
    ##################################################
    #while(WriteBytes < 128): # for 128 align
    #    bindata = struct.pack('I', 0)
    #    bin_out.write(bindata)
    #    WriteBytes += 4 
    #bin_in = open("input.ia.bin", 'rb')
    #bin_data = bin_in.read()
    #bin_in.close()
    #bin_out.write(bin_data)

    bin_out.close()


def parse_args():
    parser = argparse.ArgumentParser(description= 'Make Unified binary for AI-Analog')
    parser.add_argument('-i', '--input', type=str,
        default=None, nargs='*',
        help='input binary')

    parser.add_argument('-o', '--output', type=str,
        metavar='result file directory',
        default='./modeloutput.bin', help='directory of result file, default : ./modeloutput.bin')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input = args.input
    output_path = args.output
    ParseData(input, output_path)

