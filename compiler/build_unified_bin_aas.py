
from io import SEEK_END
import re                               # parse data
import os                               # make directory
import argparse                         # arguments parser
import struct                           # convert to binary data
import datetime as dt                   # timestamp


def check_align(size):
    ##################################################
    # Add to use binary in the same form as AI-analog
    ##################################################
    reminder = size % 128
    if reminder == 0:
        return size, 0
    else:
        return size + 128 - reminder, 128-reminder


def ParseData(path, print_log, output_path):
    CLASSIFIER = False
    SSD = False
    YOLO = False
    ##################################################
    # Figure out model by check prior box header file
    ##################################################
    check_for_yolo = open(path + '/' + 'oacts_tbl.h', 'rt')
    dataString = check_for_yolo.read()
    yolo_use = re.findall('int yolo_use = (.*?);',dataString)
    yolo_use = int(yolo_use[0])
    check_for_yolo.close()

    if os.path.exists(path + '/' + 'quantized_prior_box.h'):
        SSD = True
    elif yolo_use:
        YOLO = True
    else:
        CLASSIFIER = True
    if print_log :
        if SSD:
            print("** SSD Model **")
        elif YOLO:
            print("** YOLO Model **")
        elif CLASSIFIER:
            print("** CLASSIFIER Model **")
        else:
            print("Error")
            exit(1)

    ##################################################
    # Open output bin and set
    ##################################################
    bin_out = open(output_path, 'wb')
    WriteBytes = 0                      # for count written bytes

    ##################################################
    # Write model operation (4B)
    ##################################################
    if SSD:             # Object detection model (SSD)
        bindata = struct.pack('I', 0)
    elif CLASSIFIER:    # classification model
        bindata = struct.pack('I', 1)
    else:               # Object detection model (YOLO)
        bindata = struct.pack('I', 2)
    bin_out.write(bindata)
    WriteBytes += 4

    ##################################################
    # Stamp build time (4B)
    ##################################################

    EncodedTime = (dt.datetime.now().year % 100) * 60 * 60 * 24 * 31 * 12
    EncodedTime += (dt.datetime.now().month - 1) * 60 * 60 * 24 * 31
    EncodedTime += (dt.datetime.now().day - 1) * 60 * 60 * 24
    EncodedTime += dt.datetime.now().hour * 60 * 60
    EncodedTime += dt.datetime.now().minute * 60
    EncodedTime += dt.datetime.now().second

    if print_log :
        print("model build at " + str(dt.datetime.now()))

    bindata = struct.pack('I', EncodedTime)

    bin_out.write(bindata)
    WriteBytes += 4

    ##################################################
    # Dump network_buf.h (5 * 4 = 20B)
    ##################################################
    NetworkBuf = open(path + '/' + 'network_buf.h', 'rt')

    dataString = NetworkBuf.read()
    NetworkSizes = re.findall('_size = (.*?);',dataString)
    for i, bytedata in enumerate(NetworkSizes):
        bindata = struct.pack('I', int(bytedata,10))
        bin_out.write(bindata)
        WriteBytes += 4

    NetworkBuf.close()

    if print_log :
        print("network variable : ", NetworkSizes)


    ##################################################
    # Dump oacts_tbl.h (2 or 1 * 4 = 8B or 4B)
    ##################################################
    OactsTbl = open(path + '/' + 'oacts_tbl.h', 'rt')
    dataString = OactsTbl.read()

    if YOLO or CLASSIFIER:
        # width , height infromation
        width = re.findall('int width = (.*?);',dataString)
        for i, bytedata in enumerate(width):
            bindata = struct.pack('i', int(bytedata, 10))
            bin_out.write(bindata)
            WriteBytes +=4
        height = re.findall('int height = (.*?);',dataString)
        for i, bytedata in enumerate(height):
            bindata = struct.pack('i', int(bytedata, 10))
            bin_out.write(bindata)
            WriteBytes +=4
        if print_log:
            print(f"width: {width} height: {height}")
    
    # Check Size for yolo
    Size = re.findall('[.]size = (.*?),',dataString)

    if YOLO or CLASSIFIER:
        # How many output 
        bindata = struct.pack('I', len(Size))
        bin_out.write(bindata)
        WriteBytes += 4     
    Size_D = dict()

    if YOLO:
        for i in range(0, len(Size)):
            Size_D[i] = int(Size[i])
        Size_D = sorted(Size_D.items(), key=lambda x: x[1], reverse=True)

    OffsetAddr = re.findall('[.]buf = [(]void[*][)](.*?),',dataString)
    if YOLO:
        a = list()
        for index in dict(Size_D).keys():
            bindata = struct.pack('I', int(OffsetAddr[index],16))
            a.append(OffsetAddr[index])
            bin_out.write(bindata)
            WriteBytes += 4      
        if print_log :
            print("oacts_tbl variable : ", a)    
    else:
        for i, bytedata in enumerate(OffsetAddr):
            bindata = struct.pack('I', int(bytedata,16))
            bin_out.write(bindata)
            WriteBytes += 4
        if print_log :
            print("oacts_tbl variable : ", OffsetAddr) 
    OactsTbl.close()

    ##################################################
    # Dump post_process.c ((12 + 256 + 256)*4 or 1 * 4 = 2096B or 4B)
    ##################################################
    PostProcess = open(path + '/' + 'post_process.c', 'rt')

    dataString = PostProcess.read()
    PostProcessEnum = {"\n":"","LOGISTIC_NONE":"0","LOGISTIC_SIGMOID":"1","LOGISTIC_SOFTMAX":"2","NMS_DEFAULT":"0","NMS_FAST":"1"}
    for i, data in PostProcessEnum.items():
        dataString = dataString.replace(i,data)

    if SSD:
        PostProcessVarDict = {}
        PP_ParsingList = ['bg_in_class','logistic','nms_method','prior_scl','score_scl','loc_scl','num_class','th_conf','th_iou','num_box','img_h','img_w']
        RunPostProcessFunc = str(re.findall('uint8_t[*] prior_box_addr(.*?)}',dataString))

        for i, ParseStr in enumerate(PP_ParsingList):
            PostProcessVarDict[ParseStr] = re.findall(ParseStr + '.*?= (.*?);',RunPostProcessFunc)[0]
        for i, data in PostProcessVarDict.items():
            bindata = struct.pack('i', int(data,10))
            bin_out.write(bindata)
            WriteBytes += 4
            if print_log :
                print( i, " : ", data)

        ScrLogTable = re.findall('ssd_scr_(.*?)};',dataString)
        ScrLogElem = re.findall('0x(.*?),',str(ScrLogTable))
        for i, uintdata in enumerate(ScrLogElem):
            bindata = struct.pack('I', int(uintdata,16))
            bin_out.write(bindata)
            WriteBytes += 4

        LocExpTable = re.findall('ssd_loc_(.*?)};',dataString)
        LocExpElem = re.findall('0x(.*?),',str(LocExpTable))
        for i, uintdata in enumerate(LocExpElem):
            bindata = struct.pack('I', int(uintdata,16))
            bin_out.write(bindata)
            WriteBytes += 4


    else:   # classifier model & yolo model
        NumClass = re.findall('num_class = (.*?);',dataString)
        for i, bytedata in enumerate(NumClass):
            bindata = struct.pack('i', int(bytedata,10))
            bin_out.write(bindata)
            WriteBytes += 4
        if print_log :
            print("Num class : ", NumClass)

    PostProcess.close()


    ##################################################
    # Merge cmd and quantized box
    ##################################################
    bin_cmd = open(path + '/' + "npu_cmd.bin", 'rb')
    bin_quant = open(path + '/' + "quantized_network.bin", 'rb')
    cmd_data = bin_cmd.read()
    quant_data = bin_quant.read()

    cmd_size = bin_cmd.seek(0,SEEK_END)
    quant_size = bin_quant.seek(0,SEEK_END)

    cmd_size, cmd_add_data = check_align(cmd_size)
    quant_size, quant_add_data = check_align(quant_size)

    bin_out.write(struct.pack('I',cmd_size))        # size of cmd
    WriteBytes += 4
    bin_out.write(struct.pack('I',quant_size))      # size of quant
    WriteBytes += 4

    bin_cmd.close()
    bin_quant.close()

    # for 128 align in yolo model dummy
    iter_num = (128 % (32 + WriteBytes))/4
    bindata = struct.pack('I', 0)
    for i in range(int(iter_num)):
        bin_out.write(bindata)
        WriteBytes += 4


    ##################################################
    # Dump quantized_prior_box.h
    ##################################################
    # classifier has no prior_box.h

    if SSD:
        QuantizedPriorBox = open(path + '/' + 'quantized_prior_box.h', 'rt')

        dataString = QuantizedPriorBox.read()
        priorboxes = re.findall('0x(.*?),',dataString)
        # *** IMPORTANT!!! : data of prior boxes are dumping after cmd, quant offset, size 
        prior_size = len(priorboxes)
        bin_out.write(struct.pack('I',prior_size))  # size of prior
        WriteBytes += 4

        QuantizedPriorBox.close()


    ##################################################
    # Open output bin and set
    ##################################################
    bin_out.write(cmd_data)
    print(cmd_add_data)
    if cmd_add_data >0:
        for i in range(int(cmd_add_data/4)):
            bin_out.write(struct.pack('I',0))
    bin_out.write(quant_data)
    print(quant_add_data)
    if quant_add_data >0:
        for i in range(int(quant_add_data/4)):
            bin_out.write(struct.pack('I',0))
    if SSD:
        for i, bytedata in enumerate(priorboxes):
            bindata = struct.pack('B', int(bytedata,16))    # B : unsigned short
            bin_out.write(bindata)
    if SSD:
        total_size = WriteBytes + cmd_size + quant_size + prior_size
    else:
        total_size = WriteBytes + cmd_size + quant_size
    
    size, add_data = check_align(total_size)
    if add_data >0:
        for i in range(int(add_data/4)):
            bin_out.write(struct.pack('I',0))
    bin_out.close()


def parse_args():
    parser = argparse.ArgumentParser(description= 'For AAS (EN675 NPU)')
    parser.add_argument('-p', '--path', type=str,
        default='.', metavar='network files directory',
        help='directory of network files, default : .')

    parser.add_argument('-l', '--log', type=bool,
        default=True, metavar='log model data',
        help='Print model data, default : True')

    parser.add_argument('-o', '--output', type=str,
        metavar='result file directory',
        default='./modeloutput.bin', help='directory of result file, default : ./modeloutput.bin')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    print_log = args.log
    output_path = args.output
    ParseData(path, print_log, output_path)

