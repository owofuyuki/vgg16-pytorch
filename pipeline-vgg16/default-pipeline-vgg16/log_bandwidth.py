from time import time, sleep
from datetime import datetime
import sys

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
# print(type(sys.argv[1]))

# arg = sys.argv[1]
rx_before = 0
tx_before = 0
cnt = 0
while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)
    lines = open("/proc/net/dev", "r").readlines()

    # columnLine = lines[1]
    # _, receiveCols , transmitCols = columnLine.split("|")
    # receiveCols = map(lambda a:"recv_"+a, receiveCols.split())
    # transmitCols = map(lambda a:"trans_"+a, transmitCols.split())

    # cols = receiveCols+transmitCols

    # faces = {}
    # hieu_faces = {}
    # for line in lines[2:]:
    #     print(line)
    #     if line.find(":") < 0: continue
    #     face, data = line.split(":")
    #     faceData = dict(zip(cols, data.split()))
    #     faces[face] = faceData
    #     # if face == 'enp3s0':
    #     if face == arg:
    #         hieu_faces[face] = faces[face]
    # print(lines[3].strip())
    rx_after = int(lines[4].strip().replace(":","").split()[1])
    tx_after = int(lines[4].strip().replace(":","").split()[9])
        

    # import pprint
    # pprint.pprint(hieu_faces)

    # rx_after = int(hieu_faces['eth0']['recv_bytes'])
    # print(rx_after - rx_before, " bytes/s")
    line = f"{str(current_time)} RX:{str(rx_after - rx_before)} TX:{str(tx_after - tx_before)}"
    # print(line)
    if cnt == 1:
        with open (str(sys.argv[1]), "a") as f:
            f.write(line)
            f.write('\n')
            f.close()
    rx_before = int(lines[4].strip().replace(":","").split()[1])
    tx_before = int(lines[4].strip().replace(":","").split()[9])
    cnt = 1
    sleep(1)