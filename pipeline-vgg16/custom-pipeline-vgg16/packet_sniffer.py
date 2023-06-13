import socket, sys
from struct import *
import csv
from datetime import datetime
import argparse

try:
    s = socket.socket( socket.AF_PACKET , socket.SOCK_RAW , socket.ntohs(0x0003))
except socket.error:
    print('Socket could not be created!')
    sys.exit()

parser = argparse.ArgumentParser(
        description="Packet sniffer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
parser.add_argument('--filename', type=str, help="""File name to write csv file""",required=True)
parser.add_argument('--ip1', type=str, help="""IP Address 1""",required=True)
parser.add_argument('--ip2', type=str, help="""IP Address 2""",required=True)
parser.add_argument('--ip3', type=str, help="""IP Address 3""",required=True)

args = parser.parse_args()
all_ip = [args.ip1, args.ip2, args.ip3]

with open(f'CIFAR_log/{args.filename}.csv', 'w') as f:
    while True:
        writer = csv.writer(f)
        packet = s.recvfrom(65565)
        packet = packet[0]
        eth_length = 14
        eth_header = packet[:eth_length]
        eth = unpack('!6s6sH' , eth_header)
        eth_protocol = socket.ntohs(eth[2])

        if eth_protocol == 8:
            # print('\n')
            ip_header = packet[eth_length:20+eth_length]
            iph = unpack('!BBHHHBBH4s4s' , ip_header)                                                           
            version_ihl = iph[0]
            version = version_ihl >> 4
            ihl = version_ihl & 0xF
            iph_length = ihl * 4
            ttl = iph[5]
            protocol = iph[6]
            s_addr = socket.inet_ntoa(iph[8])
            d_addr = socket.inet_ntoa(iph[9])
            # print('Version: ' + str(version) + ' IP Header Length: ' + str(ihl) + ' TTL: ' + str(ttl) + ' Protocol: ' + str(protocol) + ' Source Address: ' + str(s_addr) + ' Destination Address: ' + str(d_addr))
            if (s_addr in all_ip) and (d_addr in all_ip) and (s_addr != d_addr):
                if protocol == 6 :
                    # print('Protocol: TCP')
                    t = iph_length + eth_length
                    tcp_header = packet[t:t+20]
                    tcph = unpack('!HHLLBBHHH' , tcp_header)
                    source_port = tcph[0]
                    dest_port = tcph[1]
                    sequence = tcph[2]
                    acknowledgement = tcph[3]
                    doff_reserved = tcph[4]
                    tcph_length = doff_reserved >> 4
                    # print('Source Port : ' + str(source_port) + ' Dest Port : ' + str(dest_port) + ' Sequence Number : ' + str(sequence) + ' Acknowledgement : ' + str(acknowledgement) + ' TCP header length : ' + str(tcph_length))
                    h_size = eth_length + iph_length + tcph_length * 4
                    data_size = len(packet) - h_size
                    # data = packet[h_size:]
                    # print('Data size:',data_size, ' Total size:', len(packet))
                    # print('Data', str(data))
                    if data_size != 0:
                        row = [datetime.now(),protocol,s_addr,d_addr,len(packet)]
                        writer.writerow(row)
                    
                """
                else:
                    row = [datetime.now(),protocol,s_addr,d_addr,0]
                    writer.writerow(row)
                    """
        f.close()