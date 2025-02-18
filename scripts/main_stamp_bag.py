#! /usr/bin/env python3
# coding:utf‐8

"""The msg format given by Nooploop does not contain a timestamp, and this script is used to generate a UWB data bag with a timestamp"""

import rosbag
from uwb_loc.msg import MyUWB

SRC_BAG_PATH = '/your_path/record.bag'
bag_r = rosbag.Bag(SRC_BAG_PATH, 'r')

DEST_BAG_PATH = '/your_path/record_new.bag'
bag_w = rosbag.Bag(DEST_BAG_PATH, 'w')

generator = bag_r.read_messages('/nlink_linktrack_tagframe0')

for msg_r in generator:
    print(msg_r)

    msg_w = MyUWB()
    msg_w.stamp = msg_r.timestamp.to_sec()
    msg_w.dis_arr = msg_r.message.dis_arr

    bag_w.write('/uwb', msg_w, msg_r.timestamp)

bag_r.close()
bag_w.close()
