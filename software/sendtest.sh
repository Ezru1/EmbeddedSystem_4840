#!/bin/sh
##########################################
#Sends software files back to workstation#
##########################################

set -x
date
scp test.txt asy2126@micro11.ee.columbia.edu:Embedded/CourseProject/Zylo/software/test.txt
echo -ne '\007'
