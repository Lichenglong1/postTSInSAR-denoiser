#! /usr/bin/env python
###############################################################################
#  gdal2roipac.py
#
#  Project:  Seamless SAR Archive
#  Purpose:  Convert any format supported by GDAL to ROI_PAC format
#  Author:   Scott Baker
#  Created:  August 2013
#
###############################################################################
#  Copyright (c) 2013, Scott Baker 
# 
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
###############################################################################

import os
import sys
from osgeo import gdal

if len(sys.argv)<3:
    print "USAGE: %s IN_FILE OUT_FILE" % sys.argv[0]
    print "\nExample: %s dem.tif roipac.dem" % sys.argv[0]
    exit()

workdir = os.getcwd()
input_name = workdir+'/'+sys.argv[1]
output_name = workdir+'/'+sys.argv[2]
#######################################
##### CREATE A ROI_PAC FORMAT DEM #####
#######################################
indataset = gdal.Open(input_name)
data = indataset.ReadAsArray()
data.flatten()
data.tofile(output_name)
adfGeoTransform = indataset.GetGeoTransform(can_return_null = True)
x_first = '%.12f' % adfGeoTransform[0]
x_step = '%.12f' % adfGeoTransform[1]
y_first = '%.12f' % adfGeoTransform[3]
y_step = '%.12f' % adfGeoTransform[5]
with open(output_name+'.rsc','w') as RSC:
    RSC.write('WIDTH          '+str(indataset.RasterXSize)+'\n')
    RSC.write('FILE_LENGTH    '+str(indataset.RasterYSize)+'\n')
    RSC.write('X_FIRST        '+x_first+'\n')
    RSC.write('Y_FIRST        '+y_first+'\n')
    RSC.write('X_STEP         '+x_step+'\n')
    RSC.write('Y_STEP         '+y_step+'\n')
    RSC.write('Z_SCALE        1\n')
    RSC.write('Z_OFFSET       0\n')
    RSC.write('X_UNIT         degrees\n')
    RSC.write('Y_UNIT         degrees\n')
    RSC.write('PROJECTION     LATLON')

### MAKE AN XML FILE FOR ISCE ###
with open(output_name+'.xml','w') as XML:
    XML.write('<imageFile>\n')
    XML.write('    <property name="BYTE_ORDER">\n')
    XML.write('        <value>l</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="DATA_TYPE">\n')
    XML.write('        <value>SHORT</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="IMAGE_TYPE">\n')
    XML.write('        <value>dem</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="REFERENCE">\n')
    XML.write('        <value>EGM96</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="WIDTH">\n')
    XML.write('        <value>'+str(indataset.RasterXSize)+'</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="LENGTH">\n')
    XML.write('        <value>'+str(indataset.RasterYSize)+'</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="FILE_NAME">\n')
    XML.write('        <value>'+output_name+'</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="DELTA_LONGITUDE">\n')
    XML.write('        <value>'+x_step+'</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="DELTA_LATITUDE">\n')
    XML.write('        <value>'+y_step+'</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="FIRST_LONGITUDE">\n')
    XML.write('        <value>'+x_first+'</value>\n')
    XML.write('    </property>\n')
    XML.write('    <property name="FIRST_LATITUDE">\n')
    XML.write('        <value>'+y_first+'</value>\n')
    XML.write('    </property>\n')
    XML.write('    <component name="Coordinate1">\n')
    XML.write('        <factorymodule>isceobj.Image</factorymodule>\n')
    XML.write('        <factoryname>createCoordinate</factoryname>\n')
    XML.write('        <doc>First coordinate of a 2D image (witdh).</doc>\n')
    XML.write('        <property name="startingValue">\n')
    XML.write('            <value>'+x_first+'</value>\n')
    XML.write('        </property>\n')
    XML.write('        <property name="delta">\n')
    XML.write('            <value>'+x_step+'</value>\n')
    XML.write("            <doc>{'doc': 'Coordinate quantization.'}</doc>\n")
    XML.write('            <units>{}</units>\n')
    XML.write('        </property>\n')
    XML.write('        <property name="size">\n')
    XML.write('            <value>'+str(indataset.RasterXSize)+'</value>\n')
    XML.write("            <doc>{'doc': 'Coordinate size.'}</doc>\n")
    XML.write('        </property>\n')
    XML.write('    </component>\n')
    XML.write('    <component name="Coordinate2">\n')
    XML.write('        <factorymodule>isceobj.Image</factorymodule>\n')
    XML.write('        <factoryname>createCoordinate</factoryname>\n')
    XML.write('        <doc>Second coordinate of a 2D image (length).</doc>\n')
    XML.write('        <property name="startingValue">\n')
    XML.write('            <value>'+y_first+'</value>\n')
    XML.write('        </property>\n')
    XML.write('        <property name="delta">\n')
    XML.write('            <value>'+y_step+'</value>\n')
    XML.write("            <doc>{'doc': 'Coordinate quantization.'}</doc>\n")
    XML.write('            <units>{}</units>\n')
    XML.write('        </property>\n')
    XML.write('        <property name="size">\n')
    XML.write('            <value>'+str(indataset.RasterYSize)+'</value>\n')
    XML.write("            <doc>{'doc': 'Coordinate size.'}</doc>\n")
    XML.write('        </property>\n')
    XML.write('    </component>\n')
    XML.write('</imageFile>')
