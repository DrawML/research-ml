#-*- coding: utf-8 -*-
import sys
import os.path
import xml.etree.ElementTree as et
import numpy as np

NO_FILE_XML  = 101
NO_FILE_DATA = 102
NO_ARG_XML   = 103
NO_ARG_DATA  = 104
FORMAT_ERR   = 105
MODEL_TYPE   = ["linear_regression"]


def error_handler(error_code, msg=""):
	if error_code == NO_ARG_XML:
		print("No XML argument")
	elif error_code == NO_ARG_XML:
		print("No data argument")
	elif error_code == NO_FILE_XML:
		print("No XML file")
	elif error_code == NO_FILE_DATA:
		print("No data file")

	elif error_code == FORMAT_ERR:
		print("Format error : ", msg)
	sys.exit(2)


if __name__ == "__main__":
	argc = len(sys.argv)
	if argc < 2:
		error_handler(NO_ARG_XML)
	elif argc < 3:
		error_handler(NO_ARG_DATA)

	xml_file = sys.argv[1]
	data_file = sys.argv[2]
	if os.path.isfile(xml_file) is False:
		error_handler(NO_FILE_XML)
	elif os.path.isfile(data_file) is False:
		error_handler(NO_FILE_DATA)

	xml_tree = et.parse(xml_file)
	root = xml_tree.getroot()
	if root.tag != "model":
		error_handler(FORMAT_ERR, "model")

	exec("from " + root.find("type").text + " import make_code")
	make_code(root)
