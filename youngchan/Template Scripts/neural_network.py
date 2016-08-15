from jinja2 import Environment, FileSystemLoader
import xml.etree.ElementTree as et
import os
import sys
from code_generator import code_generator
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))


def bind_variables(xml_info: dict, template_variables: dict):
	code_generator.bind_common_variables(xml_info, template_variables)

	layer_size = int(xml_info['layer_set_size'])
	template_variables['layer_size'] = layer_size
	input_shape = []
	output_shape = []
	activ_functions = []
	for i in range(layer_size):
		activation_key = str(i + 1) + '_layer_activation'
		activ_functions.append(code_generator.make_activation_function(xml_info[activation_key]))

		input_key = str(i + 1) + '_layer_input'
		output_key = str(i + 1) + '_layer_output'
		input_shape.append(int(xml_info[input_key]))
		output_shape.append(int(xml_info[output_key]))
	template_variables["activation_functions"] = activ_functions
	template_variables['input_shape'] = input_shape
	template_variables['output_shape'] = output_shape


def make_code(root: et.Element):
	j2_env = Environment(loader=FileSystemLoader(PARENT_DIR),
	                     trim_blocks=True)
	try:
		template = j2_env.get_template("./Template Files/template_" + root.find("model").find("type").text + ".py")
	except:
		print("template error")
		sys.exit(2)

	xml_info = dict()
	code_generator.parse_xml("", root, root, xml_info)

	template_variables = dict()

	bind_variables(xml_info, template_variables)
	code_generator.process_data(xml_info, template_variables)
	code_generator.make_optimizer(xml_info, template_variables)
	code_generator.make_initializer(xml_info, template_variables)

	output_file = open(PARENT_DIR+"/Samples/output.py", "w")
	output_file.write(template.render(template_variables))
	output_file.close()

	print("Code is generated")
