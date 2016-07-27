from jinja2 import Environment, FileSystemLoader
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))


def parse_child(parent, node, template_variable):
	"""
	Fill template variavle dictionary recursively
	A format of key of text field is (parenttag_currenttag)
		ex) from
			<xval>
				<low>10</low>
			</xval>
			to
			template[xval_low] = 10
	and format of attribute filed is (parenttag_currenttag_attribute)
		ex) from
			<input>
				<xval type="float">
					...
				</xval>
			</input>
			to
			template[input_xval_type] = "float"

	:param parent:              parent node
	:param node:                current node
	:param template_variable:   template variable dictionary
	:return:                    void
	"""
	key = parent.tag + "_" + node.tag
	if parent != node:
		template_variable[key] = node.text
	for attr in node.attrib:
		template_variable[key + "_" + attr] = node.attrib[attr]
	for child in node:
		parse_child(node, child, template_variable)


def make_code(root):
	j2_env = Environment(loader=FileSystemLoader(PARENT_DIR),
	                     trim_blocks=True)
	template = j2_env.get_template("./Template Files/" + root.find("type").text + ".obj")
	try:
		template = j2_env.get_template("./Template Files/" + root.find("type").text + ".obj")
	except:
		print("template error")
		sys.exit(2)

	template_variables = dict()
	template_variables["x_data"] = "[1,2,3]"
	template_variables["y_data"] = "[1,2,3]"

	parse_child(root, root, template_variables)

	output_file = open(PARENT_DIR+"/Samples/output.py", "w")
	output_file.write(template.render(template_variables))
	output_file.close()

	print("Code is gernerated")

# name binding codes
"""
	for child in root:
		if child.tag == "initializer":
			initializer = get_name(child.get("type"))
			template_variables["initializer"] = initializer
		elif child.tag == "optimizer":
			optimizer = get_name(child.get("type"))

			learning_rate = 0.01
			template_variables["optimizer"] = optimizer
			template_variables["learning_rate"] = learning_rate
		elif child.tag == "training_epoch":
			template_variables["training_epoch"] = 1024
"""
"""
def get_name(string):
	if string == "gradient_descent":
		return "GradientDescentOptimizer"
	elif string == "random_uniform":
		return string
"""