def make_initializer(name: str, initializer_type: str,
                     xml_info: dict, template_variables: dict):
	# data_shape will be changed associated with data pre-processing
	data_shape = template_variables["data_shape"]
	if initializer_type == "random_uniform":
		params = dict()
		params["shape"] = data_shape
		params["minval"] = float(xml_info[name+"_min"])
		params["maxval"] = float(xml_info[name+"_max"])

		template_variables[name+"_init_module"] = "tf.random_uniform"
		template_variables[name+"_params"] = params


def init_weight(xml_info: dict, template_variables: dict):
	make_initializer("weight", xml_info["weight_type"],
	                 xml_info, template_variables)
	make_initializer("bias", xml_info["bias_type"],
	                 xml_info, template_variables)

