import xml.etree.ElementTree as et


class code_generator:
	@staticmethod
	def bind_common_variables(xml_info: dict, template_variables: dict):
		template_variables["training_epoch"] = xml_info["model_training_epoch"]

		if xml_info["regularization_enable"] == "true":
			template_variables["reg_enable"] = True
			template_variables["reg_lambda"] = xml_info["regularization_lambda"]
		else:
			template_variables["reg_enable"] = False
			template_variables["reg_lambda"] = 0

	@staticmethod
	def process_data(xml_info: dict, template_variables: dict):
		"""
			data processing code...
		"""
		# template_variables["x_data"] = xml_info["x_data"]
		# template_variables["y_data"] = xml_info["y_data"]

	@staticmethod
	def make_initializer(xml_info: dict, template_variables: dict):
		initializer_type = xml_info["initializer_type"]

		if initializer_type == "random_uniform":
			params = dict()
			params["minval"] = float(xml_info["initializer_min"])
			params["maxval"] = float(xml_info["initializer_max"])

			template_variables["init_module"] = "tf.random_uniform"
			template_variables["init_params"] = params

	@staticmethod
	def make_optimizer(xml_info: dict, template_variables: dict):
		opt_name = xml_info["optimizer_type"]
		learning_rate = float(xml_info["optimizer_learning_rate"])

		if opt_name == "gradient_descent":
			params = dict()
			params["learning_rate"] = learning_rate
			if "optimizer_use_locking" in xml_info:
				params["use_locking"] = xml_info["optimizer_use_locking"]
			if "optimizer_name" in xml_info:
				params["optimizer_name"] = xml_info["optimizer_name"]
			template_variables["optimizer_module"] = "tf.train"
			template_variables["optimizer_name"] = "'GradientDescentOptimizer'"
			template_variables["optimizer_params"] = params

		elif opt_name == "adadelta":
			params = dict()
			params["learning_rate"] = learning_rate
			if "optimizer_rho" in xml_info:
				params["rho"] = xml_info["optimizer_rho"]
			if "optimizer_epsilon" in xml_info:
				params["epsilon"] = xml_info["optimizer_epsilon"]
			if "optimizer_use_locking" in xml_info:
				params["use_locking"] = xml_info["optimizer_use_locking"]
			if "optimizer_name" in xml_info:
				params["name"] = xml_info["optimizer_name"]
			template_variables["optimizer_module"] = "tf.train"
			template_variables["optimizer_name"] = "'AdadeltaOptimizer'"
			template_variables["optimizer_params"] = params

		elif opt_name == "adagrad":
			params = dict()
			params["learning_rate"] = learning_rate
			if "optimizer_initial_accumulator_value" in xml_info:
				params["initial_accumulator_value"] = xml_info["optimizer_initial_accumulator_value"]
			if "optimizer_use_locking" in xml_info:
				params["use_locking"] = xml_info["optimizer_use_locking"]
			if "optimizer_name" in xml_info:
				params["name"] = xml_info["optimizer_name"]
			template_variables["optimizer_module"] = "tf.train"
			template_variables["optimizer_name"] = "'AdagradOptimizer'"
			template_variables["optimizer_params"] = params

		elif opt_name == "momentum":
			params = dict()
			params["learning_rate"] = learning_rate
			if "optimizer_use_locking" in xml_info:
				params["use_locking"] = xml_info["optimizer_use_locking"]
			if "optimizer_name" in xml_info:
				params["name"] = xml_info["optimizer_name"]
			template_variables["optimizer_module"] = "tf.train"
			template_variables["optimizer_name"] = "'MomentumOptimizer'"
			template_variables["optimizer_params"] = params

		elif opt_name == "adam":
			params = dict()
			params["learning_rate"] = learning_rate
			if "optimizer_beta1" in xml_info:
				params["beta1"] = xml_info["optimizer_beta1"]
			if "optimizer_beta2" in xml_info:
				params["beta2"] = xml_info["optimizer_beta2"]
			if "optimizer_epsilon" in xml_info:
				params["epsilon"] = xml_info["optimizer_epsilon"]
			if "optimizer_use_locking" in xml_info:
				params["use_locking"] = xml_info["optimizer_use_locking"]
			if "optimizer_name" in xml_info:
				params["name"] = xml_info["optimizer_name"]
			template_variables["optimizer_module"] = "tf.train"
			template_variables["optimizer_name"] = "'AdadeltaOptimizer'"
			template_variables["optimizer_params"] = params

		elif opt_name == "ftrl":
			params = dict()
			params["learning_rate"] = learning_rate
			if "optimizer_learning_rate_power" in xml_info:
				params["learning_rate_power"] = xml_info["optimizer_learning_rate_power"]
			if "optimizer_initial_accumulator_value" in xml_info:
				params["initial_accumulator_value"] = xml_info["optimizer_initial_accumulator_value"]
			if "optimizer_l1_regularization_strength" in xml_info:
				params["l1_regularization_strength"] = xml_info["optimizer_l1_regularization_strength"]
			if "optimizer_l2_regularization_strength" in xml_info:
				params["l2_regularization_strength"] = xml_info["optimizer_l2_regularization_strength"]
			if "optimizer_use_locking" in xml_info:
				params["use_locking"] = xml_info["optimizer_use_locking"]
			if "optimizer_name" in xml_info:
				params["name"] = xml_info["optimizer_name"]
			template_variables["optimizer_module"] = "tf.train"
			template_variables["optimizer_name"] = "'FtrlOptimizer'"
			template_variables["optimizer_params"] = params

		elif opt_name == "rmsprop":
			params = dict()
			params["learning_rate"] = learning_rate
			if "optimizer_decay" in xml_info:
				params["decay"] = xml_info["optimizer_decay"]
			if "optimizer_momentum" in xml_info:
				params["momentum"] = xml_info["optimizer_momentum"]
			if "optimizer_epsilon" in xml_info:
				params["epsilon"] = xml_info["optimizer_epsilon"]
			if "optimizer_use_locking" in xml_info:
				params["use_locking"] = xml_info["optimizer_use_locking"]
			if "optimizer_name" in xml_info:
				params["name"] = xml_info["optimizer_name"]
			template_variables["optimizer_module"] = "tf.train"
			template_variables["optimizer_name"] = "'RMSPropOptimizer'"
			template_variables["optimizer_params"] = params

	@staticmethod
	def parse_xml(parent: et.Element, node: et.Element,
	              template_variable: dict):
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
			code_generator.parse_xml(node, child, template_variable)
