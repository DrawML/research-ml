def make_optimizer(xml_info:dict, template_variables:dict):
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
