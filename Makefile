.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = k
PYTHON_INTERPRETER = python3

#################################################################################
# Reproducibillity
#################################################################################


# reproduce the loss vs time figure
loss_vs_time:
	@python src/ri_distances/eval_data_param.py  -d='g' -N=3 -p -f=0.02 -o="loss_vs_time"

# reproduce the ICP scalability figure
icp_metrics:
	@python src/ri_distances/eval_predictor.py -d='g' -N=15 -p -f=0.12 -m='sgw' -o='icp_metrics_2.png'

# start a jupyter notebook for quick visualization of point alignment algorithm
start_jupy:
	@jupyter lab --ip 0.0.0.0 --port 8888 --allow-root

#################################################################################
# Virtual Env
#################################################################################

# Create pypi virtual env
create_env:
	@echo ">>> Create environment...\n"
	@$(PYTHON_INTERPRETER) -m venv env
	@echo ">>> Environment successfully created!\n"


## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt



#################################################################################
# Docker
#################################################################################

# build the explorer image
docker_build:
	@docker build -t se3_equiv .

docker_run:
	@docker run -it \
	-v `pwd`/results:/app/results \
	-w /app \
	-e USER=$USER \
	-p 8888:8888 \
	se3_equiv \
	/bin/bash
# -v `pwd`/Makefile:/app/Makefile \
# -v `pwd`/requirements.txt:/app/requirements.txt \
# -v `pwd`/src:/app/src \
# -v `pwd`/scripts:/app/scripts \
# -v notebooks:/app/notebooks \

# start a interactive shell in the image
docker_shell:
	@docker run --rm \
	-p 8123:5000  \
	--env-file ~/.dj/config \
	-w /app \
	-it /bin/bash


#################################################################################
# Data
#################################################################################

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src


## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
