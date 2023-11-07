FILES := $(shell find ./src -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' -o -name "*.cc")

build:
	cmake -S . -B build
	cmake --build build

format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}

DOCKER_TAG := mocap-dev:latest
.PHONY: docker-build
docker-build:
	docker build . -t ${DOCKER_TAG}
