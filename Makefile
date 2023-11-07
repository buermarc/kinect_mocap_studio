FILES := $(shell find ./src -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' -o -name "*.cc")

build:
	cmake -S . -B build
	cmake --build build

build-in-dockerfile: docker-build
	docker run -it -v $(shell pwd):/tmp/project mocap-dev:latest make _build-in-dockerfile

_build-in-dockerfile:
	cmake -S . -B build-in-dockerfile
	cmake --build build-in-dockerfile

format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}

DOCKER_TAG := mocap-dev:latest
.PHONY: docker-build
docker-build:
	docker build . -t ${DOCKER_TAG}
