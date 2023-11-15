FILES := $(shell find ./src -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' -o -name "*.cc")

.PHONY: build
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

build-in-dockerfile: docker-build
	docker run -it -v $(shell pwd):/tmp/project ${DOCKER_TAG} make _build-in-dockerfile

_build-in-dockerfile:
	cmake -S . -B build-in-dockerfile
	cmake --build build-in-dockerfile

clean:
	rm -rf build
