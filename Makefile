FILES := $(shell find ./src -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' -o -name "*.cc")

build:
	cmake -S . -b build

format:
	echo ${FILES}
	clang-format -i --style=WebKit ${FILES}
