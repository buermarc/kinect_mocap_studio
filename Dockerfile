FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade
RUN apt-get install -y build-essential curl wget vim software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y gcc-13 g++-13 cpp-13

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-x --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-9 90

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 --slave /usr/bin/g++ g++ /usr/bin/g++-13 --slave /usr/bin/gcov gcov /usr/bin/gcov-13 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-13 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-13
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-13 130

RUN apt-get install -y libsoundio1 libsoundio-dev
RUN apt-get install -y libx11-dev libgl1-mesa-glx libxcursor-dev libxrandr2 libxinerama1

WORKDIR /tmp/packages

RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1-dev/libk4abt1.1-dev_1.1.2_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1/libk4abt1.1_1.1.2_amd64.deb


ENV DEBIAN_FRONTEND=teletype
RUN yes | dpkg -i libk4a1.4_1.4.1_amd64.deb
RUN dpkg -i libk4a1.4-dev_1.4.1_amd64.deb
RUN yes | dpkg -i libk4abt1.1_1.1.2_amd64.deb
RUN dpkg -i libk4abt1.1-dev_1.1.2_amd64.deb
RUN dpkg -i k4a-tools_1.4.1_amd64.deb

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /tmp/project

RUN apt-get install -y clang python3-dev python3-numpy python3-matplotlib python3-pip
RUN python3 -m pip install cmake
RUN echo "set editing-mode vi\nset keymap vi-command" > /root/.inputrc

RUN apt-get install -y xorg-dev libglu1-mesa-dev libtclap-dev

# COPY . .
