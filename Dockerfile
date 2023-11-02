FROM ubuntu:20.04

RUN apt-get update && apt-get upgrade
RUN apt-get install -y build-essential curl wget vim

RUN apt-get install -y libsoundio1 libsoundio-dev
RUN apt-get install -y libx11-dev libgl1-mesa-glx libxcursor-dev libxrandr2 libxinerama1

WORKDIR /tmp/packages

RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1-dev/libk4abt1.1-dev_1.1.2_amd64.deb
RUN wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1/libk4abt1.1_1.1.2_amd64.deb


RUN yes | dpkg -i libk4a1.4_1.4.1_amd64.deb
RUN dpkg -i libk4a1.4-dev_1.4.1_amd64.deb
RUN yes | dpkg -i libk4abt1.1_1.1.2_amd64.deb
RUN dpkg -i libk4abt1.1-dev_1.1.2_amd64.deb
RUN dpkg -i k4a-tools_1.4.1_amd64.deb

WORKDIR /tmp/project

RUN apt-get install -y cmake
