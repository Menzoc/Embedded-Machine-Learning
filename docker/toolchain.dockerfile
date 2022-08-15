# Build and run:
#   docker build -t clion/ubuntu/aarch64-toolchain:1.0 -f toolchain.dockerfile .

FROM ubuntu:21.10

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata

RUN apt-get update \
  && apt-get install -y build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      make \
      ninja-build \
      cmake \
      autoconf \
      automake \
      locales-all \
      dos2unix \
      rsync \
      tar \
      python \
      python-dev \
      g++-11-aarch64-linux-gnu \
  && apt-get clean

