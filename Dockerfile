from ubuntu:16.04

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-kinetic-ros-base --allow-unauthenticated
RUN echo ". /opt/ros/kinetic/setup.bash" >> ~/.bashrc

RUN apt-get install -y wget git
RUN apt-get install -y ros-kinetic-cv-bridge
RUN cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

COPY . /app

WORKDIR /app
RUN pip install -r requirements.txt

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["/app/bin/docker-entrypoint.bash"]

CMD ["python libs/rosbag_extractor.py --help"]
