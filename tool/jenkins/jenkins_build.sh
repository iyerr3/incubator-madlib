#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


#####################################################################################
### If this bash script is executed as a stand-alone file, assuming this
### is not part of the MADlib source code, then the following two commands
### may have to be used:
# git clone https://github.com/apache/incubator-madlib.git
# pushd incubator-madlib
#####################################################################################
workdir=`pwd`
user_name=`whoami`
echo "Build by user $user_name in directory $workdir"
echo "-------------------------------"
echo "ls -la"
ls -la
echo "-------------------------------"
echo "rm -rf build"
rm -rf build
echo "-------------------------------"
echo "rm -rf logs"
rm -rf logs
echo "mkdir logs"
mkdir logs
echo "-------------------------------"

echo "docker kill madlib"
docker kill madlib
echo "docker rm madlib"
docker rm madlib

echo "Creating docker container"
# Pull down the base docker images
docker pull madlib/postgres_9.6:jenkins
# Launch docker container with volume mounted from workdir
echo "-------------------------------"
cat <<EOF
docker run -d --name madlib -v "${workdir}/incubator-madlib":/incubator-madlib madlib/postgres_9.6:jenkins | tee logs/docker_setup.log
EOF
docker run -d --name madlib -v "${workdir}/incubator-madlib":/incubator-madlib madlib/postgres_9.6:jenkins | tee logs/docker_setup.log
echo "-------------------------------"

## This sleep is required since it takes a couple of seconds for the docker
## container to come up, which is required by the docker exec command that follows.
sleep 5

echo "---------- Building package -----------"
# FIXME: This line can be removed once rpm is installed with the image
docker exec madlib bash -c 'apt-get install -y rpm'

# cmake, make, make install, and make package
docker exec madlib bash -c 'rm -rf /build; mkdir /build; cd /build; cmake ../incubator-madlib; make clean; make; make install; make package' | tee $workdir/logs/madlib_compile.log

echo "---------- Installing and running install-check --------------------"

# Install MADlib and run install check
docker exec madlib /build/src/bin/madpack -p postgres -c postgres/postgres@localhost:5432/postgres install | tee $workdir/logs/madlib_install.log
docker exec madlib /build/src/bin/madpack -p postgres  -c postgres/postgres@localhost:5432/postgres install-check | tee $workdir/logs/madlib_install_check.log

echo "--------- Copying packages -----------------"
echo "docker cp madlib:build $workdir"
docker cp madlib:build $workdir

echo "-------------------------------"
echo "ls -la"
ls -la
echo "-------------------------------"
echo "ls -la build"
ls -la build/
echo "-------------------------------"

# convert install-check test results to junit format for reporting
python incubator-madlib/tool/jenkins/junit_export.py $workdir/logs/madlib_install_check.log $workdir/logs/madlib_install_check.xml
