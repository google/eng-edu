#!/bin/bash
#
# A script to connect the Cloud Shell Web Preview on port 8081 to a running
# Datalab VM.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

. "$(dirname "$0")/common.sh"

assert_datalab_vm_exists

assert_datalab_vm_running

ensure_components

echo "Connecting Cloud Shell Web Preview port 8081 to a running Datalab VM"
echo ""
echo "###################################################################"
echo "### The following command will not exit until CTRL-C is pressed ###"
echo "###################################################################"
echo ""
datalab connect "${MLCC_INSTANCE}" --zone="${MLCC_ZONE}" --no-launch-browser
