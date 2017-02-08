#!/bin/bash
#
# A script to create a new Datalab VM.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

. "$(dirname "$0")/common.sh"

assert_datalab_vm_does_not_exist

ensure_components

echo "Provisioning and launching a new Datalab VM"
datalab create "${MLCC_INSTANCE}" --zone="${MLCC_ZONE}" --no-connect

echo "Success!"
