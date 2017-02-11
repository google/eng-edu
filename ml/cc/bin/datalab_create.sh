#!/bin/bash
#
# A script to create a new Datalab VM.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

. "$(dirname "$0")/common.sh"

assert_datalab_vm_does_not_exist

prompt_user_for_confirmation "Google Compute Engine and Cloud Machine Learning APIs will be enabled and new Datalab VM will be created."

ensure_components

echo "Enabling Google Compute Engine and Cloud Machine Learning APIs"
gcloud service-management enable compute_component
gcloud service-management enable ml.googleapis.com

echo "Provisioning and launching a new Datalab VM"
datalab create "${MLCC_INSTANCE}" --zone="${MLCC_ZONE}" --no-connect

echo "Success!"
