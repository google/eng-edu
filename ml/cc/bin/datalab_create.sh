#!/bin/bash
#
# A script to create a new Datalab VM.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

. "$(dirname "$0")/common.sh"

prompt_user_for_confirmation "Google Compute Engine and Cloud Machine Learning APIs will be enabled and new Datalab VM will be created."

echo "Enabling Google Compute Engine and Cloud Machine Learning APIs"
gcloud services enable compute_component
gcloud services enable ml.googleapis.com

assert_datalab_vm_does_not_exist

ensure_components

echo "Provisioning a new Datalab VM"
datalab create "${MLCC_INSTANCE}" --zone="us-central1-a" --no-connect

echo "Success!"
