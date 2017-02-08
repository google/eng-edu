#!/bin/bash
#
# A script to permanently delete an existing Datalab VM.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

. "$(dirname "$0")/common.sh"

assert_datalab_vm_exists

prompt_user_for_confirmation "Datalab VM and its disks will be deleted forever."

echo "Deleting Datalab VM and its disks"
gcloud --quiet compute instances delete "${MLCC_INSTANCE}" \
    --zone="${MLCC_ZONE}" --delete-disks all

echo "Removing Datalab components and tools from the gcloud shell"
sudo gcloud --quiet components remove datalab

echo "Success!"
