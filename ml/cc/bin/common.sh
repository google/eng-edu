#!/bin/bash
#
# Common settings and functions for managing a Datalab VM.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

MLCC_ZONE=""
MLCC_USER_NAME=$(echo "${USER}" | sed 's/[^a-zA-Z0-9]//g')
MLCC_INSTANCE=mlccvm-${MLCC_USER_NAME}

function ensure_components() {
  echo "Installing required Datalab components and tools into the gcloud shell"
  sudo gcloud --quiet components install datalab
}

function prompt_user_for_confirmation() {
  echo "$1"
  read -e -p "Do you want to proceed? [y/n] " -n 1 -r confirm
  echo

  if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "aborting"
    exit
  fi
}

function assert_datalab_vm_does_not_exist() {
  echo "Checking Datalab VM ${MLCC_INSTANCE} does not exist"
  INFO=$(gcloud compute instances list --format='value(name)')
  if [[ "${INFO}" == "" ]]; then
    return
  fi
  STATUS=$(grep "${MLCC_INSTANCE}" <<< "${INFO}")
  if [[ "${STATUS}" == "" ]]; then
    true
  else
    echo "ERROR: Datalab VM already exists"
    exit 1
  fi
}

function resolve_zone() {
  MLCC_ZONE=$(gcloud compute instances list --limit 1 \
      --filter "name:$MLCC_INSTANCE" --format 'value(zone)')
  if [[ -z "$MLCC_ZONE" ]]; then
    echo "ERROR: Unable to find zone for a Datalab VM instance $MLCC_INSTANCE"
    exit 1
  fi
  echo "Found Datalab VM ${MLCC_INSTANCE} in zone $MLCC_ZONE"
}

function assert_datalab_vm_exists() {
  echo "Checking Datalab VM ${MLCC_INSTANCE} exists"
  INFO=$(gcloud compute instances list --format='value(name)')
  if [[ "${INFO}" == "" ]]; then
    echo "ERROR: Datalab VM does not exist"
    exit 1
  fi
  STATUS=$(grep "${MLCC_INSTANCE}" <<< "${INFO}")
  if [[ "${STATUS}" == "" ]]; then
    echo "ERROR: Datalab VM does not exist"
    exit 1
  else
    true
  fi
  resolve_zone
}

function assert_datalab_vm_running() {
  echo "Checking Datalab VM ${MLCC_INSTANCE} is running"
  INFO=$(gcloud compute instances describe ${MLCC_INSTANCE} --zone ${MLCC_ZONE})
  STATUS=$(grep "status:" <<< "${INFO}")
  if [[ "${STATUS}" == "status: RUNNING" ]]; then
    true
  else
    echo "ERROR: Datalab VM must be running to continue: ${STATUS}"
    echo "Please start Datalab VM manually or recreate it if fails to start"
    exit 1
  fi
}
