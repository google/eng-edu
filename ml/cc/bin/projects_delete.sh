#!/bin/bash
#
# A script to delete multiple Datalab VM projects.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

if [ "$#" -lt 3 ]; then
  echo "Please provide all required arguments to continue."
  echo "Usage:    ./projects_delete.sh  project-prefix  \
      --students \"email1 [email2 [email3 ...]]]\""
  echo "Example:  ./projects_delete.sh  learnml-20170106  \
      --students \"student1@gmail.com student2@gmail.com\""
  exit 1
fi

PROJECT_PREFIX=$1
shift

if [ "$1" == "--students" ]; then
  shift
  if [ -z "$1" ]; then
    echo "There must be at least one student"
    exit 1
  else
    STUDENT_EMAILS=(${1,,})
  fi
else
  echo "--students flag is required e.g. --students \"student1@gmail.com\""
  exit 1
fi
shift

echo "Updating/installing gcloud components"
sudo gcloud components update
sudo gcloud components install alpha

EMAILS=$@

for EMAIL in "${STUDENT_EMAILS[@]}"; do
  # set current project to the student's project
  PROJECT_ID=$(echo "${PROJECT_PREFIX}--${EMAIL}" \
      | sed 's/@/x/g' | sed 's/\./x/g' | cut -c 1-30)

  echo "Deleting project $PROJECT_ID for $EMAIL ... "
  gcloud alpha projects delete $PROJECT_ID --quiet
done

echo "Success!"
