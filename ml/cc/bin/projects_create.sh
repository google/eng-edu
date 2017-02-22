#!/bin/bash
#
# A script to create multiple Datalab VM projects.
#
# author: psimakov@google.com (Pavel Simakov)

set -e

PROJECT_HOME_URL="https://console.cloud.google.com/home/dashboard?project"
LABELS=""

if [ "$#" -lt 6 ]; then
  echo "Please provide all required arguments to continue."
  echo "Usage:    ./projects_create.sh billingid project-prefix \
      --owners \"email1 [email 2 [email3...]]\" \
      --students \"email1 [email2 [email3 ...]]]\""
  echo "You can optionally pass project labels as the last argument: \
      --labels \"name1=value1,name2=value2\""
  echo "Example:  ./projects_create.sh 0X0X0X-0X0X0X-0X0X0X learnml-170106 \
      --owners \"owner1@gmail.com owner2@gmail.com\" \
      --students \"student1@gmail.com student2@gmail.com\""
  exit 1
fi

ACCOUNT_ID=$1
shift

PROJECT_PREFIX=$1
shift

if [ "$1" == "--owners" ]; then
  shift
  if [ -z "$1" ]; then
    echo "There must be at least one owner"
    exit 1
  else
    OWNER_EMAILS=(${1,,}) # Make lowercase
  fi
else
  echo "--owners flag is required e.g. --owners \"owner1@gmail.com\""
  exit 1
fi
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

if [ "$1" == "--labels" ]; then
  shift
  if [ -z "$1" ]; then
    echo "There must be at least one label"
    exit 1
  else
    echo "Using labels $1"
    LABELS="--labels=$1"
  fi
else
  echo "No labels specified"
fi

TOTAL_STUDENT_EMAILS=${#STUDENT_EMAILS[@]}
TOTAL_OWNER_EMAILS=${#OWNER_EMAILS[@]}
ORIG_PROJECT=$(gcloud config get-value project)
LIST_FN="account-list-$(date +%s).csv"
PROGRESS=1

echo "Updating/installing gcloud components"
sudo gcloud components update
sudo gcloud components install alpha

truncate -s 0 $LIST_FN

echo "Creating $TOTAL_STUDENT_EMAILS projects; recording progress to $LIST_FN"
for STUDENT_EMAIL in "${STUDENT_EMAILS[@]}"; do
  # set current project to the student's project
  PROJECT_ID=$(echo "${PROJECT_PREFIX}--${STUDENT_EMAIL}" \
      | sed 's/@/x/g' | sed 's/\./x/g' | cut -c 1-30)

  echo "Creating project $PROJECT_ID for $STUDENT_EMAIL \
      ($PROGRESS of $TOTAL_STUDENT_EMAILS)"
  gcloud alpha projects create $PROJECT_ID $LABELS

  # wait for project to fully materialize
  sleep 2

  echo "Adding student as editor"
  gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member user:$STUDENT_EMAIL --role roles/editor

  echo "Adding Facilitators/TAs as owners"
  for OWNER_EMAIL in "${OWNER_EMAILS[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member user:$OWNER_EMAIL --role roles/owner
  done

  echo "Enabling Billing"
  gcloud alpha billing accounts projects link $PROJECT_ID \
      --account-id=$ACCOUNT_ID

  echo "Enabling Compute and ML APIs"
  gcloud service-management enable compute_component --project=$PROJECT_ID
  gcloud service-management enable ml.googleapis.com --project=$PROJECT_ID

  echo "Adding ML service account"
  gcloud beta ml init-project --project=$PROJECT_ID --quiet

  echo "Adding new firewall rule to access Datalab VM"
  gcloud config set project $PROJECT_ID
  gcloud compute firewall-rules create allow-datalab \
      --quiet --allow=tcp:22,tcp:8081

  # set current project back to the original value
  gcloud config set project $ORIG_PROJECT

  # output the email, project id, and a link to the project console
  printf "%s, %s, %s=%s\n" \
      $STUDENT_EMAIL $PROJECT_ID $PROJECT_HOME_URL $PROJECT_ID | tee -a $LIST_FN
  (( PROGRESS++ ))
done

sort -k1 -n -t, $LIST_FN

echo "Success!"

