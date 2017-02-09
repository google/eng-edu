# Google Datalab for Machine Learning Education


### Setting Up {#setup}

Here is how to setup your own Datalab VM running on Google Cloud:

1. Select existing or create new Google Cloud Project using your favorite
  browser.

2. Start Cloud Shell. Click "Activate Google Cloud Shell" at the top of the
  console window. Click on "Start cloud shell" in the dialog box that opens the
  first time.

  ![Start Cloud Shell](img/cloud_shell.png)

3. Create a new Datalab VM. From the Cloud Shell console, run the following
  commands:

        git clone https://github.com/google/eng-edu.git
        ./eng-edu/ml/cc/bin/datalab_create.sh

4. Connect Cloud Shell to a running Datalab VM. From the Cloud Shell console,
  run the following command:

        ./eng-edu/ml/cc/bin/datalab_connect.sh

  If you are asked to setup an SSH key or a passphrase, press enter to stick
  with the defaults.

5. Wait for the Cloud Shell console message that Datalab is ready for use.

  Click on the "Web preview" button of the Cloud Shell window menu (the up
  arrow icon at the top-left), then under "Change port" select port 8081 to
  open a webpage containing the Datalab application.

  ![Start Cloud Shell](img/web_preview.png)

  New browser tab with the Datalab application should open. Click "Accept" to
  accept the terms of service and proceed.

6. Click on the Notebook button to open a new notebook. Setup is now complete.
  You can start writing and executing code.


### Troubleshooting {#troubleshooting}

* Google Compute Engine and Cloud ML APIs should already be enabled for your
  project. If not enabled, you can enable them by going to these two links:
    * [Compute Engine API]
      (https://console.cloud.google.com/apis/api/compute_component/overview)
    * [Cloud ML API]
      (https://console.cloud.google.com/apis/api/ml.googleapis.com/overview)

  Click the "Enable" button and wait until operation completes.

  ![Enable API](img/enable_api.png){height="29px"}


* If you receive a message that the Cloud ML service accounts are unable to
  access resources in your Google Cloud project run the following command from
  your Cloud Shell console:

          gcloud beta ml init-project

* Everything was working fine, but then the connection to Cloud Shell was lost.
  To restart the Datalab VM, reopen the Cloud Shell window as described above.
  From the Cloud Shell console, run the following command:

        ./eng-edu/ml/cc/bin/datalab_connect.sh

* If you need to delete a Datalab VM after you're done using it, run the
  following command from the Cloud Shell console:

        ./eng-edu/ml/cc/bin/datalab_delete.sh
