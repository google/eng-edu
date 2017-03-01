# Google Datalab for Machine Learning Education


### Setting Up For Individual Self-Study Use

Here is how to set up your own Datalab VM running on Google Cloud:

1. Select existing or create new Google Cloud Project using your favorite
  browser. You must be in the OWNER role to continue.

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

  If you are asked to set up an SSH key or a passphrase, press enter to stick
  with the defaults.

5. Wait for the Cloud Shell console message that Datalab VM is ready for use.

  Click on the "Web preview" button of the Cloud Shell window menu (the up
  arrow icon at the top-left), then under "Change port" select port 8081 to
  open a webpage containing the Datalab application.

  ![Start Cloud Shell](img/web_preview.png)

  New browser tab with the Datalab application should open. Click "Accept" to
  accept the terms of service and proceed.

6. Click on the Notebook button to open a new notebook. Setup is now complete.
  You can start writing and executing code.

7. You may choose to permanently delete your Datalab VM after use. There is
  **no undo** for this operation. The Datalab VM, the interactive notebooks and
  all data will be permanently deleted. If you wish to proceed, from the Cloud
  Shell console, run the following command:

        ./eng-edu/ml/cc/bin/datalab_delete.sh


### Setting Up For A Class Use: One Teacher, Multiple Students

Here is how to set up several Datalab VMs running on Google Cloud for access by
your students:

#### As A Teacher

1. Select existing or create new Google Cloud Project using your favorite
  browser. You must be in the OWNER role to continue. All students would need
  a valid Google account to participate.

2. Start Cloud Shell. Click "Activate Google Cloud Shell" at the top of the
  console window. Click on "Start cloud shell" in the dialog box that opens the
  first time.

  ![Start Cloud Shell](img/cloud_shell.png)

3. Download required shell scripts. From the Cloud Shell console, run the
  following command:

        git clone https://github.com/google/eng-edu.git

  To automate bulk operations with Google Cloud projects and Datalab VMs a
  specialized command-line utility will be used. It can execute numerous
  functions and has many options that control its behaviour. We guide you
  below through the exact options to use. If you want to see all supported
  commands and their options, run one of the following command from the
  Cloud Shell console:

        python ./eng-edu/ml/cc/src/manage.py -h
        python ./eng-edu/ml/cc/src/manage.py projects_create -h
        python ./eng-edu/ml/cc/src/manage.py projects_delete -h

4. Prepare the following:
  * the billing account ID you would like to use to cover the costs of
  running student projects; you can find this information in the Billing
  section of Google Cloud Platform web interface; the value is a series of
  letters and numbers separated by dashes (i.e.`XXXXXX-XXXXXX-XXXXXX`)
  * a space-separated list of student's emails (i.e. `example.student1@gmail.com example.student2@gmail.com`); students will be given a role of EDITOR in their
  respetive projects
  * a space-separated list of owner emails,
  including your email as well as those of Teaching Assistants (i.e.
  `example.teacher@gmail.com example.ta@gmail.com`); owners will be given
  a role of OWNER in all student projects
  * a unique prefix for student project names (i.e. `project-name-prefix`);
  a portion of student email will be appended to it to create a unique project
  name (i.e. `project-name-prefix--examplestudent1`)

5. Using information above, format, and run the following command from the Cloud
  Shell console (note to remove a `--dry_run` flag):

        python ./eng-edu/ml/cc/src/manage.py projects_create --billing_id XXXXXX-XXXXXX-XXXXXX --prefix project-name-prefix --owners "example.teacher@gmail.com example.ta@gmail.com" --students "example.student1@gmail.com example.student2@gmail.com" --dry_run

6. Review the content of the audit file `account-list-#######.csv`. It contains
  a list of all students along with a Datalab VM project URL for each project
  created.

7. Send the appropriate Datalab VM project URL to each student. Invite each
  student to visit allocated project URL and follow the instructions in the
  next section. Projects can't be shared or exchanged between students.

8. You may choose to permanently delete all projects after the class ends.
  There is **no undo** for this operation. All student projects, Datalab VMs
  and data will be permanently deleted. If you wish to proceed, from the Cloud
  Shell console, run the following command (note to remove a `--dry_run` flag):

        python ./eng-edu/ml/cc/src/manage.py projects_delete --prefix project-name-prefix --students "example.student1@gmail.com example.student2@gmail.com" --dry_run

#### As A Student

1. Wait for a Teacher to inform your that your Datalab VM project is ready.

2. Using your favorite browser, navigate to a Google Cloud Project URL provided
  to you by a teacher.

3. Start Cloud Shell. Click "Activate Google Cloud Shell" at the top of the
  console window. Click on "Start cloud shell" in the dialog box that opens the
  first time.

  ![Start Cloud Shell](img/cloud_shell.png)

4. Connect Cloud Shell to a running Datalab VM. From the Cloud Shell console,
  run the following command:

        git clone https://github.com/google/eng-edu.git
        ./eng-edu/ml/cc/bin/datalab_connect.sh

  If you are asked to set up an SSH key or a passphrase, press enter to stick
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

### Troubleshooting

* **Symptom**: Waiting for Datalab VM to become accessible via "Web preview"
  never completes. The Cloud Shell console hangs after these messages:

        Connecting to mlccvm-{xxxxxxx}
        Ensuring that mlccvm-{xxxxxxx} can be connected to via SSH

  **Resolution**: The easiest resolution is to re-image your Datalab VM by
  first deleting and then re-creating it. There is **no undo** for this
  operation. The Datalab VM, the interactive notebooks and all data will be
  permanently deleted. If you wish to proceed, from the Cloud Shell console,
  run the following commands:

        ./eng-edu/ml/cc/bin/datalab_delete.sh
        ./eng-edu/ml/cc/bin/datalab_create.sh

  Make sure both commands complete with success. Now you are ready to connect
  to a re-imaged Datalab VM. From the Cloud Shell console, run the following
  command:

        ./eng-edu/ml/cc/bin/datalab_connect.sh

* **Symptom**: Everything was working fine, but then the connection to Cloud
  Shell was lost because I either closed my browser window or Cloud Shell has
  timed out.

  **Resolution**: To restart the Datalab VM, reopen the Cloud Shell window as
  described above. From the Cloud Shell console, run the following command:

        ./eng-edu/ml/cc/bin/datalab_connect.sh
