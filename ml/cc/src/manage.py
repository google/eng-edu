#!/usr/bin/python
#
# Copyright 2017 Google Inc. All Rights Reserved.
#

"""Classes for automation and management of Google Cloud Platform tasks.

We present a specialized command-line utility for automation and management
of Google Cloud Platform tasks. It can execute numerous functions and has
many options that control its behavior.


Fundamentally, its a simple layer directly on top of Google Cloud SDK
command-line tools gcloud and gsutils. We don't use any Python client libraries
for Google Cloud APIs. We simply call gcloud and gsutils with a lot of command-
line arguments. We do it from multiple threads and with good error recovery.
Trust me, it's much easier to do it all in Python than in bash.


In some places things are not that simple... For example: datalab_create command
supports deployment and testing of Jupyter notebooks in the remove Datalab VMs.
To accomplish this, we copy this very manage.py file into the remove Datalab VM
and then into the specific Docker VM instance, hosting a datalab. Then we invoke
manage.py inside the Docker VM (locally) so it can execute all tests in the
exact context where the actual Jupyter notebooks will run. Pretty wild, but
works very well. Google Cloud is a distributed system -- so we embrace that
fully!


Numerous commands and command-line options are available. To see a full list
run the following:

    git clone https://github.com/google/eng-edu.git
    python ./eng-edu/ml/cc/src/manage.py


Numerous tests are available to validate the package. To execute all tests
run the following:

    git clone https://github.com/google/eng-edu.git
    python ./eng-edu/ml/cc/src/manage.py tests_run


One must run this script as './eng-edu/ml/cc/src/manage.py' from the folder,
containing the git repo as shown above. This is a requirement. The most recent
Google Cloud SDK (https://cloud.google.com/sdk/downloads) is requred to execute
the actual provisioning commands. Execution of the test suite requires only
Python 2.7.


Good luck!
"""

__author__ = 'Pavel Simakov (psimakov@google.com)'

# TODO(psimakov): after 'datalab create' completes, we unable to start content
# bundle delivery and testing right away; we need to wait until provisioning of
# of Docker and TF inside this VM has finished; currenty only 'datalab connect'
# knowns how to do that wait properly and we don't; if we proceed without the
# wait, we may receive ssh timeout or errors because CPU is maxed out or Docket
# is not yet ready

import argparse
import copy_reg
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import time
import traceback
import types


MAX_STUDENTS = 40
MAX_OWNERS = 10
POOL_CONCURRENCY = 16
DOCKER_IMAGE_PREFIX = 'gcr.io/cloud-datalab/datalab:'

logging.basicConfig(format=(
    '%(asctime)s '
    '| %(levelname)s '
    '| %(processName)s '
    '| %(message)s'), datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger(__name__)
LOG.level = logging.INFO


def to_unicode(val, encoding='utf-8'):
  if isinstance(val, unicode):
    return val
  elif isinstance(val, str):
    return val.decode(encoding)
  else:
    raise Exception('Unexpected value: %s' % val)


def _pickle_method(m):
  if m.im_self is None:
    return getattr, (m.im_class, m.im_func.func_name)
  else:
    return getattr, (m.im_self, m.im_func.func_name)


def args_to_str(args):
  results = []
  for arg in args:
    if ' ' in arg:
      results.append('"%s"' % arg)
    else:
      results.append(arg)
  return ' '.join(results)


# permits serialization of instance methods for multiprocessing
copy_reg.pickle(types.MethodType, _pickle_method)


class Task(object):
  """An atomic task."""

  def __init__(self, caption, args):
    self.expect_errors = False
    self.max_try_count = 0

    self.caption = caption
    self.args = args

  def append(self, args):
    self.args += args

  def _run_once(self):
    """Runs task once."""
    try:
      child = subprocess.Popen(
          self.args,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)
    except Exception as e:  # pylint: disable=broad-except
      if not self.expect_errors:
        LOG.error('Command failed: %s; %s', args_to_str(self.args), e)
      raise
    stdout, stderr = child.communicate()
    if child.returncode:
      raise Exception('Command failed with an error code %s: %s\n(%s)\n(%s)' % (
          child.returncode, args_to_str(self.args),
          to_unicode(stderr), to_unicode(stdout)))
    return to_unicode(stdout), to_unicode(stderr)


class Command(object):
  """An executable command."""

  NAME = None
  COMMANDS = {}
  TESTS = []

  def dry_run_cmd(self, task):
    """Runs a task as fake shell command, returns no result."""
    if task.caption:
      LOG.info(task.caption)
    LOG.info('Shell: %s', task.args)
    if self.args.mock_gcloud_data:
      mock_responses = json.loads(self.args.mock_gcloud_data)
    else:
      mock_responses = {}
    key = None
    if task.args:
      key = ' '.join(str(item) for item in task.args)
    mock_response = mock_responses.get(key, None)
    if not mock_response:
      return None, None
    ret_code, ret_value = mock_response
    if ret_code:
      raise Exception('Error %s executing command: %s',
                      ret_code, args_to_str(task.args))
    return ret_value, None

  def real_run_cmd(self, task):
    """Runs a task as a real shell command, checks result and returns output."""
    if task.caption:
      LOG.info(task.caption)

    try_count = 0
    while True:
      try:
        if self.args and self.args.verbose:
          LOG.info('Shell: %s', args_to_str(task.args))
        return task._run_once()  # pylint: disable=protected-access
      except Exception as e:  # pylint: disable=broad-except
        try_count += 1
        if task.max_try_count <= try_count:
          if not task.expect_errors:
            LOG.warning('Command failed permanently after %s retries: %s; %s',
                        try_count, args_to_str(task.args), e)
          raise
        LOG.warning('Retrying failed command: %s; %s',
                    args_to_str(task.args), e)

  @classmethod
  def run(cls, task):
    """Executes a task."""
    raise NotImplementedError()

  @classmethod
  def register(cls, action_class):
    assert action_class.NAME, 'Action requires name'
    assert action_class.NAME not in cls.COMMANDS, (
        'Action %s already registered' % action_class.NAME)
    cls.COMMANDS[action_class.NAME] = action_class()

  @classmethod
  def _root_parser(cls):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=cls.COMMANDS.keys(),
                        help='A name of the action to execute.')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    return parser

  @classmethod
  def default_parser(cls, command):
    """Creates default command argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=[command.NAME],
                        help='A name of the action to execute.')
    parser.add_argument(
        '--dry_run', help='Whether to do a dry run. No real gcloud commands '
        'will be issued.', action='store_true')
    parser.add_argument(
        '--gcloud_bin', help='A specific location the the gcloud executable '
        'to use for cloud access.', default='gcloud')
    parser.add_argument(
        '--mock_gcloud_data', help='For system use during testing.')
    parser.add_argument(
        '--serial', help='Whether to disable concurrent task execution.',
        action='store_true')
    parser.add_argument(
        '--no_tests', help='Whether to skip running tests.',
        action='store_true')
    parser.add_argument(
        '--verbose', help='Whether to print detailed logging information.',
        action='store_true')
    return parser

  @classmethod
  def run_all_unit_tests(cls):
    """Runs all unit tests."""
    import tests_manage  # pylint: disable=g-import-not-at-top
    logging.disable(logging.ERROR)
    tests_manage.TestSuite.run_all_unit_tests()
    logging.disable(-1)

  @classmethod
  def dispatch(cls):
    """Receives, parses, validates and dispatches arguments to an action."""
    parser = cls._root_parser()
    try:
      parsed_args = parser.parse_args()
      cmd = Command.COMMANDS.get(
          parser.parse_args().action, (None, None))
      if not cmd:
        print 'Uknown action: %s' % parsed_args.action
        raise IllegalArgumentError()
    except:  # pylint: disable=bare-except
      print 'Usage: project.py [-h] action ...'
      print 'Valid actions are: [%s]' % ', '.join(
          sorted(Command.COMMANDS.keys()))
      sys.exit(1)

    assert cmd
    cmd_parser = cmd.make_parser
    cmd_args = cmd_parser().parse_args()
    assert cmd_args.action == parser.parse_args().action

    class NullLog(object):

      def info(self, *args):
        pass

      def error(self, *args):
        pass

      def warning(self, *args):
        pass

    if not cmd_args.no_tests:
      global LOG
      original_log = LOG
      LOG = NullLog()
      try:
        cls.run_all_unit_tests()
      finally:
        LOG = original_log

    if cmd_args.dry_run:
      LOG.warning('Dry run -- no actual commands will be executed!!!')

    LOG.info('Started %s', cmd_args.action)
    if cmd_args.serial:
      LOG.info('Disabling concurrent execution')
    else:
      LOG.info('Enabling concurrent execution')

    cmd.run = (
        cmd.dry_run_cmd if cmd_args.dry_run else cmd.real_run_cmd)

    assert not cmd.args
    cmd.args = cmd_args
    try:
      cmd.execute()
    finally:
      cmd.args = None
    LOG.info('Completed %s', cmd_args.action)

  def __init__(self):
    self.args = None

  def make_parser(self):
    """Returns args parser for this command."""
    raise NotImplementedError()

  def execute(self):
    """Executes the command."""
    raise NotImplementedError()


def get_vm_name_for_username(username):
  vm_id = re.sub(r'[^A-Za-z0-9]+', '', username)
  return 'mlccvm-%s' % vm_id


def get_user_consent_to(cmd, message):
  if not cmd.args.dry_run:
    user_action = raw_input(
        '\n\n\n###########   WARNING   ###########\n\n'
        '%s'
        '\n\n###########   WARNING   ###########\n'
        '\nDo you want to continue (y/n)?' % message.strip())
    assert user_action in ['y', 'Y'], 'Cancelled by user'


def setup_gcloud_components(cmd):
  update_comp = Task('Updating components', [
      'sudo', cmd.args.gcloud_bin, '--quiet', 'components', 'update'])
  cmd.run(update_comp)
  install_comp = Task('Installing components: alpha, datalab', [
      'sudo', cmd.args.gcloud_bin, '--quiet', 'components', 'install',
      'alpha', 'datalab'])
  cmd.run(install_comp)


def enable_compute_and_ml_apis(cmd, project_id):
  """Enables essential APIs."""
  enable_compute = Task('Enabling Compute API', [
      cmd.args.gcloud_bin, 'service-management', 'enable',
      'compute_component', '--project', project_id])
  enable_compute.max_try_count = 3
  cmd.run(enable_compute)

  enable_ml = Task('Enabling ML API', [
      cmd.args.gcloud_bin, 'service-management', 'enable',
      'ml.googleapis.com', '--project', project_id])
  enable_ml.max_try_count = 3
  cmd.run(enable_ml)


def provision_vm(cmd, student_email, project_id, zone, vm_name, image_name,
                 can_delete):
  """Provisions a new VM."""
  if check_vm_exists(cmd, project_id, zone, vm_name):
    if not can_delete:
      LOG.warning('Skipping Datalab VM provisioning; existing VM found')
      return False
    delete_vm = Task('Deleting existing Datalab VM', [
        cmd.args.gcloud_bin, '--quiet', 'compute', 'instances', 'delete',
        vm_name, '--project', project_id,
        '--zone', zone, '--delete-disks', 'all'])
    cmd.run(delete_vm)

    # wait while deletion completes
    while not cmd.args.dry_run:
      if not check_vm_exists(cmd, project_id, zone, vm_name):
        break
      else:
        wait_2_s = Task('Waiting for instance deletion', ['sleep', '2'])
        cmd.run(wait_2_s)

  create_vm = Task('Provisioning new Datalab VM', [
      'datalab', 'create', vm_name])
  if student_email:
    create_vm.append(['--for-user', student_email])
  create_vm.append([
      '--project', project_id, '--zone', zone, '--no-connect'])
  if image_name:
    create_vm.append(['--image-name', image_name])
  cmd.run(create_vm)
  return True


def validate_content_bundle(
    cmd, project_id, zone, vm_name, target, bundle_home):
  """Tests notebook content by executing it."""

  def ssh_task(args):
    return Task(None, [
        cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
        '--project', project_id, '--zone', zone, '--command'
    ] + args)

  LOG.info('Preparing Datalab VM for content bundle validation')
  self_fn = '%s/manage.py' % bundle_home

  add_self_to_vm = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'copy-files',
      '--project', project_id, '--zone', zone,
      __file__, '%s:~%s' % (vm_name, self_fn)])
  cmd.run(add_self_to_vm)

  add_self_to_docker = ssh_task([
      'docker cp ~%s %s:%s' % (self_fn, target, self_fn)])
  cmd.run(add_self_to_docker)

  LOG.info('Testing content bundle')

  try:
    remove_self_from_docker = ssh_task([
        'docker exec %s python %s bundle_test --no_tests '
        '%s--bundle_home %s>~%s/.bundle_test.out.txt '
        '2>~%s/.bundle_test.err.txt' % (target, self_fn,
                                        '--serial ' if cmd.args.serial else '',
                                        bundle_home, bundle_home, bundle_home)])
    cmd.run(remove_self_from_docker)
  except Exception as e:  # pylint: disable=broad-except
    LOG.error('Error executing %s\n%s', remove_self_from_docker.args, e)

  create_audit_dir = Task(None, [
      'mkdir', '-p', './mlcc-tmp/content_bundle_audit/'])
  cmd.run(create_audit_dir)

  get_files_off_vm = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'copy-files',
      '--project', project_id, '--zone', zone,
      '%s:~%s/' % (vm_name, bundle_home), './mlcc-tmp/content_bundle_audit/'])
  cmd.run(get_files_off_vm)

  remove_self_from_docker = ssh_task([
      'docker exec %s rm -f %s' % (target, self_fn)])
  cmd.run(remove_self_from_docker)

  remove_logs_from_vm = ssh_task(['rm -f ~%s/.bundle_test.*.txt' % bundle_home])
  cmd.run(remove_logs_from_vm)

  remove_self_from_vm = ssh_task(['rm -f ~%s' % self_fn])
  cmd.run(remove_self_from_vm)

  BundleTest.validate_content_bundle_audit('%s-%s' % (vm_name, target), cmd)


def deploy_content_bundle(cmd, project_id, zone, vm_name):
  """Deploys content to the Datalab VM."""
  bundle_source = './mlcc-tmp/content_bundle/Downloads/mlcc/'
  bundle_target = '/content/datalab/notebooks/Downloads/'
  bundle_home = '/content/datalab/notebooks/Downloads/mlcc'

  LOG.info('Deploying content bundle')
  delete_remote = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
      '--project', project_id, '--zone', zone, '--command',
      'rm -r ~%smlcc/ || true' % bundle_target])
  cmd.run(delete_remote)
  create_remote = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
      '--project', project_id, '--zone', zone, '--command',
      'mkdir -p ~%smlcc/' % bundle_target])
  cmd.run(create_remote)
  copy_to_remote = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'copy-files',
      '--project', project_id, '--zone', zone,
      bundle_source,
      '%s:~%s' % (vm_name, bundle_target)])
  cmd.run(copy_to_remote)

  # wait while Docker starts up
  target = None
  while True:
    docker_ps = Task(None, [
        cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
        '--project', project_id, '--zone', zone, '--command',
        'docker ps --format "{{.ID}}\t{{.Image}}"'])
    out, _ = cmd.run(docker_ps)
    if out:
      for line in out.strip().split('\n')[1:]:
        parts = line.split('\t')
        if parts[1].startswith(DOCKER_IMAGE_PREFIX):
          target = parts[0]
          break
    if target or cmd.args.dry_run:
      break
    else:
      wait_15_s = Task(
          'Waiting for Docker container %s...' % DOCKER_IMAGE_PREFIX,
          ['sleep', '15'])
      cmd.run(wait_15_s)

  clean_container = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
      '--project', project_id, '--zone', zone, '--command',
      'docker exec %s rm -rf %s/' % (target, bundle_home)])
  cmd.run(clean_container)

  init_container = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
      '--project', project_id, '--zone', zone, '--command',
      'docker exec %s mkdir -p %s/' % (target, bundle_home)])
  cmd.run(init_container)

  copy_to_container = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
      '--project', project_id, '--zone', zone, '--command',
      'docker cp ~%s/ %s:%s' % (
          bundle_home, target, bundle_target)])
  cmd.run(copy_to_container)

  if cmd.args.validate_content_bundle:
    validate_content_bundle(cmd, project_id, zone, vm_name, target, bundle_home)


def active_project_exists(cmd, project_id):
  check_project = Task(None, [
      cmd.args.gcloud_bin, 'projects', 'describe', project_id,
      '--format', 'value(lifecycleState)'
  ])
  check_project.expect_errors = True
  try:
    out, _ = cmd.run(check_project)
    return out and out.strip() == 'ACTIVE'
  except:  # pylint: disable=bare-except
    return False


def project_exists_in_any_state(cmd, project_id):
  check_project = Task(None, [
      cmd.args.gcloud_bin, 'projects', 'describe', project_id,
      '--format', 'value(lifecycleState)'
  ])
  check_project.expect_errors = True
  try:
    cmd.run(check_project)
    return True
  except:  # pylint: disable=bare-except
    return False


def check_vm_exists(cmd, project_id, zone, vm_name):
  check_vm = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'instances', 'describe',
      '--project', project_id, '--zone', zone, vm_name
  ])
  check_vm.expect_errors = True
  try:
    cmd.run(check_vm)
    return True
  except:  # pylint: disable=bare-except
    return False


class ContentBundleContext(object):
  """Manages a local copy of content bundle."""

  def __init__(self, cmd):
    self.cmd = cmd
    self.bundle = cmd.bundle

  def __enter__(self):
    if self.bundle:
      remove_tmp = Task(None, [
          'rm', '-rf', './mlcc-tmp/'])
      self.cmd.run(remove_tmp)
      create_tmp = Task(None, [
          'mkdir', '-p', './mlcc-tmp/content_bundle/'])
      self.cmd.run(create_tmp)

      assert self.bundle.endswith('.tar.gz'), (
          'The *.tar.gz file is expected: %s' % self.bundle)
      if self.bundle.startswith('gs://'):
        copy_to_local = Task('Loading remote content bundle %s' % self.bundle, [
            'gsutil', 'cp', self.bundle, './mlcc-tmp/content_bundle.tar.gz'])
      else:
        assert os.path.isfile(self.bundle), (
            'Content bundle not found: %s' % self.bundle)
        copy_to_local = Task('Loading local content bundle %s' % self.bundle, [
            'cp', self.bundle, './mlcc-tmp/content_bundle.tar.gz'])
      self.cmd.run(copy_to_local)

      untar = Task(None, [
          'tar', '-zxvf', './mlcc-tmp/content_bundle.tar.gz', '-C',
          './mlcc-tmp/content_bundle/'])
      self.cmd.run(untar)
      return None

  def __exit__(self, exc_type, unused_exc_value, unused_traceback):
    if not exc_type and self.bundle:
      remove_tmp = Task(None, [
          'rm', '-rf', './mlcc-tmp/'])
      self.cmd.run(remove_tmp)


class RunTests(Command):
  """An action that runs all tests."""

  NAME = 'tests_run'

  def make_parser(self):
    """Creates default argument parser."""
    return self.default_parser(self)

  def execute(self):
    """Runs all tests."""
    pass


class ProjectsCreate(Command):
  """An action that creates projects in bulk."""

  NAME = 'projects_create'
  CREATE_PROJECTS_RESULTS_FN = 'account-list-%s.csv' % int(time.time())

  SUPPORTED_DATALAB_IMAGES = {
      'TF_RC0': ('TensorFlow 0.11, RC0',
                 'gcr.io/cloud-datalab/datalab:local-20170127'),
      'TF_1': ('TensorFlow 1.0.0',
               'gcr.io/cloud-datalab/datalab:local-20170224')
  }

  def __init__(self):
    super(ProjectsCreate, self).__init__()
    self.billing_id = None
    self.bundle = None
    self.image_name = None
    self.labels = None
    self.owner_emails = None
    self.student_emails = None
    self.prefix = None
    self.zone = None

  def _images_names(self):
    items = []
    for image_key, image_def in self.SUPPORTED_DATALAB_IMAGES.iteritems():
      items.append('"%s" (%s)' % (image_key, image_def[0]))
    return ', '.join(items)

  def make_parser(self):
    """Creates default argument parser."""
    parser = self.default_parser(self)
    parser.add_argument(
        '--billing_id', help='A billing account ID you would like to '
        'use to cover the costs of running student projects. You can find this '
        'information in the Billing section of Google Cloud Platform web '
        'interface. The value is a series of letters and numbers separated '
        'by dashes (i.e. XXXXXX-XXXXXX-XXXXXX).', required=True)
    parser.add_argument(
        '--content_bundle', help='An optional name of the tar.gz file with the '
        'content bundle to deploy to each Datalab VM. This could be a name of '
        'the local file (i.e. ~/my_notebooks.tar.gz) or a name of a file in '
        'the Google Cloud Storage Bucket '
        '(i.e. gs://my_busket/my_notebooks.tar.gz).')
    parser.add_argument(
        '--image_name', help='An optional name of the specific VM image to '
        'pass into "datalab create". Default image will be used if no value '
        'is provided. Supported images are: %s.' % self._images_names())
    parser.add_argument(
        '--labels', help='A comma-separated list of project labels '
        '(i.e. "foo=bar,alice=bob").')
    parser.add_argument(
        '--owners', help='A space-separated list of owner emails, including '
        'your email as well as those of Teaching Assistants . Owners will be '
        'given a role of OWNER in all student projects.', required=True)
    parser.add_argument(
        '--prefix', help='A unique prefix for student project '
        'names. A portion of student email will be appended to it to create '
        'a unique project name.', required=True)
    parser.add_argument(
        '--provision_vm', help='Whether to re-provision Datalab VMs in the '
        'existing projects. If set, existing Datalab VMs will be deleted and '
        'new Datalab VMs will be provisioned: there is NO UNDO. If not set, '
        'existing project VMs will not be altered, but fresh VMs will be '
        'provisioned in the newly created projects.', action='store_true')
    parser.add_argument(
        '--students', help='A space-separated list of student\'s emails. '
        'Students will be given a role of EDITOR in their respetive projects.',
        required=True)
    parser.add_argument(
        '--zone', help='A name of the Google Cloud Platform zone to host '
        'your resources.', default='us-central1-a')
    return parser

  def _parse_args(self, args):
    """Parses args."""
    self.args.validate_content_bundle = False
    self.billing_id = args.billing_id
    self.bundle = args.content_bundle

    image_key = args.image_name
    if image_key:
      assert image_key in self.SUPPORTED_DATALAB_IMAGES, (
          'Unsupported image name "%s". '
          'Supported images are: %s.' % (image_key, self._images_names()))
      self.image_name = self.SUPPORTED_DATALAB_IMAGES[image_key][1]
    else:
      self.image_name = None

    self.labels = args.labels
    self.owner_emails = args.owners.lower().split(' ')
    self.student_emails = args.students.lower().split(' ')
    self.prefix = args.prefix
    self.zone = args.zone

    if len(self.owner_emails) > MAX_OWNERS:
      raise Exception('Too many owners: %s max.' % MAX_OWNERS)
    if len(self.student_emails) > MAX_STUDENTS:
      raise Exception('Too many students: %s max.' % MAX_STUDENTS)
    self.assert_emails_and_project_ids_are_unique()

  @classmethod
  def project_id(cls, prefix, email):
    user_id = re.sub(r'[^A-Za-z0-9]+', '', email)
    return ('%s--%s' % (prefix, user_id))[:29]

  def _project_home(self, project_id):
    return (
        'https://console.cloud.google.com/home/dashboard?'
        'project=%s' % project_id)

  @classmethod
  def vm_name(cls, email):
    parts = email.split('@')
    assert len(parts) > 1, 'Bad email: %s' % email
    return get_vm_name_for_username(parts[0])

  def assert_emails_and_project_ids_are_unique(self):
    projects_ids = {}
    for student_email in self.student_emails:
      project_id = self.project_id(self.prefix, student_email)
      if project_id in projects_ids:
        existing_email = projects_ids[project_id]
        if existing_email == student_email:
          raise Exception('Dupicate email %s.', student_email)
        raise Exception('Emails %s and %s lead to a duplicate project_id %s.',
                        student_email, existing_email, project_id)
      projects_ids[project_id] = student_email

  def create_project(self, student_email, project_id):
    """Creates and configures new project."""
    create_project = Task('Creating new project %s for student %s' % (
        project_id, student_email), [
            self.args.gcloud_bin, 'alpha', 'projects', 'create', project_id])
    if self.labels:
      create_project.append(['--labels', self.labels])
    self.run(create_project)

    # wait while creation completes
    while not self.args.dry_run:
      if active_project_exists(self, project_id):
        break
      else:
        wait_2_s = Task('Waiting for project creation', ['sleep', '2'])
        self.run(wait_2_s)

    for owner_email in self.owner_emails:
      add_owner_role = Task('Adding %s as owner' % owner_email, [
          self.args.gcloud_bin, 'projects', 'add-iam-policy-binding',
          project_id, '--member', 'user:%s' % owner_email,
          '--role', 'roles/owner'])
      self.run(add_owner_role)

    add_student_role = Task('Adding %s as student' % student_email, [
        self.args.gcloud_bin, 'projects', 'add-iam-policy-binding', project_id,
        '--member', 'user:%s' % student_email, '--role', 'roles/editor'])
    self.run(add_student_role)

    enable_billing = Task('Enabling Billing', [
        self.args.gcloud_bin, 'alpha', 'billing', 'accounts', 'projects',
        'link', project_id, '--account-id', self.billing_id])
    self.run(enable_billing)

    enable_compute_and_ml_apis(self, project_id)

  def _execute_one_non_raising(self, student_email):
    try:
      return self._execute_one_raising(student_email)
    except:  # pylint: disable=bare-except
      LOG.error(traceback.format_exc())
      raise

  def _execute_one_raising(self, student_email):
    """Creates one student project."""
    project_id = self.project_id(self.prefix, student_email)
    vm_name = self.vm_name(student_email)

    need_to_provision_vm = True
    if not project_exists_in_any_state(self, project_id):
      self.create_project(student_email, project_id)
    else:
      if self.args.provision_vm:
        LOG.warning('Re-provisioning Datalab VM for existing '
                    'project %s for student %s', project_id, student_email)
      else:
        LOG.warning('Skipping re-provisioning of Datalab VM for existing '
                    'project %s for student %s', project_id, student_email)
        need_to_provision_vm = False

    if need_to_provision_vm:
      provision_vm(self, student_email, project_id, self.zone, vm_name,
                   self.image_name, True)

    if self.bundle:
      deploy_content_bundle(self, project_id, self.zone, vm_name)

    return student_email, project_id, vm_name, self._project_home(project_id)

  def execute(self):
    """Creates projects in bulk."""
    self._parse_args(self.args)
    setup_gcloud_components(self)

    with ContentBundleContext(self):
      LOG.info('Creating Datalab VM projects for %s students and %s owners',
               len(self.student_emails), len(self.owner_emails))
      if self.args.serial:
        rows = []
        for student_email in self.student_emails:
          row = self._execute_one_non_raising(student_email)
          rows.append(row)
      else:
        pool = multiprocessing.Pool(processes=POOL_CONCURRENCY)
        rows = pool.map(self._execute_one_non_raising, self.student_emails)

    LOG.info('Writing results to ./%s', self.CREATE_PROJECTS_RESULTS_FN)
    with open(self.CREATE_PROJECTS_RESULTS_FN, 'w') as output:
      output.write('student_email\tproject_id\tvm_name\tproject_url\n')
      for row in sorted(rows, key=lambda row: row[0]):
        output.write('\t'.join(row))
        output.write('\n')

    print self.CREATE_PROJECTS_RESULTS_FN


class ProjectsDelete(Command):
  """An action that deletes projects in bulk."""

  NAME = 'projects_delete'

  def __init__(self):
    super(ProjectsDelete, self).__init__()
    self.student_emails = None
    self.prefix = None

  def make_parser(self):
    """Creates default argument parser."""
    parser = self.default_parser(self)
    parser.add_argument(
        '--prefix', help='A unique prefix for student project '
        'names. A portion of student email will be appended to it to create '
        'a unique project name.', required=False)
    parser.add_argument(
        '--project_ids',
        help='A space-separated list of project ids. '
        'A project with each specified id will be deleted.',
        required=False
    )
    parser.add_argument(
        '--provision_vm', help='Whether to re-provision Datalab VMs in the '
        'existing projects. If set, existing Datalab VMs will be deleted and '
        'new Datalab VMs will be provisioned: there is NO UNDO. If not set, '
        'existing project VMs will not be altered, but fresh VMs will be '
        'provisioned in the newly created projects.', action='store_true')
    parser.add_argument(
        '--students', help='A space-separated list of student\'s emails. '
        'A project for each specified student will be deleted.',
        required=False
    )
    return parser

  def _parse_args(self, args):
    """Parses args."""
    self.prefix = args.prefix
    self.student_emails = None
    if args.students:
      self.student_emails = args.students.lower().split(' ')
    self.project_ids = None
    if args.project_ids:
      self.project_ids = args.project_ids.lower().split(' ')
    assert self.student_emails or self.project_ids, (
        'Please provide --student_emails or --project_ids.')
    assert not(self.student_emails and self.project_ids), (
        'Please provide --student_emails or --project_ids, not both.')
    if self.student_emails:
      assert self.student_emails and self.prefix, (
          'Please provide --prefix when providing --student_emails.')
    else:
      assert not self.prefix, (
          'The --prefix can\'t be specified when providing --project_ids.')

  def delete_project(self, project_id):
    if not active_project_exists(self, project_id):
      LOG.warning('Project not found; unable to delete: %s', project_id)
    else:
      delete_project = Task('Deleting project %s' % project_id, [
          self.args.gcloud_bin, '--quiet', 'alpha', 'projects', 'delete',
          project_id])
      self.run(delete_project)

  def delete_by_student_emails(self, student_emails):
    for student_email in student_emails:
      project_id = ProjectsCreate.project_id(self.prefix, student_email)
      self.delete_project(project_id)

  def delete_by_project_ids(self, project_ids):
    for project_id in project_ids:
      self.delete_project(project_id)

  def execute(self):
    """Deletes projects in bulk."""
    self._parse_args(self.args)
    setup_gcloud_components(self)
    if self.student_emails:
      LOG.info('Deleting Datalab VM projects for %s students',
               len(self.student_emails))
      self.delete_by_student_emails(self.student_emails)
    if self.project_ids:
      LOG.info('Deleting Datalab VM projects for %s project ids',
               len(self.project_ids))
      self.delete_by_project_ids(self.project_ids)


class DatalabCommonMixin(object):
  """A mixin with common helper functions."""

  def resolve_zone_for(self, project_id, vm_name):
    find_vm = Task(None, [
        self.args.gcloud_bin, 'compute', 'instances', 'list',
        '--project', project_id, '--limit', '1',
        '--filter', 'name:%s' % vm_name, '--format', 'value(zone)'])
    out, _ = self.run(find_vm)
    zone = None
    if out:
      zone = str(out).strip()
    return zone

  def get_project_id_and_vm_name(self):
    """Computes project_id and vm_name."""
    username = os.environ.get('USER', None)
    assert username, 'Failed to determine current username'

    vm_name = get_vm_name_for_username(username)
    get_current_project_id = Task(None, [
        self.args.gcloud_bin, 'config', 'get-value', 'project'])
    out, _ = self.run(get_current_project_id)
    project_id = None
    if out:
      project_id = str(out).strip()
    assert project_id or self.args.dry_run, (
        'Failed to determine current project')
    return project_id, vm_name


class DatalabCreate(Command, DatalabCommonMixin):
  """An action that creates a Datalab VM in the current project."""

  NAME = 'datalab_create'
  REQUIRED_IMAGE_URL_PREFIX = 'gcr.io/'

  def make_parser(self):
    """Creates default argument parser."""
    parser = self.default_parser(self)
    parser.add_argument(
        '--content_bundle', help='An optional name of the tar.gz file with the '
        'content bundle to deploy to each Datalab VM. This could be a name of '
        'the local file (i.e. ~/my_notebooks.tar.gz) or a name of a file in '
        'the Google Cloud Storage Bucket '
        '(i.e. gs://my_busket/my_notebooks.tar.gz).')
    parser.add_argument(
        '--image_url', help='An optional URL of the specific VM image to '
        'pass into "datalab create", for example: '
        '"gcr.io/cloud-datalab/datalab:local-20170127". Default image will '
        'be used if no value is provided.')
    parser.add_argument(
        '--provision_vm', help='Whether to re-provision Datalab VM. '
        'If set, existing Datalab VM will be deleted and '
        'new Datalab VM will be provisioned: there is NO UNDO. If not set, '
        'existing Datalab VM will not be altered, but fresh VM will be '
        'provisioned if none exists.', action='store_true')
    parser.add_argument(
        '--validate_content_bundle', help='Whether to execute all notebooks '
        'in the content bundle to validate them.', action='store_true')
    parser.add_argument(
        '--zone', help='A name of the Google Cloud Platform zone to host '
        'your resources.', default='us-central1-a')
    return parser

  def execute(self):
    """Creates a Datalab VM in the current project."""
    if self.args.image_url:
      assert self.args.image_url.startswith(self.REQUIRED_IMAGE_URL_PREFIX), (
          'Image URL must start with %s' % self.REQUIRED_IMAGE_URL_PREFIX)

    project_id, vm_name = self.get_project_id_and_vm_name()

    get_user_consent_to(
        self,
        '1. Current project %s will be altered: Google Compute Engine and '
        'Cloud Machine Learning APIs will be enabled.\nTo change your current '
        'project, execute the following from shell: "gcloud config set '
        'project <your_project_name>".\n\n'
        '2. A Datalab VM %s will be created or re-provisioned in zone %s.' % (
            project_id, vm_name, self.args.zone))

    LOG.info('Provisioning Datalab VM %s in project %s', vm_name, project_id)

    setup_gcloud_components(self)
    enable_compute_and_ml_apis(self, project_id)

    provision_vm(self, None, project_id, self.args.zone, vm_name,
                 self.args.image_url, self.args.provision_vm)
    if self.args.content_bundle and check_vm_exists(
        self, project_id, self.args.zone, vm_name):
      self.bundle = self.args.content_bundle
      with ContentBundleContext(self):
        deploy_content_bundle(self, project_id, self.args.zone, vm_name)


class DatalabConnect(Command, DatalabCommonMixin):
  """An action that connects to a Datalab VM of the current project."""

  NAME = 'datalab_connect'

  CTRL_C_MSG = '''

Connecting Cloud Shell Web Preview port 8081 to a running Datalab VM

###################################################################
### The following command will not exit until CTRL-C is pressed ###
###################################################################

'''

  def make_parser(self):
    """Creates default argument parser."""
    return self.default_parser(self)

  def execute(self):
    """Connects Cloud Shell to a Datalab VM of the current project."""

    # TODO(psimakov): how to assert we are in the Cloud Shell?
    LOG.warning('Make sure you run this command from the Google Cloud Shell. '
                'We unable to verify this automatically.')

    project_id, vm_name = self.get_project_id_and_vm_name()
    zone = self.resolve_zone_for(project_id, vm_name)

    supress_assert = self.args.dry_run and not self.args.mock_gcloud_data
    if not supress_assert:
      assert zone, 'Datalab VM %s not found in project %s' % (
          vm_name, project_id)

    setup_gcloud_components(self)

    LOG.info(self.CTRL_C_MSG)
    connect_to = Task(
        'Connecting to Datalab VM %s in zone %s in project %s' % (
            vm_name, zone, project_id), [
                'datalab', 'connect', vm_name, '--project', project_id,
                '--zone', zone, '--no-launch-browser'])
    self.run(connect_to)
    # TODO(psimakov): somehow this does not work in the Cloud Shell; why?


class DatalabDelete(Command, DatalabCommonMixin):
  """An action that deletes a Datalab VM in the current project."""

  NAME = 'datalab_delete'

  def make_parser(self):
    """Creates default argument parser."""
    return self.default_parser(self)

  def execute(self):
    """Deletes a Datalab VM in the current project."""
    project_id, vm_name = self.get_project_id_and_vm_name()
    zone = self.resolve_zone_for(project_id, vm_name)

    supress_assert = self.args.dry_run and not self.args.mock_gcloud_data
    if not supress_assert:
      assert zone, 'Datalab VM %s not found in project %s' % (
          vm_name, project_id)

    get_user_consent_to(
        self,
        '1. Project %s will be altered.\n\n'
        '2. Datalab VM %s in zone %s and ALL ITS DISKS AND DATA will be '
        'deleted forever.' % (project_id, vm_name, zone))

    delete_vm = Task('Deleting Datalab VM and its disks', [
        self.args.gcloud_bin, '--quiet', 'compute', 'instances', 'delete',
        vm_name, '--project', project_id, '--zone', zone, '--delete-disks',
        'all'])
    self.run(delete_vm)

    delete_component = Task(
        'Removing Datalab components and tools from the gcloud shell',
        ['sudo', self.args.gcloud_bin, '--quiet', 'components', 'remove',
         'datalab'])
    self.run(delete_component)


class BundleTest(Command):
  """Enumerates and executes all Jupyter notebooks and creates a report."""

  NAME = 'bundle_test'

  POOL_CONCURRENCY = 4

  STD_OUT = './mlcc-tmp/content_bundle_audit/mlcc/.bundle_test.out.txt'
  STD_ERR = './mlcc-tmp/content_bundle_audit/mlcc/.bundle_test.err.txt'

  def make_parser(self):
    """Creates default argument parser."""
    parser = self.default_parser(self)
    parser.add_argument(
        '--bundle_home', help='A folder that contains the content bundle.',
        required=True)
    return parser

  @classmethod
  def validate_content_bundle_audit(cls, docker_host, cmd):
    """Analyse the output of this command."""
    LOG.info('Checking if content bundle is valid')

    if not cmd.args.dry_run:
      assert os.path.isfile(cls.STD_OUT)
      assert os.path.isfile(cls.STD_ERR)

      results_fn = 'bundle-test-%s.json' % int(time.time())
      remote_log_fn = 'remote-log-%s-%s.txt' % (docker_host, int(time.time()))

      with open(cls.STD_ERR, 'r') as err:
        LOG.info('Writing host %s log into file ./%s',
                 docker_host, remote_log_fn)
        with open(remote_log_fn, 'w') as stream:
          stream.write(err.read())

      with open(cls.STD_OUT, 'r') as out:
        text = out.read()
        LOG.info('Writing host %s test results into file ./%s',
                 docker_host, results_fn)
        with open(results_fn, 'w') as stream:
          stream.write(text)

      data = json.loads(text)
      assert data['command'] == 'BundleTest'

      count = 0
      errors = 0
      for result in data['data']:
        if result.get('error_code') == 0:
          count += 1
        else:
          errors += 1
      LOG.info('Found %s completed tests', count)
      if errors:
        LOG.error('Found %s failed tests', errors)

  def iterate_notebooks(self, folder):
    if self.args.mock_gcloud_data:
      mock_responses = json.loads(self.args.mock_gcloud_data)
      if 'iterate_notebooks' in mock_responses:
        for afile in mock_responses.get('iterate_notebooks'):
          yield afile
        return
    for root, unused_dirs, files in os.walk(folder):
      for afile in files:
        if not afile.startswith('.') and afile.endswith('.ipynb'):
          yield os.path.join(root, afile)

  def test_one_notebook(self, fn):
    """Test a notebook by executing it."""

    # Read more about it here:
    # http://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/

    result_fn = '%s.new.ipynb' % fn

    run_notebook = Task(None, [
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
        '--output', result_fn, '--ExecutePreprocessor.timeout=-1', fn])

    out = None
    err = None
    result = None

    try:
      out, err = self.run(run_notebook)
      if os.path.isfile(result_fn):
        with open(result_fn, 'r') as stream:
          result = stream.read()
        remove_result_file = Task(None, ['rm', '-f', fn])
        self.run(remove_result_file)
      error_code = 0
    except Exception as e:  # pylint: disable=broad-except
      err = str(e)
      error_code = 1

    return {'notebook': fn, 'cmd': run_notebook.args,
            'error_code': error_code, 'stdout': out, 'stderr': err,
            'result': result}

  def execute(self):
    """Enumerates and executes all Jupyter notebooks and creates a report."""
    LOG.info('Executing content bundle')
    if self.args.serial:
      rows = []
      for filename in self.iterate_notebooks(self.args.bundle_home):
        row = self.test_one_notebook(filename)
        rows.append(row)
    else:
      pool = multiprocessing.Pool(processes=self.POOL_CONCURRENCY)
      rows = pool.map(self.test_one_notebook,
                      self.iterate_notebooks(self.args.bundle_home))

    rows.sort(key=lambda item: item['notebook'])
    result = {'command': 'BundleTest', 'args': vars(self.args), 'data': rows}
    print json.dumps(result, indent=4, sort_keys=True)


Command.register(BundleTest)
Command.register(DatalabCreate)
Command.register(DatalabConnect)
Command.register(DatalabDelete)
Command.register(ProjectsCreate)
Command.register(ProjectsDelete)
Command.register(RunTests)


if __name__ == '__main__':
  Command.dispatch()
