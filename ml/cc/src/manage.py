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
for Google Cloud APIs. We simply call gcloud and gsutils with a lot of
arguments. We do it from multiple threads and with good error recovery. Trust
me, it's much easier to do it all in Python than in bash.


Numerous commands and command-line options are available. To see a full list
run the following:

    python ./eng-edu/ml/cc/src/manage.py


Numerous tests are available to validate the package. To execute all tests
run the following:

    python ./eng-edu/ml/cc/src/manage.py tests_run


The most recent Google Cloud SDK (https://cloud.google.com/sdk/downloads) is
requred to execute actual provisioning commands. Tests require only Python 2.7.


Good luck!
"""

__author__ = 'Pavel Simakov (psimakov@google.com)'


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
import unittest


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
        return task._run_once()
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
    suites_list = []
    for test_class in cls.TESTS:
      suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
      suites_list.append(suite)
    result = unittest.TextTestRunner().run(unittest.TestSuite(suites_list))
    if not result.wasSuccessful() or result.errors:
      raise Exception(result)

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
        '\n\n\nWARNING!!!\n'
        '%s'
        '\n\n\nDo you want to continue (y/n)?' % message)
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


def deploy_content_bundle(cmd, project_id, zone, vm_name):
  """Deploys content to the Datalab VM."""
  bundle_source = './mlcc-tmp/content_bundle/Downloads/mlcc/'
  bundle_target = '/content/datalab/notebooks/Downloads/'

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
      'docker exec %s rm -rf %smlcc/' % (target, bundle_target)])
  cmd.run(clean_container)

  init_container = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
      '--project', project_id, '--zone', zone, '--command',
      'docker exec %s mkdir -p %smlcc/' % (target, bundle_target)])
  cmd.run(init_container)

  copy_to_container = Task(None, [
      cmd.args.gcloud_bin, 'compute', 'ssh', vm_name,
      '--project', project_id, '--zone', zone, '--command',
      'docker cp ~%smlcc/ %s:%s' % (
          bundle_target, target, bundle_target)])
  cmd.run(copy_to_container)


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

  def __exit__(self, *args):
    if self.bundle:
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
                 'gcr.io/cloud-datalab/datalab:local-20170127')
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

  @classmethod
  def add_common_args(cls, parser):
    parser.add_argument(
        '--prefix', help='A unique prefix for student project '
        'names. A portion of student email will be appended to it to create '
        'a unique project name.', required=True)
    parser.add_argument(
        '--students', help='A space-separated list of student\'s emails. '
        'Students will be given a role of EDITOR in their respetive projects.',
        required=True
    )

  def _images_names(self):
    items = []
    for image_key, image_def in self.SUPPORTED_DATALAB_IMAGES.iteritems():
      items.append('"%s" (%s)' % (image_key, image_def[0]))
    return ', '.join(items)

  def make_parser(self):
    """Creates default argument parser."""
    parser = self.default_parser(self)
    self.add_common_args(parser)
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
        '--provision_vm', help='Whether to re-provision Datalab VMs in the '
        'existing projects. If set, existing Datalab VMs will be deleted and '
        'new Datalab VMs will be provisioned: there is NO UNDO. If not set, '
        'existing project VMs will not be altered, but fresh VMs will be '
        'provisioned in the newly created projects.', action='store_true')
    parser.add_argument(
        '--zone', help='A name of the Google Cloud Platform zone to host '
        'your resources.', default='us-central1-a')
    return parser

  def _parse_args(self, args):
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

    LOG.info('Writing results to %s', self.CREATE_PROJECTS_RESULTS_FN)
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
    ProjectsCreate.add_common_args(parser)
    return parser

  def _parse_args(self, args):
    self.student_emails = args.students.lower().split(' ')
    self.prefix = args.prefix

  def execute(self):
    """Deletes projects in bulk."""
    self._parse_args(self.args)
    LOG.info('Deleting Datalab VM projects for %s students',
             len(self.student_emails))
    setup_gcloud_components(self)
    for student_email in self.student_emails:
      project_id = ProjectsCreate.project_id(self.prefix, student_email)
      if not active_project_exists(self, project_id):
        LOG.warning('Project not found; unable to delete: %s', project_id)
      else:
        delete_project = Task('Deleting project %s' % project_id, [
            self.args.gcloud_bin, '--quiet', 'alpha', 'projects', 'delete',
            project_id])
        self.run(delete_project)


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
        '1. Project %s will be altered: Google Compute Engine and Cloud '
        'Machine Learning APIs will be enabled.\n'
        '2. A Datalab VM %s will be created or re-provisioned in zone %s.' % (
            project_id, vm_name, self.args.zone))

    LOG.info('Provisioning Datalab VM %s in project %s', vm_name, project_id)

    setup_gcloud_components(self)
    enable_compute_and_ml_apis(self, project_id)

    provisioned = provision_vm(self, None, project_id, self.args.zone, vm_name,
                               self.args.image_url, self.args.provision_vm)
    if provisioned and self.args.content_bundle:
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
        '1. Project %s will be altered.\n'
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


# TODO(psimakov): move all tests out
class CommonTestMixin(object):
  """A mixin that has test helper functions."""

  PREFIX = ' | Shell: ['
  GCLOUD_PY = os.path.join(os.path.dirname(__file__), 'manage.py')

  def filter_log(self, items):
    """Extracts only the log lines that contain shell commands."""
    results = []
    for item in items:
      index = item.find(self.PREFIX)
      if index == -1:
        continue
      results.append(item[index + len(self.PREFIX): -1])
    results.append(u'')  # to match the new line at the end of the data file
    return results

  def assert_file_equals(self, actual, fn):
    """Asserts that file content is equal to data."""
    with open(fn, 'r') as expected:
      self.assertEquals(
          to_unicode(expected.read()).split('\n'),
          self.filter_log(actual.split('\n')))


class CoreTests(unittest.TestCase):
  """Tests."""

  def test_project_id(self):
    project_id = ProjectsCreate.project_id('foo', 'foo@gmail.com')
    self.assertEquals('foo--foogmailcom', project_id)

  def test_project_id_email_too_long(self):
    project_id = ProjectsCreate.project_id(
        'foo', '123456789012345678901234567890foo@gmail.com')
    self.assertEquals('foo--123456789012345678901234', project_id)

  def test_vm_name(self):
    vm_name = ProjectsCreate.vm_name('foo@gmail.com')
    self.assertEquals('mlccvm-foo', vm_name)


class RetryTests(unittest.TestCase):
  """Test task retires."""

  def test_not_retriable_is_not_retried(self):

    class MyTask(Task):

      def _run_once(self):
        raise Exception('I always fail.')

    with self.assertRaisesRegexp(Exception, 'I always fail.'):
      Command().real_run_cmd(MyTask(None, []))

  def test_retriable_is_retried(self):

    class MyTask(Task):

      def __init__(self):
        super(MyTask, self).__init__(None, [])
        self.try_count = 0

      def _run_once(self):
        self.try_count += 1
        if self.try_count == 1:
          raise Exception('I always fail once.')

    with self.assertRaisesRegexp(Exception, 'I always fail once.'):
      Command().real_run_cmd(MyTask())

    task = MyTask()
    task.max_try_count = 3
    Command().real_run_cmd(task)
    self.assertEquals(2, task.try_count)

  def test_max_try_count_is_respected(self):

    class MyTask(Task):

      def __init__(self):
        super(MyTask, self).__init__(None, [])
        self.try_count = 0

      def _run_once(self):
        self.try_count += 1
        raise Exception('I always fail even with retry.')

    with self.assertRaisesRegexp(Exception, 'I always fail even with retry.'):
      Command().real_run_cmd(MyTask())

    task = MyTask()
    task.max_try_count = 3
    with self.assertRaisesRegexp(Exception, 'I always fail even with retry.'):
      Command().real_run_cmd(task)
    self.assertEquals(3, task.try_count)


class ProjectCommandTests(unittest.TestCase, CommonTestMixin):
  """Test projects commands."""

  EXPECTED_PROJECTS_CREATE = os.path.join(
      os.path.dirname(__file__), 'test_projects_create.txt')
  EXPECTED_PROJECTS_CREATE_PROJECT_EXISTS = os.path.join(
      os.path.dirname(__file__), 'test_projects_create_project_exists.txt')
  EXPECTED_PROJECTS_CREATE_PROJECT_EXISTS_NOREP = os.path.join(
      os.path.dirname(__file__),
      'test_projects_create_project_exists_norepr.txt')
  EXPECTED_PROJECTS_CREATE_VM_EXISTS = os.path.join(
      os.path.dirname(__file__), 'test_projects_create_vm_exists.txt')
  EXPECTED_PROJECTS_CREATE_LABELS = os.path.join(
      os.path.dirname(__file__), 'test_projects_create_labels.txt')
  EXPECTED_PROJECTS_CREATE_IMAGE = os.path.join(
      os.path.dirname(__file__), 'test_projects_create_image.txt')
  EXPECTED_PROJECTS_DELETE = os.path.join(
      os.path.dirname(__file__), 'test_projects_delete.txt')
  EXPECTED_PROJECTS_DELETE_MISSING = os.path.join(
      os.path.dirname(__file__), 'test_projects_delete_missing.txt')
  EXPECTED_PROJECTS_CONTENT_BUNDLE = os.path.join(
      os.path.dirname(__file__), 'test_projects_create_content_bundle.txt')

  MOCK_RESP_NO_PROJECTS_NO_VMS = {
      'gcloud projects describe my-prefix--student1examplecom '
      '--format value(lifecycleState)': (1, None),

      'gcloud compute instances describe '
      '--project my-prefix--student1examplecom '
      '--zone us-central1-a mlccvm-student1': (1, None),

      'gcloud projects describe my-prefix--student2examplecom '
      '--format value(lifecycleState)': (1, None),

      'gcloud compute instances describe '
      '--project my-prefix--student2examplecom '
      '--zone us-central1-a mlccvm-student2': (1, None),
  }

  MOCK_RESP_YES_PROJECTS_NO_VMS = {
      'gcloud projects describe my-prefix--student1examplecom '
      '--format value(lifecycleState)': (0, 'ACTIVE\n'),

      'gcloud compute instances describe '
      '--project my-prefix--student1examplecom '
      '--zone us-central1-a mlccvm-student1': (1, None),

      'gcloud projects describe my-prefix--student2examplecom '
      '--format value(lifecycleState)': (0, 'ACTIVE'),

      'gcloud compute instances describe '
      '--project my-prefix--student2examplecom '
      '--zone us-central1-a mlccvm-student2': (1, None),
  }

  MOCK_RESP_YES_PROJECTS_YES_VMS = {
      'gcloud projects describe my-prefix--student1examplecom '
      '--format value(lifecycleState)': (0, 'ACTIVE\n'),

      'gcloud compute instances describe '
      '--project my-prefix--student1examplecom '
      '--zone us-central1-a mlccvm-student1': (0, None),

      'gcloud projects describe my-prefix--student2examplecom '
      '--format value(lifecycleState)': (0, 'ACTIVE'),

      'gcloud compute instances describe '
      '--project my-prefix--student2examplecom '
      '--zone us-central1-a mlccvm-student2': (0, None),
  }

  MOCK_RESP_SOME_DELETED_PROJECTS = {
      'gcloud projects describe my-prefix--student1examplecom '
      '--format value(lifecycleState)': (0, 'SOME NON ACTOVE STATE'),

      'gcloud compute instances describe '
      '--project my-prefix--student1examplecom '
      '--zone us-central1-a mlccvm-student1': (1, None),

      'gcloud projects describe my-prefix--student2examplecom '
      '--format value(lifecycleState)': (1, None),

      'gcloud compute instances describe '
      '--project my-prefix--student2examplecom '
      '--zone us-central1-a mlccvm-student2': (1, None),
  }

  def _run(self, args):
    return Command().real_run_cmd(Task(None, args))

  def _assert_account_list(self, out):
    self.assertTrue(out)
    assert out.startswith('account-list-'), out

    with open(out[:-1], 'r') as stream:
      self.assertEquals([
          'student_email\tproject_id\tvm_name\tproject_url',
          'student1@example.com\tmy-prefix--student1examplecom\t'
          'mlccvm-student1\t'
          'https://console.cloud.google.com/home/dashboard?'
          'project=my-prefix--student1examplecom',
          'student2@example.com\tmy-prefix--student2examplecom\t'
          'mlccvm-student2\thttps://console.cloud.google.com/home/dashboard?'
          'project=my-prefix--student2examplecom',
          ''
      ], stream.read().split('\n'))
    os.remove(out[:-1])

  def test_no_args(self):
    with self.assertRaisesRegexp(Exception, 'error: too few arguments'):
      self._run(['python', self.GCLOUD_PY, '--no_tests'])

  def test_bad_action(self):
    with self.assertRaisesRegexp(Exception, 'invalid choice: \'bad_action\''):
      self._run(['python', self.GCLOUD_PY, 'bad_action', '--no_tests'])

  def test_projects_create(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_NO_PROJECTS_NO_VMS),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com'])

    self.assert_file_equals(err, self.EXPECTED_PROJECTS_CREATE)
    self._assert_account_list(out)

  def test_projects_create_project_exists(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_YES_PROJECTS_NO_VMS),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com',
        '--provision_vm'])

    self.assert_file_equals(err, self.EXPECTED_PROJECTS_CREATE_PROJECT_EXISTS)
    self._assert_account_list(out)

  def test_projects_create_project_exists_no_reprovision(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_YES_PROJECTS_NO_VMS),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com'])

    self.assert_file_equals(
        err, self.EXPECTED_PROJECTS_CREATE_PROJECT_EXISTS_NOREP)
    self._assert_account_list(out)

  def test_projects_create_vm_exists(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_YES_PROJECTS_YES_VMS),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com',
        '--provision_vm'])

    self.assert_file_equals(err, self.EXPECTED_PROJECTS_CREATE_VM_EXISTS)
    self._assert_account_list(out)

  def test_projects_create_with_labels(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_NO_PROJECTS_NO_VMS),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com',
        '--labels', 'foo=bar,alice=john'])

    self.assert_file_equals(err, self.EXPECTED_PROJECTS_CREATE_LABELS)
    self._assert_account_list(out)

  def test_projects_create_bad_image(self):
    self.maxDiff = None

    with self.assertRaisesRegexp(Exception,
                                 'Unsupported image name "bad_image_name".'):
      self._run([
          'python', self.GCLOUD_PY,
          'projects_create', '--no_tests', '--dry_run', '--serial',
          '--billing_id', '12345', '--prefix', 'my-prefix',
          '--owners', 'owner1@example.com owner2@example.com',
          '--students', 'student1@example.com student2@example.com',
          '--image_name', 'bad_image_name'])

  def test_projects_create_with_image(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_NO_PROJECTS_NO_VMS),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com',
        '--image_name', 'TF_RC0'])

    self.assert_file_equals(err, self.EXPECTED_PROJECTS_CREATE_IMAGE)
    self._assert_account_list(out)

  def test_projects_delete(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY, 'projects_delete', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_YES_PROJECTS_YES_VMS),
        '--prefix', 'my-prefix',
        '--students', 'student1@example.com student2@example.com'])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_PROJECTS_DELETE)

  def test_projects_delete_missing(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY, 'projects_delete', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_SOME_DELETED_PROJECTS),
        '--prefix', 'my-prefix',
        '--students', 'student1@example.com student2@example.com'])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_PROJECTS_DELETE_MISSING)

  def test_projects_create_concurrent(self):
    self.maxDiff = None

    expected = ['student_email\tproject_id\tvm_name\tproject_url']
    student_emails = []
    for index in xrange(0, 10):
      student_emails.append('student%s@example.com' % index)
      expected.append(
          'student%s@example.com\tmy-prefix--student%sexamplecom\t'
          'mlccvm-student%s\thttps://console.cloud.google.com/home/dashboard?'
          'project=my-prefix--student%sexamplecom' % (
              index, index, index, index))

    out, _ = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--billing_id', '12345',
        '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', ' '.join(student_emails)])

    self.assertTrue(out)
    assert out.startswith('account-list-'), out
    with open(out[:-1], 'r') as stream:
      self.assertEquals(expected + [''], stream.read().split('\n'))
    os.remove(out[:-1])

  def test_content_bundle(self):
    self.maxDiff = None

    mock_response_data = {
        'gcloud compute ssh mlccvm-student1 --project '
        'my-prefix--student1examplecom --zone us-central1-a '
        '--command docker ps --format "{{.ID}}\t{{.Image}}"': (
            0, 'ID\tIMAGE\nfoo1\t%s-bar1' % DOCKER_IMAGE_PREFIX),

        'gcloud compute ssh mlccvm-student2 --project '
        'my-prefix--student2examplecom --zone us-central1-a '
        '--command docker ps --format "{{.ID}}\t{{.Image}}"': (
            0, 'ID\tIMAGE\nfoo2\t%s-bar2' % DOCKER_IMAGE_PREFIX),
    }

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(mock_response_data),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com',
        '--content_bundle', os.path.join(
            os.path.dirname(__file__), 'test_content_bundle.tar.gz')])

    self.assert_file_equals(err, self.EXPECTED_PROJECTS_CONTENT_BUNDLE)
    self._assert_account_list(out)


class DatalabCommandTests(unittest.TestCase, CommonTestMixin):
  """Test datalab commands."""

  EXPECTED_DATALAB_CREATE = os.path.join(
      os.path.dirname(__file__), 'test_datalab_create.txt')
  EXPECTED_DATALAB_CREATE_EX = os.path.join(
      os.path.dirname(__file__), 'test_datalab_create_ex.txt')
  EXPECTED_DATALAB_CONNECT = os.path.join(
      os.path.dirname(__file__), 'test_datalab_connect.txt')
  EXPECTED_DATALAB_DELETE = os.path.join(
      os.path.dirname(__file__), 'test_datalab_delete.txt')

  def setUp(self):
    super(DatalabCommandTests, self).setUp()
    self.user = os.environ.get('USER', None)
    os.environ['USER'] = 'testuser'

  def tearDown(self):
    os.environ['USER'] = self.user
    super(DatalabCommandTests, self).tearDown()

  def _run(self, args):
    return Command().real_run_cmd(Task(None, args))

  def test_create_minimal_args(self):
    self.maxDiff = None

    mock_response_data = {
        'gcloud config get-value project': (0, 'my-sample-project-123'),
        'gcloud compute instances describe --project my-sample-project-123 '
        '--zone us-central1-a mlccvm-testuser': (1, None)}

    out, err = self._run([
        'python', self.GCLOUD_PY, 'datalab_create', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(mock_response_data),
        '--provision_vm'])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_DATALAB_CREATE)

  def test_create_bad_image_url(self):
    with self.assertRaisesRegexp(Exception,
                                 'Image URL must start with gcr.io/'):
      self._run([
          'python', self.GCLOUD_PY, 'datalab_create', '--no_tests', '--dry_run',
          '--image_url', 'foo/gcr.io/cloud-datalab/datalab:testimage'])

  def test_create_all_args(self):
    self.maxDiff = None

    mock_response_data = {
        'gcloud config get-value project': (0, 'my-sample-project-123\n')}

    out, err = self._run([
        'python', self.GCLOUD_PY, 'datalab_create', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(mock_response_data),
        '--provision_vm', '--zone', 'test-zone',
        '--image_url', 'gcr.io/cloud-datalab/datalab:testimage',
        '--content_bundle', os.path.join(
            os.path.dirname(__file__), 'test_content_bundle.tar.gz')])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_DATALAB_CREATE_EX)

  def test_connect(self):
    self.maxDiff = None

    mock_response_data = {
        'gcloud config get-value project': (0, 'my-sample-project-123\n'),
        'gcloud compute instances list --project my-sample-project-123 '
        '--limit 1 --filter name:mlccvm-testuser --format value(zone)': (
            0, 'testzone')}

    out, err = self._run([
        'python', self.GCLOUD_PY, 'datalab_connect', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(mock_response_data)])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_DATALAB_CONNECT)

  def test_connect_no_vm(self):
    mock_response_data = {
        'gcloud config get-value project': (0, 'my-sample-project-123\n'),
        'gcloud compute instances list --project my-sample-project-123 '
        '--limit 1 --filter name:mlccvm-testuser --format value(zone)': (
            0, None)}
    with self.assertRaisesRegexp(Exception,
                                 'Datalab VM mlccvm-testuser not found in '
                                 'project my-sample-project-123'):
      self._run([
          'python', self.GCLOUD_PY, 'datalab_connect', '--no_tests',
          '--dry_run', '--mock_gcloud_data', json.dumps(mock_response_data)])

  def test_delete(self):
    self.maxDiff = None

    mock_response_data = {
        'gcloud config get-value project': (0, 'my-sample-project-123\n'),
        'gcloud compute instances list --project my-sample-project-123 '
        '--limit 1 --filter name:mlccvm-testuser --format value(zone)': (
            0, 'testzone')}

    out, err = self._run([
        'python', self.GCLOUD_PY, 'datalab_delete', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(mock_response_data)])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_DATALAB_DELETE)


Command.register(DatalabCreate)
Command.register(DatalabConnect)
Command.register(DatalabDelete)
Command.register(ProjectsCreate)
Command.register(ProjectsDelete)
Command.register(RunTests)


Command.TESTS.append(CoreTests)
Command.TESTS.append(DatalabCommandTests)
Command.TESTS.append(ProjectCommandTests)
Command.TESTS.append(RetryTests)


if __name__ == '__main__':
  Command.dispatch()
