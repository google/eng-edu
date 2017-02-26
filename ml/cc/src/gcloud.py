#!/usr/bin/python
#
# Copyright 2017 Google Inc. All Rights Reserved.
#

"""Classes for automation and management of Google Cloud Platform tasks."""

__author__ = 'Pavel Simakov (psimakov@google.com)'


import argparse
import copy_reg
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import time
import types
import unittest


MAX_STUDENTS = 40
MAX_OWNERS = 10


logging.basicConfig()
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


# permits serialization of instance methods for multiprocessing
copy_reg.pickle(types.MethodType, _pickle_method)


class Task(object):
  """An atomic task."""

  def __init__(self, caption, args):
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
      LOG.error('Command failed: %s; %s', self.args, e)
      raise
    stdout, stderr = child.communicate()
    if child.returncode:
      raise Exception('Command failed with an error code %s: %s\n(%s)\n(%s)' % (
          child.returncode, self.args, to_unicode(stderr), to_unicode(stdout)))
    return to_unicode(stdout), to_unicode(stderr)


class Command(object):
  """An executable command."""

  NAME = None
  COMMANDS = {}
  TESTS = []

  @classmethod
  def dry_run_cmd(cls, task):
    """Runs a task as fake shell command, returns no result."""
    if task.caption:
      LOG.info(task.caption)
    LOG.info('Dry run: %s', task.args)

  @classmethod
  def real_run_cmd(cls, task):
    """Runs a task as a real shell command, checks result and returns output."""
    if task.caption:
      LOG.info(task.caption)

    try_count = 0
    while True:
      try:
        return task._run_once()
      except Exception as e:  # pylint: disable=broad-except
        try_count += 1
        if task.max_try_count <= try_count:
          LOG.warning('Command failed permanently after %s retries: %s; %s',
                      try_count, task.args, e)
          raise
        LOG.warning('Retrying failed command: %s; %s', task.args, e)

  @classmethod
  def run(cls, task):
    """Executes a task."""
    raise NotImplementedError()

  def make_parser(self):
    """Returns args parser for this command."""
    raise NotImplementedError()

  def execute(self):
    """Executes the command."""
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
        '--serial', help='Whether to disable concurrent task execution.',
        action='store_true')
    parser.add_argument(
        '--no_tests', help='Whether to skip running tests.',
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

    Command.run = (
        Command.dry_run_cmd if cmd_args.dry_run else Command.real_run_cmd)

    LOG.info('Started %s', cmd_args.action)
    if cmd_args.serial:
      LOG.info('Executing serially')
    else:
      LOG.info('Executing concurrently')
    cmd.execute(cmd_args)
    LOG.info('Completed %s', cmd_args.action)


class ProjectsCreate(Command):
  """An action that creates projects in bulk."""

  NAME = 'projects_create'
  CREATE_PROJECTS_RESULTS_FN = 'account-list-%s.csv' % int(time.time())

  def __init__(self):
    self.billing_id = None
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
        '--labels', help='A comma-separated list of project labels '
        '(i.e. "foo=bar,alice=bob").')
    parser.add_argument(
        '--owners', help='A space-separated list of owner emails, including '
        'your email as well as those of Teaching Assistants . Owners will be '
        'given a role of OWNER in all student projects.', required=True)
    parser.add_argument(
        '--zone', help='A name of the Google Cloud Platform zone to host '
        'your resources.', default='us-central1-a')
    return parser

  def _parse_args(self, args):
    self.billing_id = args.billing_id
    self.labels = args.labels
    self.owner_emails = args.owners.lower().split(' ')
    self.student_emails = args.students.lower().split(' ')
    self.prefix = args.prefix
    self.zone = args.zone
    if len(self.owner_emails) > MAX_OWNERS:
      raise Exception('Too many owners: %s max.' % MAX_OWNERS)
    if len(self.student_emails) > MAX_STUDENTS:
      raise Exception('Too many students: %s max.' % MAX_STUDENTS)

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
    vm_id = re.sub(r'[^A-Za-z0-9]+', '', parts[0])
    return 'mlccvm-%s' % vm_id

  @classmethod
  def run_common_tasks(cls):
    update_comp = Task('Updating components', [
        'sudo', 'gcloud', '--quiet', 'components', 'update'])
    cls.run(update_comp)
    install_comp = Task('Installing components: alpha, datalab', [
        'sudo', 'gcloud', '--quiet', 'components', 'install', 'alpha',
        'datalab'])
    cls.run(install_comp)

  def _create_project(self, student_email):
    """Creates one student project."""
    project_id = self.project_id(self.prefix, student_email)
    vm_name = self.vm_name(student_email)

    create_project = Task('Creating project %s for student %s' % (
        project_id, student_email), [
            'gcloud', 'alpha', 'projects', 'create', project_id])
    if self.labels:
      create_project.append(['--labels', self.labels])
    self.run(create_project)

    wait_2_s = Task('Waiting for project to fully materialize', [
        'sleep', 2])
    self.run(wait_2_s)

    for owner_email in self.owner_emails:
      add_owner_role = Task('Adding %s as owner' % owner_email, [
          'gcloud', 'projects', 'add-iam-policy-binding', project_id,
          '--member', 'user:%s' % owner_email, '--role', 'roles/owner'])
      self.run(add_owner_role)

    add_student_role = Task('Adding %s as student' % student_email, [
        'gcloud', 'projects', 'add-iam-policy-binding', project_id,
        '--member', 'user:%s' % student_email, '--role', 'roles/editor'])
    self.run(add_student_role)

    enable_billing = Task('Enabling Billing', [
        'gcloud', 'alpha', 'billing', 'accounts', 'projects', 'link',
        project_id, '--account-id', self.billing_id])
    self.run(enable_billing)

    enable_compute = Task('Enabling Compute API', [
        'gcloud', 'service-management', 'enable', 'compute_component',
        '--project', project_id])
    enable_compute.max_try_count = 3
    self.run(enable_compute)

    enable_ml = Task('Enabling ML API', [
        'gcloud', 'service-management', 'enable', 'ml.googleapis.com',
        '--project', project_id])
    enable_ml.max_try_count = 3
    self.run(enable_ml)

    provision_datalab = Task('Provisioning datalab', [
        'datalab', 'create', vm_name, '--for-user', student_email,
        '--project', project_id, '--zone', self.zone, '--no-connect'])
    self.run(provision_datalab)

    return student_email, project_id, vm_name, self._project_home(project_id)

  def execute(self, args):
    """Creates projects in bulk."""
    self._parse_args(args)
    self.run_common_tasks()
    LOG.info('Creating Datalab VM projects for %s students and %s owners',
             len(self.student_emails), len(self.owner_emails))

    if args.serial:
      rows = []
      for student_email in self.student_emails:
        row = self._create_project(student_email)
        rows.append(row)
    else:
      pool = multiprocessing.Pool()
      rows = pool.map(self._create_project, self.student_emails)

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

  def execute(self, args):
    """Deletes projects in bulk."""
    self._parse_args(args)
    LOG.info('Deleting Datalab VM projects for %s students',
             len(self.student_emails))
    ProjectsCreate.run_common_tasks()
    for student_email in self.student_emails:
      project_id = ProjectsCreate.project_id(self.prefix, student_email)
      delete_project = Task('Deleting project %s' % project_id, [
          'gcloud', '--quiet', 'alpha', 'projects', 'delete', project_id])
      self.run(delete_project)


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


class ArgsTests(unittest.TestCase):
  """Tests."""

  PREFIX = 'INFO:__main__:Dry run: ['

  GCLOUD_PY = os.path.join(os.path.dirname(__file__), 'gcloud.py')
  EXPECTED_PROJECTS_CREATE = os.path.join(
      os.path.dirname(__file__), 'test_projects_create.txt')
  EXPECTED_PROJECTS_CREATE_LABELS = os.path.join(
      os.path.dirname(__file__), 'test_projects_create_labels.txt')
  EXPECTED_PROJECTS_DELETE = os.path.join(
      os.path.dirname(__file__), 'test_projects_delete.txt')

  def _run(self, args):
    return Command().real_run_cmd(Task(None, args))

  def _assert_file_equals(self, actual, fn):
    with open(fn, 'r') as expected:
      self.assertEquals(
          to_unicode(expected.read()).split('\n'),
          self.filter_log(actual.split('\n')))

  def filter_log(self, items):
    """Extracts only the log lines that contain shell commands."""
    results = []
    for item in items:
      if item.startswith(self.PREFIX) and item.endswith(']'):
        results.append(item[len(self.PREFIX): -1])
    results.append(u'')  # to match the new line at the end of the data file
    return results

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
        '--billing_id', '12345',
        '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com'])

    self._assert_file_equals(err, self.EXPECTED_PROJECTS_CREATE)

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

  def test_projects_create_with_labels(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--billing_id', '12345',
        '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com',
        '--labels', 'foo=bar,alice=john'])

    self._assert_file_equals(err, self.EXPECTED_PROJECTS_CREATE_LABELS)

    self.assertTrue(out)
    assert out.startswith('account-list-'), out
    with open(out[:-1], 'r') as stream:
      self.assertEquals([
          'student_email\tproject_id\tvm_name\tproject_url',
          'student1@example.com\tmy-prefix--student1examplecom\t'
          'mlccvm-student1\thttps://console.cloud.google.com/home/dashboard?'
          'project=my-prefix--student1examplecom',
          'student2@example.com\tmy-prefix--student2examplecom\t'
          'mlccvm-student2\thttps://console.cloud.google.com/home/dashboard?'
          'project=my-prefix--student2examplecom',
          ''], stream.read().split('\n'))

  def test_projects_delete(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_delete', '--no_tests', '--dry_run',
        '--prefix', 'my-prefix',
        '--students', 'student1@example.com student2@example.com'])
    self.assertFalse(out)
    self._assert_file_equals(err, self.EXPECTED_PROJECTS_DELETE)

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


class RetryTests(unittest.TestCase):
  """Tests."""

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


Command.register(ProjectsCreate)
Command.register(ProjectsDelete)

Command.TESTS.append(ArgsTests)
Command.TESTS.append(CoreTests)
Command.TESTS.append(RetryTests)


if __name__ == '__main__':
  Command.dispatch()
