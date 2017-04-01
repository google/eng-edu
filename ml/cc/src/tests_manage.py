#!/usr/bin/python
#
# Copyright 2017 Google Inc. All Rights Reserved.
#

"""Tests."""

__author__ = 'Pavel Simakov (psimakov@google.com)'


import json
import os
import unittest
import manage


def _abs_path(fn):
  """Returns path to FN, relative to __file__."""
  return os.path.join(os.path.dirname(__file__), fn)


def _rel_path(fn):
  """Returns path to FN, reative to the root of the project."""
  return os.path.join('./eng-edu/ml/cc/src', fn)


class CommonTestMixin(object):
  """A mixin that has test helper functions."""

  PREFIX = ' | Shell: ['
  GCLOUD_PY = _rel_path('manage.py')

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
          manage.to_unicode(expected.read()).split('\n'),
          self.filter_log(actual.split('\n')))


class CoreTests(unittest.TestCase):
  """Tests."""

  def test_project_id(self):
    project_id = manage.ProjectsCreate.project_id('foo', 'foo@gmail.com')
    self.assertEquals('foo--foogmailcom', project_id)

  def test_project_id_email_too_long(self):
    project_id = manage.ProjectsCreate.project_id(
        'foo', '123456789012345678901234567890foo@gmail.com')
    self.assertEquals('foo--123456789012345678901234', project_id)

  def test_vm_name(self):
    vm_name = manage.ProjectsCreate.vm_name('foo@gmail.com')
    self.assertEquals('mlccvm-foo', vm_name)


class RetryTests(unittest.TestCase):
  """Test task retires."""

  def test_not_retriable_is_not_retried(self):

    class MyTask(manage.Task):

      def _run_once(self):
        raise Exception('I always fail.')

    with self.assertRaisesRegexp(Exception, 'I always fail.'):
      manage.Command().real_run_cmd(MyTask(None, []))

  def test_retriable_is_retried(self):

    class MyTask(manage.Task):

      def __init__(self):
        super(MyTask, self).__init__(None, [])
        self.try_count = 0

      def _run_once(self):
        self.try_count += 1
        if self.try_count == 1:
          raise Exception('I always fail once.')

    with self.assertRaisesRegexp(Exception, 'I always fail once.'):
      manage.Command().real_run_cmd(MyTask())

    task = MyTask()
    task.max_try_count = 3
    manage.Command().real_run_cmd(task)
    self.assertEquals(2, task.try_count)

  def test_max_try_count_is_respected(self):

    class MyTask(manage.Task):

      def __init__(self):
        super(MyTask, self).__init__(None, [])
        self.try_count = 0

      def _run_once(self):
        self.try_count += 1
        raise Exception('I always fail even with retry.')

    with self.assertRaisesRegexp(Exception, 'I always fail even with retry.'):
      manage.Command().real_run_cmd(MyTask())

    task = MyTask()
    task.max_try_count = 3
    with self.assertRaisesRegexp(Exception, 'I always fail even with retry.'):
      manage.Command().real_run_cmd(task)
    self.assertEquals(3, task.try_count)


class ProjectCommandTests(unittest.TestCase, CommonTestMixin):
  """Test projects commands."""

  EXPECTED_PROJECTS_CREATE = _abs_path(
      'test_projects_create.txt')
  EXPECTED_PROJECTS_CREATE_PROJECT_EXISTS = _abs_path(
      'test_projects_create_project_exists.txt')
  EXPECTED_PROJECTS_CREATE_PROJECT_EXISTS_NOREP = _abs_path(
      'test_projects_create_project_exists_norepr.txt')
  EXPECTED_PROJECTS_CREATE_VM_EXISTS = _abs_path(
      'test_projects_create_vm_exists.txt')
  EXPECTED_PROJECTS_CREATE_LABELS = _abs_path(
      'test_projects_create_labels.txt')
  EXPECTED_PROJECTS_CREATE_IMAGE = _abs_path(
      'test_projects_create_image.txt')
  EXPECTED_PROJECTS_DELETE = _abs_path(
      'test_projects_delete.txt')
  EXPECTED_PROJECTS_DELETE_BY_ID = _abs_path(
      'test_projects_delete_by_id.txt')
  EXPECTED_PROJECTS_DELETE_MISSING = _abs_path(
      'test_projects_delete_missing.txt')
  EXPECTED_PROJECTS_CONTENT_BUNDLE = _abs_path(
      'test_projects_create_content_bundle.txt')

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
    return manage.Command().real_run_cmd(manage.Task(None, args))

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

  def test_delete_no_args(self):
    with self.assertRaisesRegexp(Exception,
                                 'Please provide --student_emails or '
                                 '--project_ids.'):
      self._run([
          'python', self.GCLOUD_PY, 'projects_delete', '--no_tests',
          '--dry_run', '--prefix', 'foo'])

  def test_delete_too_many_args(self):
    with self.assertRaisesRegexp(Exception,
                                 'Please provide --student_emails or '
                                 '--project_ids, not both.'):
      self._run([
          'python', self.GCLOUD_PY, 'projects_delete', '--no_tests',
          '--dry_run', '--prefix', 'foo',
          '--students', 'student1@example.com student2@example.com',
          '--project_ids', 'id1 id2',
      ])

  def test_delete_does_not_need_prefix(self):
    with self.assertRaisesRegexp(Exception,
                                 'The --prefix can\'t be specified when '
                                 'providing --project_ids.'):
      self._run([
          'python', self.GCLOUD_PY, 'projects_delete', '--no_tests',
          '--dry_run', '--prefix', 'foo',
          '--project_ids', 'id1 id2',
      ])

  def test_delete_needs_prefix(self):
    with self.assertRaisesRegexp(Exception,
                                 'Please provide --prefix when providing '
                                 '--student_emails.'):
      self._run([
          'python', self.GCLOUD_PY, 'projects_delete', '--no_tests',
          '--dry_run',
          '--students', 'student1@example.com student2@example.com',
      ])

  def test_projects_delete_by_student_emails(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY, 'projects_delete', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_YES_PROJECTS_YES_VMS),
        '--prefix', 'my-prefix',
        '--students', 'student1@example.com student2@example.com'])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_PROJECTS_DELETE)

  def test_projects_delete_by_project_ids(self):
    self.maxDiff = None

    out, err = self._run([
        'python', self.GCLOUD_PY, 'projects_delete', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(self.MOCK_RESP_YES_PROJECTS_YES_VMS),
        '--project_ids',
        'my-prefix--student1examplecom my-prefix--student2examplecom'])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_PROJECTS_DELETE_BY_ID)

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
            0, 'ID\tIMAGE\nfoo1\t%s-bar1' % manage.DOCKER_IMAGE_PREFIX),

        'gcloud compute ssh mlccvm-student2 --project '
        'my-prefix--student2examplecom --zone us-central1-a '
        '--command docker ps --format "{{.ID}}\t{{.Image}}"': (
            0, 'ID\tIMAGE\nfoo2\t%s-bar2' % manage.DOCKER_IMAGE_PREFIX),
    }

    out, err = self._run([
        'python', self.GCLOUD_PY,
        'projects_create', '--no_tests', '--dry_run', '--serial',
        '--mock_gcloud_data', json.dumps(mock_response_data),
        '--billing_id', '12345', '--prefix', 'my-prefix',
        '--owners', 'owner1@example.com owner2@example.com',
        '--students', 'student1@example.com student2@example.com',
        '--content_bundle', _rel_path('test_content_bundle.tar.gz')])

    self.assert_file_equals(err, self.EXPECTED_PROJECTS_CONTENT_BUNDLE)
    self._assert_account_list(out)


class DatalabCommandTests(unittest.TestCase, CommonTestMixin):
  """Test datalab commands."""

  EXPECTED_DATALAB_CREATE = _abs_path(
      'test_datalab_create.txt')
  EXPECTED_DATALAB_CREATE_EX = _abs_path(
      'test_datalab_create_ex.txt')
  EXPECTED_DATALAB_CREATE_VALIDATE = _abs_path(
      'test_datalab_create_validate.txt')
  EXPECTED_DATALAB_CONNECT = _abs_path(
      'test_datalab_connect.txt')
  EXPECTED_DATALAB_DELETE = _abs_path(
      'test_datalab_delete.txt')
  EXPECTED_BUNDLE_TEST = _abs_path(
      'test_bundle_test.txt')

  EXPECTED_BUNDLE_TEST_OUTPUT = '''{
    "args": {
        "action": "bundle_test", 
        "bundle_home": "som/folder", 
        "dry_run": true, 
        "gcloud_bin": "gcloud", 
        "mock_gcloud_data": "{\\"iterate_notebooks\\": [\\"foo\\", \\"bar\\"]}", 
        "no_tests": true, 
        "serial": true, 
        "verbose": false
    }, 
    "command": "BundleTest", 
    "data": [
        {
            "cmd": [
                "jupyter", 
                "nbconvert", 
                "--to", 
                "notebook", 
                "--execute", 
                "--output", 
                "bar.new.ipynb", 
                "--ExecutePreprocessor.timeout=-1", 
                "bar"
            ], 
            "error_code": 0, 
            "notebook": "bar", 
            "result": null, 
            "stderr": null, 
            "stdout": null
        }, 
        {
            "cmd": [
                "jupyter", 
                "nbconvert", 
                "--to", 
                "notebook", 
                "--execute", 
                "--output", 
                "foo.new.ipynb", 
                "--ExecutePreprocessor.timeout=-1", 
                "foo"
            ], 
            "error_code": 0, 
            "notebook": "foo", 
            "result": null, 
            "stderr": null, 
            "stdout": null
        }
    ]
}'''

  def setUp(self):
    super(DatalabCommandTests, self).setUp()
    self.user = os.environ.get('USER', None)
    os.environ['USER'] = 'testuser'

  def tearDown(self):
    os.environ['USER'] = self.user
    super(DatalabCommandTests, self).tearDown()

  def _run(self, args):
    return manage.Command().real_run_cmd(manage.Task(None, args))

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
        '--content_bundle', _rel_path('test_content_bundle.tar.gz')])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_DATALAB_CREATE_EX)

  def test_create_validate_content_bundle_args(self):
    self.maxDiff = None

    mock_response_data = {
        'gcloud config get-value project': (0, 'my-sample-project-123\n')}

    out, err = self._run([
        'python', self.GCLOUD_PY, 'datalab_create', '--no_tests', '--dry_run',
        '--mock_gcloud_data', json.dumps(mock_response_data),
        '--provision_vm', '--zone', 'test-zone',
        '--image_url', 'gcr.io/cloud-datalab/datalab:testimage',
        '--validate_content_bundle', '--content_bundle',
        _rel_path('test_content_bundle.tar.gz')])

    self.assertFalse(out)
    self.assert_file_equals(err, self.EXPECTED_DATALAB_CREATE_VALIDATE)

  def test_bundle_test(self):
    self.maxDiff = None

    mock_response_data = {'iterate_notebooks': ['foo', 'bar']}

    out, err = self._run([
        'python', self.GCLOUD_PY, 'bundle_test', '--no_tests', '--dry_run',
        '--serial', '--mock_gcloud_data', json.dumps(mock_response_data),
        '--bundle_home', 'som/folder'])

    self.assertTrue(out)
    data = json.loads(out)
    self.assertMultiLineEqual(self.EXPECTED_BUNDLE_TEST_OUTPUT,
                              json.dumps(data, sort_keys=True, indent=4))
    self.assert_file_equals(err, self.EXPECTED_BUNDLE_TEST)

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


class TestSuite(object):
  """A collection of tests."""

  TESTS = []

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


TestSuite.TESTS.append(CoreTests)
TestSuite.TESTS.append(DatalabCommandTests)
TestSuite.TESTS.append(ProjectCommandTests)
TestSuite.TESTS.append(RetryTests)


if __name__ == '__main__':
  TestSuite.run_all_unit_tests()
