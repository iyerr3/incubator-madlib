#!/usr/env python

import re
from collections import namedtuple


"""
Example of JUnit output:

<?xml version="1.0" encoding="UTF-8"?>
<testsuite tests="3">
    <testcase classname="foo1" name="ASuccessfulTest"/>
    <testcase classname="foo2" name="AnotherSuccessfulTest"/>
    <testcase classname="foo3" name="AFailingTest">
        <failure type="NotEnoughFoo"> details about failure </failure>
    </testcase>
</testsuite>
"""


TestResult = namedtuple("TestResult", 'name suite status duration')


def _test_result_factory(install_check_log):
    """
    Args:
        @param install_

    Returns:
        Next result of type test_result
    """
    with open(install_check_log, 'r') as ic_log:
        for line in ic_log:
            m = re.match(r"^TEST CASE RESULT\|Module: (.*)\|(.*)\|(.*)\|Time: ([0-9]+)(.*)", line)
            if m:
                yield TestResult(name=m.group(2), suite=m.group(1),
                                 status=m.group(3), duration=m.group(4))
# ----------------------------------------------------------------------


def _add_header(out_log, n_tests):
    header = ['<?xml version="1.0" encoding="UTF-8"?>',
              '<testsuite tests="{0}">'.format(n_tests), '']
    out_log.write('\n'.join(header))


def _add_footer(out_log):
    header = ['', '</testsuite>']
    out_log.write('\n'.join(header))


def _add_test_case(out_log, test_results):
    for res in test_results:
        output = [('<testcase classname="{t.suite}" name="{t.name}" '
                   'status="{t.status}" time="{t.duration}">'.
                   format(t=res))]
        output.append('</testcase>')
        out_log.write('\n'.join(output))


def main(install_check_log, test_output_log):

    # need number of test results - so have to create the iterable
    all_test_results = [i for i in _test_result_factory(install_check_log)]

    with open(test_output_log, 'w') as out_log:
        _add_header(out_log, len(all_test_results))
        _add_test_case(out_log, all_test_results)
        _add_footer(out_log)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
