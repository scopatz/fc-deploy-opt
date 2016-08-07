#!/usr/bin/env xonsh
"""Diffs the latex file and builds the pdf."""
import os
import sys
from argparse import ArgumentParser


def replace_inputs(diffname, files):
    with open(diffname, 'r') as fh:
        s = fh.read()
    for f in files:
        fbase, _ = os.path.splitext(f)
        s = s.replace('\\input{' + fbase + '}', '\\input{' + fbase + '-diff}')
    with open(diffname, 'w') as fh:
        fh.write(s)


def difftex(old):
    files = set(`.*?\.tex`) - set(`.*?-diff\.tex`)
    for f in files:
        print('diffing ' + f)
        fbase, fext = os.path.splitext(f)
        oldspec = old + ':' + f
        oldname = '/tmp/{0}-{1}{2}'.format(fbase, old, fext)
        diffname = '{0}-diff{1}'.format(fbase, fext)
        git show @(oldspec) > @(oldname)
        latexdiff @(oldname) @(f) > @(diffname)
        replace_inputs(diffname, files)


def main(args=None):
    parser = ArgumentParser('diff')
    parser.add_argument('old', help='Tree to compare against.')
    parser.add_argument('--manuscript', help='Diffed manuscript name', 
                        default='paper-diff', dest='manuscript')
    ns = parser.parse_args(args=args or $ARGS[1:])

    difftex(ns.old)
    make @('manuscript=' + ns.manuscript)
    

if __name__ == '__main__':
    main()
