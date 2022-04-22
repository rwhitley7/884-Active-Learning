#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import kmeans_pytorch2


class UnitTests(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(kmeans_pytorch2)

    def test_project(self):
        self.assertTrue(False, "write more tests here")