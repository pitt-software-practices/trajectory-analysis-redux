import pytest
from pytest import approx
import importlib.resources
import numpy as np
import traj_analysis
from traj_analysis import ClusterStats
# pytest will read the setup and teardown methods to structure a class-specific
# test environment, instantiating TrajStats only one time and checking outputs
class TestClassClusterStats():
# These tests REQUIRE that 'test.xyz' is in the 'traj_analysis' directory
# within the package directories. This first setup method is only called once
# per class and will instantiate TrajStats with the data in 'test.xyz'.
    def setup_class(self):
        rich_atomid = 1
        atomid = 2
        vacid = 3
        try:
            with importlib.resources.path(traj_analysis, 'test.xyz') as p:
                self.test = ClusterStats(str(p), atomid, rich_atomid, vacid)
        except:
            print("Did you put 'test.xyz' inside the package install folder 'traj_analysis'?\n")
            print("All tests require this file to initialize a class (in a test fixture function).\n")

    def test_clusterstats_initialization(self):
        correct_ids = [1, 2, 3]
        correct_name = "test.xyz"
        correct_parsplice_flag = False
        assert all([correct_ids == [self.test.rich_atomid, self.test.atomid, self.test.vacid],
                    self.test.filename.split('/')[-1] == correct_name,
                    self.test.parsplice == False])

    def test_partial_rdf_shape(self):
        total_pdf = self.test.calc_rdfs(cutoff = 5)
        correct_shape = (500, 7)
        assert total_pdf.shape == correct_shape

    def test_cutoff_range(self):
        correct_value = 4.995
        assert correct_value == approx(self.test.average_pdf[-1,0])

    def test_rdf_values(self):
        correct_values = [1.175, 1.249, 0.929, 1.196, 0.869, 0.0]
        assert all([correct_values[0] == approx(self.test.average_pdf[-1,1], 0.001),
                    correct_values[1] == approx(self.test.average_pdf[-1,2], 0.001),
                    correct_values[2] == approx(self.test.average_pdf[-1,3], 0.001),
                    correct_values[3] == approx(self.test.average_pdf[-1,4], 0.001),
                    correct_values[4] == approx(self.test.average_pdf[-1,5], 0.001),
                    correct_values[5] == approx(self.test.average_pdf[-1,6], 0.001)])

    def test_bond_averages(self):
        correct_rich_maxima = [2.545, 3.625, 4.435]
        correct_het_maxima = [2.505, 3.605, 4.435]
        correct_maxima = [1.935, 3.565, 4.125, 4.825]
        assert all([correct_rich_maxima == approx(self.test.bond_averages()['mean'][0]),
                    correct_het_maxima == approx(self.test.bond_averages()['mean'][1]),
                    correct_maxima == approx(self.test.bond_averages()['mean'][2])])
