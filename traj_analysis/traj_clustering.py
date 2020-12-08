import sys
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
from scipy.signal import argrelextrema

def bondstats(average_rdf, std_rdf, plot_number):
    """ Find average peaks in binned RDF and their standard deviations using scipy argrelextrema

    Parameters
    -----------
    average_rdf: (nbins,7) ndarray
    std_rdf: (nbins,7) ndarray
    plot_number: int determined by OVITO GUI for bimetallic system w/ vacancies

    Return
    ----------
    maxima: (1,3) ndarray with max peaks in average_rdf
    std_rdf: (1,3) ndarray with standard dev of the max peaks
    """
    # check if value is greater than 35 entries in either direction (order)
    index = argrelextrema(average_rdf[:, plot_number], np.greater, order = 35)[0]
    maxima = average_rdf[:,0][index]
    maxima_std = std_rdf[:,0][index]
    return (maxima, maxima_std)

class ClusterStats():
    """ Cluster Statistics
    Takes LAMMPS MC/NVT data, calculating the time-averaged partial radial
    distribution functions (partial RDFs) for all the bond types in the system.

    Parameters:
    ___________
    filename: OVITO readable file such as .lmp or .xyz (ideally preprocessed by the user)
    atomid: integer label for the dopant atom type in the file
    rich_atomid: integer label for the rich component in the file
    vacid: integer label for the vacancy 'atom type'
    parsplice: boolean flag for different time resolutions
    """

    def __init__(self, filename, atomid, rich_atomid, vacid, parsplice = False):

        self.filename = filename
        self.atomid = atomid
        self.rich_atomid = rich_atomid
        self.vacid = vacid
        self.parsplice = parsplice
        self.pipeline = import_file(filename, sort_particles = True)

    # calculate distribution functions with OVITO
    def calc_rdfs(self, cutoff):
        self.cutoff = cutoff
        # Modifier to calculate distribution functions
        if self.parsplice == False:
            modifier = CoordinationAnalysisModifier(cutoff = self.cutoff, number_of_bins = 500, partial = True)
            number_of_bins = 500
        else:
            modifier = CoordinationAnalysisModifier(cutoff = self.cutoff, number_of_bins = 1000, partial = True)
            number_of_bins = 1000
        self.pipeline.modifiers.append(modifier)
        # storage for all pdfs in all frames
        #self.average_pdf = np.zeros((modifier.number_of_bins,7))
        data = self.pipeline.compute(0)
        self.total_pdf = np.reshape(data.tables['coordination-rdf'].xy(), (number_of_bins, 7, 1))
        for frame in range(1,self.pipeline.source.num_frames):
            data = self.pipeline.compute(frame)
            self.total_pdf = np.append(self.total_pdf, np.reshape(data.tables['coordination-rdf'].xy(), (number_of_bins, 7, 1)), axis = 2)

        self.average_pdf = np.average(self.total_pdf, axis = 2)
        self.std_pdf = np.std(self.total_pdf, axis = 2)

        return self.average_pdf

    # plot the distributions
    def plot_rdfs(self, rich = 'Cu', dilute = 'Ni'):
        """ Plot radial distribution functions for the atom types

        Parameters
        -----------
        rich: (optional) string with rich-phase atom type
        dilute: (optional) string with dilute-phase atom type
        """
        fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
        # heterobond
        axs[1].errorbar(self.average_pdf[:,0], self.average_pdf[:,4], self.std_pdf[:,4],  fmt = 'o--', markersize = 4, capsize = 4)
        # different order of data indexing depending on the order of atom types
        if self.rich_atomid == 1 and self.atomid == 2:
            # rich atomid first
            axs[0].errorbar(self.average_pdf[:,0], self.average_pdf[:, 1], self.std_pdf[:,1], fmt = 'o--', markersize = 4, capsize = 4)
            axs[0].set_title(rich + '-' + rich + ' bond')
            # heterobond
            axs[1].set_title(rich + '-' + dilute + ' bond')
            # atomid
            axs[2].errorbar(self.average_pdf[:,0], self.average_pdf[:,5], self.std_pdf[:,5],  fmt = 'o--', markersize = 4, capsize = 4)
            axs[2].set_title(dilute + '-' + dilute + ' bond')
        # in case the order is swapped
        elif self.atomid == 1 and self.rich_atomid == 2:
            # atomid first in this case
            axs[0].errorbar(self.average_pdf[:,0], self.average_pdf[:,1], self.std_pdf[:,1],  fmt = 'o--', markersize = 4, capsize = 4)
            axs[0].set_title(dilute + '-' + dilute + ' bond')
            # heterobond
            axs[1].set_title(dilute + '-' + rich + ' bond')
            # rich_atomid
            axs[2].errorbar(self.average_pdf[:,0], self.average_pdf[:,5], self.std_pdf[:,5], fmt = 'o--', markersize = 4, capsize = 4)
            axs[2].set_title(rich + '-' + rich + ' bond')
        else:
            print("Error: Inconsistent particle type IDs")
            sys.exit(1)
        # for pretty labeling, make a large, invisible subplot with the axis labels only
        fig.add_subplot(111, frameon = False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.title('Average Partial Distribution Functions\n')
        plt.xlabel('Pair Separation Distance (angstroms)')
        plt.ylabel('Average g(r)')

    # plot the RDFs with respect to the lattice vacancies
    def plot_vacrdfs(self, rich = 'Cu', dilute = 'Ni'):
        """ Plot radial distribution functions for the vacancy distances

        Parameters
        -----------
        rich: (optional) string with rich-phase atom type
        dilute: (optional) string with dilute-phase atom type
        """
        if self.vacid == 3:
            if self.rich_atomid == 1 and self.atomid == 2:
                fig, axs = plt.subplots(1,3, sharex = True)
                # rich_atomid
                axs[0].errorbar(self.average_pdf[:,0], self.average_pdf[:,2], self.std_pdf[:,2], fmt = 'o--', markersize = 4, capsize = 4)
                axs[0].set_title(rich + '-' + 'vacancy distance')
                # atomid
                axs[1].errobar(self.average_pdf[:,0], self.average_pdf[:,3], self.std_pdf[:,3], fmt = 'o--', markersize = 4, capsize = 4)
                axs[1].set_title(dilute + '-' + 'vacancy distance')
            # in case the order is swapped
            elif self.atomid == 1 and self.rich_atomid == 2:
                fig, axs = plt.subplots(1,3, sharex = True)
                # atomid
                axs[0].errorbar(self.average_pdf[:,0], self.average_pdf[:,2], self.std_pdf[:,2], fmt = 'o--', markersize = 4, capsize = 4)
                axs[0].set_title(dilute + '-' + 'vacancy distance')
                # rich_atomid
                axs[1].errorbar(self.average_pdf[:,0], self.average_pdf[:,3], self.std_pdf[:,3], fmt = 'o--', markersize = 4, capsize = 4)
                axs[1].set_title(rich + '-' + 'vacancy distance')
            else:
                print('Error: Inconsistent particle type IDs')
                sys.exit(1)
            # vacancy-vacancy
            axs[2].errorbar(self.average_pdf[:,0], self.average_pdf[:,6], self.std_pdf[:,6],  fmt = 'o--', markersize = 4, capsize = 4)
            axs[2].set_title('Vacancy-vacancy distance')
            fig.tight_layout()
        else:
            print("Error: Vacid must be equal to 3")
            sys.exit(1)

    # report local maxima in the RDFs
    def bond_averages(self):
        """ Uses scipy function to find the local maxima in the RDF graphs for
        nearest and next-nearest neighbors

        Returns
        ---------
        list: 3 numpy arrays for rich RDF maxima, heterobond maxima, and dilute maxima
        """
        if self.vacid == 3:
            # heterobond
            self.het_maxima, self.het_maxima_std = bondstats(self.average_pdf, self.std_pdf, 4)
            if self.rich_atomid == 1 and self.atomid == 2:
                # bonds
                self.rich_maxima, self.rich_maxima_std = bondstats(self.average_pdf, self.std_pdf, 1)
                self.maxima, self.maxima_std = bondstats(self.average_pdf, self.std_pdf, 5)
                # distances to vacancies
                self.dilute_vac_maxima, self.dilute_vac_maxima_std = bondstats(self.average_pdf, self.std_pdf, 3)
                self.rich_vac_maxima, self.rich_vac_maxima_std = bondstats(self.average_pdf, self.std_pdf,2)
            # in case the order is swapped
            elif self.atomid == 1 and self.rich_atomid == 2:
                self.rich_maxima, self.rich_maxima_std = bondstats(self.average_pdf, self.std_pdf, 5)
                self.maxima, self.maxima_std = bondstats(self.average_pdf, self.std_pdf, 1)
                self.dilute_vac_maxima, self.dilute_vac_maxima_std = bondstats(self.average_pdf, self.std_pdf, 2)
                self.rich_vac_maxima, self.rich_vac_maxima_std = bondstats(self.average_pdf, self.std_pdf, 3)
            else:
                print("Error: Inconsistent particle type IDs")
                sys.exit(1)
            self.vac_vac_maxima, self.vac_vac_maxima_std = bondstats(self.average_pdf, self.std_pdf, 6)
        else:
            print("Error: Vacid must be equal to 3")
            sys.exit(1)
        self.bondstats = {'mean': [self.rich_maxima, self.het_maxima, self.maxima, self.rich_vac_maxima, self.dilute_vac_maxima, self.vac_vac_maxima],
                        'std': [self.rich_maxima_std, self.het_maxima_std, self.maxima_std, self.rich_vac_maxima_std, self.dilute_vac_maxima_std, self.vac_vac_maxima_std]}
        return self.bondstats
